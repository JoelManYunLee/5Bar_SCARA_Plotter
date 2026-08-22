/**
 * plotter_server — ESP32 Wi-Fi motor server for the 5-Bar SCARA plotter
 *
 * Hosts the HTTP endpoint the Flask server (server/app.py + process.py) talks to
 * during calibration. Each arm is jogged a fixed amount; the moment that arm's
 * limit switch trips, the motor stops and its angle becomes known (locked to a
 * reference), and absolute step tracking begins from there. Positions survive
 * power cycles via NVS, so the machine stays calibrated between sessions.
 *
 * Once both arms are homed, /plot is accepted. Travelling (pen up) to the
 * drawing rectangle's bottom-left corner (0,0) is a separate, explicit action —
 * POST /home from the server's "return home" button — run asynchronously, so
 * /position keeps reporting live angles the whole way. POST /goto drives the
 * same async travel to an arbitrary point, for the server's XY jog joystick.
 *
 *   POST /motor   {motor:"A"|"B", direction:"cw"|"ccw", degrees:<n>}
 *                 → jog; stop early if the switch engages
 *                 ← {ok:true, limit:<bool>, homed:<bool>, position_deg:<n|null>}
 *   POST /home    (only after both arms homed)
 *                 → travel (pen up) to the drawing area's (0,0) corner
 *                 ← {ok:true}  (202; travel runs async)
 *   POST /goto    {x,y}  (only after both arms homed)
 *                 → travel (pen up) to an arbitrary drawing-space point; a
 *                   request while already travelling retargets it live
 *                 ← {ok:true}  (202; travel runs async)
 *   POST /plot    {width,height,paths,weights}  (only after both arms homed)
 *                 → fit into the drawing area, run IK, stream strokes with pen lift
 *                 ← {ok:true, accepted:{paths,points}}  (202; drawing runs async)
 *   GET  /position → {ok:true, ready:<bool>, plotting:<bool>, moving:<bool>, a:{homed,deg}, b:{homed,deg}}
 *   GET  /         → health text
 *
 * Drivetrain: NEMA-17 (200 full steps) → TB6600 @ 1/4 microstep → 32:1 harmonic.
 *   STEPS_PER_REV = 200 * 4 * 32 = 25600 steps per output-shaft revolution.
 *
 * Dependencies (Arduino IDE → Library Manager / Boards Manager):
 *   • esp32 board package  (WiFi, WebServer, Preferences)
 *   • AccelStepper  ≥ 1.64
 *   • ArduinoJson   ≥ 6.0
 *   • ESP32Servo    ≥ 1.1   (pen lift)
 */

#include <WiFi.h>
#include <WebServer.h>
#include <Preferences.h>
#include <ArduinoJson.h>
#include <AccelStepper.h>
#include <ESP32Servo.h>
#include <vector>
#include <math.h>

// ── Network ────────────────────────────────────────────────────────────────────
static const char*    WIFI_SSID = "lion";
static const char*    WIFI_PASS = "shawn429";
static const uint16_t HTTP_PORT = 9000;   // must match PLOTTER_PORT on the server

// ── Drivetrain (TB6600 @ 1/4 microstep, 32:1 harmonic gearbox) ─────────────────
#define MOTOR_FULL_STEPS   200        // NEMA-17, 1.8° per full step
#define MICROSTEPS         4          // TB6600 microstep DIP setting
#define GEAR_RATIO         32.0f      // harmonic drive reduction (output : motor)

#define STEPS_PER_REV   (MOTOR_FULL_STEPS * MICROSTEPS * GEAR_RATIO)  // 25600
#define STEPS_PER_DEG   (STEPS_PER_REV / 360.0f)

// Known output-shaft angle where each arm's limit switch engages. Must match
// LIMIT_A_DEG / LIMIT_B_DEG on the server so hardware and server agree on zero.
#define LIMIT_A_DEG   270.0f
#define LIMIT_B_DEG   -90.0f

// ── Five-bar geometry (mirrors sim/simulator.py and ik_test.ino) ───────────────
#define LINK_BASE   10.0f    // distance between the two motor shafts
#define LINK_PROX   13.0f    // proximal link, motor → elbow
#define LINK_DIST   15.0f    // distal link,  elbow → pen

// Ready pose struck once both arms are homed: each proximal link 45° above the
// horizontal, symmetric about +Y (left arm 135°, right arm 45°, CCW from +X).
#define READY_A_DEG   135.0f
#define READY_B_DEG   45.0f

// Pen is the lower of the two four-bar solutions (hangs below the elbows).
#define PEN_LOWER_SOLUTION   1

// If a motor turns the wrong way, flip its sign here (+1 or -1).
#define DIR_A   (+1)
#define DIR_B   (+1)

// ── Pin map (STEP/DIR/ENABLE to TB6600, plus a limit switch per arm) ───────────
#define A_STEP_PIN   32
#define A_DIR_PIN    33
#define A_EN_PIN     27
#define B_STEP_PIN   2
#define B_DIR_PIN    4
#define B_EN_PIN     14
#define DRIVER_ENABLE_ACTIVE_LOW   1

// Switches wired between the pin and GND; internal pull-ups → LOW when pressed.
#define A_LIMIT_PIN  25
#define B_LIMIT_PIN  26
#define LIMIT_ACTIVE_LOW   1

// ── Jog motion profile ─────────────────────────────────────────────────────────
#define JOG_MAX_SPEED   2000.0f   // steps/s (slow enough to catch the switch)
#define JOG_ACCEL       4000.0f   // steps/s²

// ── Pen lift servo ─────────────────────────────────────────────────────────────
#define PEN_SERVO_PIN     13
#define PEN_MIN_US        500     // servo pulse bounds passed to attach()
#define PEN_MAX_US        2500
#define PEN_UP_US         1500    // pen clear of the paper
#define PEN_DOWN_LIGHT_US 1750    // light contact  (stroke weight → 0)
#define PEN_DOWN_HEAVY_US 1950    // full pressure  (stroke weight → 1)
#define PEN_SETTLE_MS     220     // time for the servo to reach position

// ── Drawing area: the image is fitted into this rectangle (robot units) ────────
#define DRAW_CX   0.0f    // centre X of the drawing area
#define DRAW_CY   7.0f    // centre Y
#define DRAW_W    8.0f    // max width  (aspect ratio preserved)
#define DRAW_H    6.0f    // max height

// ── Draw motion profile & safety cap ───────────────────────────────────────────
#define DRAW_MAX_SPEED   1500.0f
#define DRAW_ACCEL       3000.0f
#define MAX_PLOT_POINTS  3000     // guard ESP32 RAM; lower server detail if exceeded

// ─────────────────────────────────────────────────────────────────────────────

AccelStepper motorA(AccelStepper::DRIVER, A_STEP_PIN, A_DIR_PIN);
AccelStepper motorB(AccelStepper::DRIVER, B_STEP_PIN, B_DIR_PIN);
WebServer    server(HTTP_PORT);
Preferences  prefs;
Servo        penServo;

// Calibration / plotting state.
static bool atReady      = false;  // both arms homed, accepting /plot and /home
static bool penDown      = false;
static bool travelActive = false;  // travelling to (0,0) via /home, driven from loop() so /position stays live

// A drawing job, filled by /plot and executed one point per loop() so the HTTP
// server stays responsive while the strokes stream out.
struct PlotJob {
  std::vector<float>    px, py;      // points in drawing space (robot units)
  std::vector<uint32_t> pathBegin;   // index where each path starts
  std::vector<float>    pathWeight;  // pen pressure per path (0..1)
  bool     active  = false;
  uint32_t pathIdx = 0;
  uint32_t ptIdx   = 0;
};
static PlotJob job;

struct Arm {
  AccelStepper* stepper;
  int    enPin;
  int    limitPin;
  int    dirSign;
  float  limitDeg;      // reference angle at the switch
  bool   homed;         // false until the switch has been tripped
  const char* nvsHomed; // NVS keys
  const char* nvsPos;
};

Arm armA{ &motorA, A_EN_PIN, A_LIMIT_PIN, DIR_A, LIMIT_A_DEG, false, "a_homed", "a_pos" };
Arm armB{ &motorB, B_EN_PIN, B_LIMIT_PIN, DIR_B, LIMIT_B_DEG, false, "b_homed", "b_pos" };

static bool limit_pressed(const Arm& arm) {
  int v = digitalRead(arm.limitPin);
  return LIMIT_ACTIVE_LOW ? (v == LOW) : (v == HIGH);
}

static float arm_position_deg(const Arm& arm) {
  return arm.stepper->currentPosition() / STEPS_PER_DEG;
}

static void persist(const Arm& arm) {
  prefs.putBool(arm.nvsHomed, arm.homed);
  prefs.putLong(arm.nvsPos, arm.stepper->currentPosition());
}

static void restore(Arm& arm) {
  arm.homed = prefs.getBool(arm.nvsHomed, false);
  arm.stepper->setCurrentPosition(prefs.getLong(arm.nvsPos, 0));
}

// Jog one arm by `degrees` in `cw`/`ccw`. Stops early on a fresh limit-switch
// press (rising edge) so you can still back an already-pressed arm off the
// switch. Returns true if the switch tripped during this move.
static bool jog_arm(Arm& arm, const String& direction, float degrees) {
  long steps = lroundf(degrees * STEPS_PER_DEG);
  int  sign  = (direction == "ccw") ? +1 : -1;   // CCW increases the tracked angle
  arm.stepper->move(sign * steps);

  bool wasPressed = limit_pressed(arm);
  bool hit = false;

  while (arm.stepper->distanceToGo() != 0) {
    arm.stepper->run();
    bool pressed = limit_pressed(arm);
    if (pressed && !wasPressed) { hit = true; break; }   // just made contact
    wasPressed = pressed;
  }

  if (hit) {
    // Contact defines this arm's known angle; start absolute tracking here.
    arm.stepper->setCurrentPosition(lroundf(arm.limitDeg * STEPS_PER_DEG));
    arm.homed = true;
  }
  persist(arm);
  return hit;
}

// ── Kinematics ──────────────────────────────────────────────────────────────────

// Two circle intersections; returns the count, points written to out[][2].
static int circle_intersections(float x0, float y0, float r0,
                                float x1, float y1, float r1,
                                float out[2][2]) {
  float dx = x1 - x0, dy = y1 - y0;
  float d  = sqrtf(dx * dx + dy * dy);
  if (d == 0.0f || d > r0 + r1 || d < fabsf(r0 - r1)) return 0;
  float a  = (r0 * r0 - r1 * r1 + d * d) / (2.0f * d);
  float h  = sqrtf(fmaxf(r0 * r0 - a * a, 0.0f));
  float xm = x0 + a * dx / d, ym = y0 + a * dy / d;
  float ux = -dy / d, uy = dx / d;
  out[0][0] = xm + h * ux;  out[0][1] = ym + h * uy;
  out[1][0] = xm - h * ux;  out[1][1] = ym - h * uy;
  return (h < 1e-6f) ? 1 : 2;
}

// Motor angles (radians, CCW from +X) for a pen target; picks the outward elbow
// branch (left elbow left, right elbow right) matching sim/simulator.py.
static bool inverse_kinematics(float px, float py, float* thA, float* thB) {
  const float ax = -LINK_BASE / 2.0f, ay = 0.0f;
  const float bx =  LINK_BASE / 2.0f, by = 0.0f;
  float left[2][2], right[2][2];
  int nl = circle_intersections(ax, ay, LINK_PROX, px, py, LINK_DIST, left);
  int nr = circle_intersections(bx, by, LINK_PROX, px, py, LINK_DIST, right);
  if (nl == 0 || nr == 0) return false;

  float e1x = left[0][0], e1y = left[0][1];
  if (nl == 2 && left[1][0] < e1x) { e1x = left[1][0]; e1y = left[1][1]; }
  float e2x = right[0][0], e2y = right[0][1];
  if (nr == 2 && right[1][0] > e2x) { e2x = right[1][0]; e2y = right[1][1]; }

  *thA = atan2f(e1y - ay, e1x - ax);
  *thB = atan2f(e2y - by, e2x - bx);
  return true;
}

// Coordinated blocking move of both arms to target output angles (deg).
static void move_to_angles(float thetaA_deg, float thetaB_deg) {
  motorA.moveTo(lroundf(thetaA_deg * STEPS_PER_DEG));
  motorB.moveTo(lroundf(thetaB_deg * STEPS_PER_DEG));
  while (motorA.distanceToGo() != 0 || motorB.distanceToGo() != 0) {
    motorA.run();
    motorB.run();
  }
  persist(armA);
  persist(armB);
}

// Travel (pen up) to a drawing-space point, asynchronously — travel_step()
// drives it from loop() so /position keeps reporting live angles the whole
// way. Safe to call again mid-travel to retarget (e.g. a dragged jog request),
// which just redirects the in-flight move instead of restarting it.
static bool start_travel_to(float x, float y) {
  float ta, tb;
  if (!inverse_kinematics(x, y, &ta, &tb)) {
    Serial.println("[goto] target unreachable — staying at the current pose");
    return false;
  }

  if (!travelActive) {
    if (penDown) pen_up();
    motorA.setMaxSpeed(DRAW_MAX_SPEED); motorA.setAcceleration(DRAW_ACCEL);
    motorB.setMaxSpeed(DRAW_MAX_SPEED); motorB.setAcceleration(DRAW_ACCEL);
  }
  motorA.moveTo(lroundf(degrees(ta) * STEPS_PER_DEG));
  motorB.moveTo(lroundf(degrees(tb) * STEPS_PER_DEG));
  travelActive = true;                                 // travel_step() takes it from here
  return true;
}


// ── Pen + plotting ──────────────────────────────────────────────────────────────

static void pen_up() {
  penServo.writeMicroseconds(PEN_UP_US);
  delay(PEN_SETTLE_MS);
  penDown = false;
}

static void pen_down(float weight) {
  weight = constrain(weight, 0.0f, 1.0f);
  int us = PEN_DOWN_LIGHT_US + (int)(weight * (PEN_DOWN_HEAVY_US - PEN_DOWN_LIGHT_US));
  penServo.writeMicroseconds(us);
  delay(PEN_SETTLE_MS);
  penDown = true;
}

// Move the pen to a drawing-space point via IK. Returns false if unreachable.
static bool move_to_xy(float x, float y) {
  float ta, tb;
  if (!inverse_kinematics(x, y, &ta, &tb)) return false;
  motorA.moveTo(lroundf(degrees(ta) * STEPS_PER_DEG));
  motorB.moveTo(lroundf(degrees(tb) * STEPS_PER_DEG));
  while (motorA.distanceToGo() != 0 || motorB.distanceToGo() != 0) {
    motorA.run();
    motorB.run();
  }
  return true;
}

static void plot_finish() {
  if (penDown) pen_up();
  move_to_angles(READY_A_DEG, READY_B_DEG);            // park at the ready pose
  motorA.setMaxSpeed(JOG_MAX_SPEED); motorA.setAcceleration(JOG_ACCEL);
  motorB.setMaxSpeed(JOG_MAX_SPEED); motorB.setAcceleration(JOG_ACCEL);
  job.active = false;
  job.px.clear(); job.py.clear();
  job.pathBegin.clear(); job.pathWeight.clear();
  job.px.shrink_to_fit(); job.py.shrink_to_fit();
  Serial.println("[plot] complete — parked at ready pose");
}

// Draws one point per call so /position and future controls stay responsive.
static void plot_step() {
  uint32_t nPts   = job.px.size();
  uint32_t nPaths = job.pathBegin.size();
  if (job.ptIdx >= nPts) { plot_finish(); return; }

  uint32_t pathEnd = (job.pathIdx + 1 < nPaths) ? job.pathBegin[job.pathIdx + 1] : nPts;
  bool atStart = (job.ptIdx == job.pathBegin[job.pathIdx]);

  if (atStart && penDown) pen_up();                    // lift before travelling to a new stroke
  bool reached = move_to_xy(job.px[job.ptIdx], job.py[job.ptIdx]);
  if (reached && atStart) pen_down(job.pathWeight[job.pathIdx]);

  job.ptIdx++;
  if (job.ptIdx >= pathEnd) {                           // stroke finished
    if (penDown) pen_up();
    job.pathIdx++;
    if (job.pathIdx >= nPaths) plot_finish();
  }
}

// Drives the /home travel to (0,0) one AccelStepper tick at a time from
// loop(), instead of blocking, so /position keeps reporting live angles.
static void travel_step() {
  motorA.run();
  motorB.run();
  if (motorA.distanceToGo() != 0 || motorB.distanceToGo() != 0) return;

  motorA.setMaxSpeed(JOG_MAX_SPEED); motorA.setAcceleration(JOG_ACCEL);
  motorB.setMaxSpeed(JOG_MAX_SPEED); motorB.setAcceleration(JOG_ACCEL);
  persist(armA); persist(armB);
  travelActive = false;
  Serial.println("[goto] arrived");
}

// ── HTTP handlers ──────────────────────────────────────────────────────────────

static void send_json(int code, const String& body) {
  server.send(code, "application/json", body);
}

static void handle_motor() {
  Serial.println("[HTTP] /motor request received");

  if (server.method() != HTTP_POST) {
    send_json(405, "{\"ok\":false,\"error\":\"POST only\"}");
    return;
  }

  if (job.active) {
    send_json(409, "{\"ok\":false,\"error\":\"busy — plotting\"}");
    return;
  }
  if (travelActive) {
    send_json(409, "{\"ok\":false,\"error\":\"busy — travelling home\"}");
    return;
  }

  StaticJsonDocument<192> doc;

  String raw = server.arg("plain");
  Serial.printf("[HTTP] body: %s\n", raw.c_str());

  if (deserializeJson(doc, raw)) {
    Serial.println("[HTTP] ERROR: bad JSON");
    send_json(400, "{\"ok\":false,\"error\":\"bad JSON\"}");
    return;
  }

  String motor = doc["motor"] | "";
  String dir   = doc["direction"] | "";
  float deg    = doc["degrees"] | 0.0f;

  motor.toUpperCase();
  dir.toLowerCase();

  Serial.printf(
      "[motor] motor=%s direction=%s degrees=%.2f\n",
      motor.c_str(),
      dir.c_str(),
      deg
  );

  Arm* arm =
      (motor == "A") ? &armA :
      (motor == "B") ? &armB :
                       nullptr;

  if (!arm) {
    send_json(400, "{\"ok\":false,\"error\":\"motor must be A or B\"}");
    return;
  }

  bool hit = jog_arm(*arm, dir, deg);

  Serial.printf(
      "[motor] jog finished hit=%d homed=%d position=%.2f\n",
      hit,
      arm->homed,
      arm_position_deg(*arm)
  );

  if (armA.homed && armB.homed) atReady = true;

  String posStr =
      arm->homed ? String(arm_position_deg(*arm), 3) : String("null");

  String body =
      String("{\"ok\":true,\"motor\":\"") + motor +
      "\",\"limit\":" + (hit ? "true" : "false") +
      ",\"homed\":" + (arm->homed ? "true" : "false") +
      ",\"position_deg\":" + posStr + "}";

  send_json(200, body);
}

static void handle_position() {
  String aPos = armA.homed ? String(arm_position_deg(armA), 3) : String("null");
  String bPos = armB.homed ? String(arm_position_deg(armB), 3) : String("null");
  String body = String("{\"ok\":true,\"ready\":") + (atReady ? "true" : "false") +
                ",\"plotting\":" + (job.active ? "true" : "false") +
                ",\"moving\":" + (travelActive ? "true" : "false") +
                ",\"a\":{\"homed\":" + (armA.homed ? "true" : "false") +
                ",\"deg\":" + aPos + "},\"b\":{\"homed\":" + (armB.homed ? "true" : "false") +
                ",\"deg\":" + bPos + "}}";
  send_json(200, body);
}

// Triggered by the server's "return home" button — travels (pen up) to (0,0).
static void handle_home() {
  if (server.method() != HTTP_POST) { send_json(405, "{\"ok\":false,\"error\":\"POST only\"}"); return; }
  if (!atReady)      { send_json(409, "{\"ok\":false,\"error\":\"not ready — home both arms first\"}"); return; }
  if (job.active)    { send_json(409, "{\"ok\":false,\"error\":\"busy — a plot is already running\"}"); return; }
  if (travelActive)  { send_json(409, "{\"ok\":false,\"error\":\"busy — already travelling home\"}"); return; }
  if (!start_travel_to(DRAW_CX - DRAW_W / 2.0f, DRAW_CY - DRAW_H / 2.0f)) {
    send_json(409, "{\"ok\":false,\"error\":\"origin (0,0) unreachable\"}");
    return;
  }
  send_json(202, "{\"ok\":true}");
}

// Drives the XY jog joystick — POST {x,y} (drawing-space units). Unlike /home,
// a request while already travelling just retargets the in-flight move, so
// dragging the joystick continuously redirects the pen instead of queuing up.
static void handle_goto() {
  if (server.method() != HTTP_POST) { send_json(405, "{\"ok\":false,\"error\":\"POST only\"}"); return; }
  if (!atReady)   { send_json(409, "{\"ok\":false,\"error\":\"not ready — home both arms first\"}"); return; }
  if (job.active) { send_json(409, "{\"ok\":false,\"error\":\"busy — a plot is already running\"}"); return; }

  StaticJsonDocument<128> doc;
  if (deserializeJson(doc, server.arg("plain")) || doc["x"].isNull() || doc["y"].isNull()) {
    send_json(400, "{\"ok\":false,\"error\":\"bad JSON — need {x,y}\"}");
    return;
  }
  float x = doc["x"], y = doc["y"];
  if (!start_travel_to(x, y)) {
    send_json(409, "{\"ok\":false,\"error\":\"target unreachable\"}");
    return;
  }
  send_json(202, "{\"ok\":true}");
}

static void handle_plot() {
  if (server.method() != HTTP_POST) { send_json(405, "{\"ok\":false,\"error\":\"POST only\"}"); return; }
  if (!atReady)   { send_json(409, "{\"ok\":false,\"error\":\"not ready — home both arms first\"}"); return; }
  if (job.active) { send_json(409, "{\"ok\":false,\"error\":\"busy — a plot is already running\"}"); return; }
  if (travelActive) { send_json(409, "{\"ok\":false,\"error\":\"busy — travelling home\"}"); return; }

  // The stroke JSON can be large; size the parser from the biggest free block.
  DynamicJsonDocument doc((ESP.getMaxAllocHeap() * 3) / 4);
  if (deserializeJson(doc, server.arg("plain"))) {
    send_json(400, "{\"ok\":false,\"error\":\"bad or too-large JSON — lower detail\"}");
    return;
  }

  float w = doc["width"]  | 0.0f;
  float h = doc["height"] | 0.0f;
  JsonArray paths   = doc["paths"].as<JsonArray>();
  JsonArray weights = doc["weights"].as<JsonArray>();
  if (paths.isNull() || w <= 0.0f || h <= 0.0f) {
    send_json(400, "{\"ok\":false,\"error\":\"missing width/height/paths\"}");
    return;
  }

  // Fit the image into the drawing rectangle, aspect preserved, Y flipped
  // (image origin is top-left y-down; robot space is y-up).
  float scale = fminf(DRAW_W / w, DRAW_H / h);

  job.px.clear(); job.py.clear(); job.pathBegin.clear(); job.pathWeight.clear();
  uint32_t total = 0, pi = 0;
  for (JsonArray path : paths) {
    if (path.size() < 1) { pi++; continue; }
    if (total + path.size() > MAX_PLOT_POINTS) {
      job.px.clear(); job.py.clear(); job.pathBegin.clear(); job.pathWeight.clear();
      send_json(413, "{\"ok\":false,\"error\":\"too many points — lower detail on the server\"}");
      return;
    }
    job.pathBegin.push_back(total);
    job.pathWeight.push_back(pi < weights.size() ? (float)weights[pi] : 1.0f);
    for (JsonArray pt : path) {
      float ix = pt[0], iy = pt[1];
      job.px.push_back(DRAW_CX + (ix - w / 2.0f) * scale);
      job.py.push_back(DRAW_CY - (iy - h / 2.0f) * scale);
      total++;
    }
    pi++;
  }

  if (total == 0) { send_json(400, "{\"ok\":false,\"error\":\"no drawable points\"}"); return; }

  motorA.setMaxSpeed(DRAW_MAX_SPEED); motorA.setAcceleration(DRAW_ACCEL);
  motorB.setMaxSpeed(DRAW_MAX_SPEED); motorB.setAcceleration(DRAW_ACCEL);
  job.pathIdx = 0; job.ptIdx = 0; job.active = true;   // execution runs in loop()

  Serial.printf("[plot] accepted %u paths, %u points\n", (unsigned)job.pathBegin.size(), (unsigned)total);
  String body = String("{\"ok\":true,\"accepted\":{\"paths\":") + (unsigned)job.pathBegin.size() +
                ",\"points\":" + (unsigned)total + "}}";
  send_json(202, body);
}

static void handle_root() {
  server.send(200, "text/plain", "5-Bar SCARA plotter motor server. POST /motor to jog.");
}

// ── Arduino entry points ────────────────────────────────────────────────────────

static void setup_arm(Arm& arm) {
  pinMode(arm.enPin, OUTPUT);
  digitalWrite(arm.enPin, DRIVER_ENABLE_ACTIVE_LOW ? LOW : HIGH);  // enable driver
  pinMode(arm.limitPin, INPUT_PULLUP);
  arm.stepper->setPinsInverted(arm.dirSign < 0, false, false);  // DIR sign → CCW = +position
  arm.stepper->setMaxSpeed(JOG_MAX_SPEED);
  arm.stepper->setAcceleration(JOG_ACCEL);
  restore(arm);
}

void setup() {
  Serial.begin(115200);
  Serial.println("\n[plotter_server] booting …");

  prefs.begin("plotter", false);
  setup_arm(armA);
  setup_arm(armB);

  ESP32PWM::allocateTimer(0);
  penServo.setPeriodHertz(50);
  penServo.attach(PEN_SERVO_PIN, PEN_MIN_US, PEN_MAX_US);
  pen_up();

  Serial.printf("[geom] steps/rev=%.0f  steps/deg=%.3f\n", (float)STEPS_PER_REV, STEPS_PER_DEG);
  Serial.printf("[cal] A homed=%d %.2f°  B homed=%d %.2f°\n",
                armA.homed, arm_position_deg(armA), armB.homed, arm_position_deg(armB));
  if (armA.homed && armB.homed) atReady = true;  // already calibrated from NVS; no boot-time motion

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[wifi] connecting");
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print('.'); }
  Serial.printf("\n[wifi] connected — http://%s:%u\n", WiFi.localIP().toString().c_str(), HTTP_PORT);

  server.on("/", HTTP_GET, handle_root);
  server.on("/motor", HTTP_POST, handle_motor);
  server.on("/plot", HTTP_POST, handle_plot);
  server.on("/home", HTTP_POST, handle_home);
  server.on("/goto", HTTP_POST, handle_goto);
  server.on("/position", HTTP_GET, handle_position);
  server.begin();
  Serial.println("[http] server started");
}

void loop() {
  server.handleClient();
  if (job.active)    plot_step();    // stream the current drawing, one point per loop
  if (travelActive)  travel_step();  // drive the travel-home move, one tick per loop
}
