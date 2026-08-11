/**
 * plotter_server — ESP32 Wi-Fi motor server for the 5-Bar SCARA plotter
 *
 * Hosts the HTTP endpoint the Flask server (server/app.py + process.py) talks to
 * during calibration. Each arm is jogged a fixed amount; the moment that arm's
 * limit switch trips, the motor stops and its angle becomes known (locked to a
 * reference), and absolute step tracking begins from there. Positions survive
 * power cycles via NVS, so the machine stays calibrated between sessions.
 *
 *   POST /motor   {motor:"A"|"B", direction:"cw"|"ccw", degrees:<n>}
 *                 → jog; stop early if the switch engages
 *                 ← {ok:true, limit:<bool>, homed:<bool>, position_deg:<n|null>}
 *   GET  /position → {ok:true, a:{homed,deg}, b:{homed,deg}}
 *   GET  /         → health text
 *
 * Drivetrain: NEMA-17 (200 full steps) → TB6600 @ 1/4 microstep → 32:1 harmonic.
 *   STEPS_PER_REV = 200 * 4 * 32 = 25600 steps per output-shaft revolution.
 *
 * Dependencies (Arduino IDE → Library Manager / Boards Manager):
 *   • esp32 board package  (WiFi, WebServer, Preferences)
 *   • AccelStepper  ≥ 1.64
 *   • ArduinoJson   ≥ 6.0
 */

#include <WiFi.h>
#include <WebServer.h>
#include <Preferences.h>
#include <ArduinoJson.h>
#include <AccelStepper.h>
#include <math.h>

// ── Network ────────────────────────────────────────────────────────────────────
static const char*    WIFI_SSID = "YOUR_WIFI_SSID";
static const char*    WIFI_PASS = "YOUR_WIFI_PASSWORD";
static const uint16_t HTTP_PORT = 9000;   // must match PLOTTER_PORT on the server

// ── Drivetrain (TB6600 @ 1/4 microstep, 32:1 harmonic gearbox) ─────────────────
#define MOTOR_FULL_STEPS   200        // NEMA-17, 1.8° per full step
#define MICROSTEPS         4          // TB6600 microstep DIP setting
#define GEAR_RATIO         32.0f      // harmonic drive reduction (output : motor)

#define STEPS_PER_REV   (MOTOR_FULL_STEPS * MICROSTEPS * GEAR_RATIO)  // 25600
#define STEPS_PER_DEG   (STEPS_PER_REV / 360.0f)

// Known output-shaft angle where each arm's limit switch engages. Must match
// LIMIT_A_DEG / LIMIT_B_DEG on the server so hardware and server agree on zero.
#define LIMIT_A_DEG   90.0f
#define LIMIT_B_DEG   90.0f

// If a motor turns the wrong way, flip its sign here (+1 or -1).
#define DIR_A   (+1)
#define DIR_B   (+1)

// ── Pin map (STEP/DIR/ENABLE to TB6600, plus a limit switch per arm) ───────────
#define A_STEP_PIN   25
#define A_DIR_PIN    26
#define A_EN_PIN     27
#define B_STEP_PIN   32
#define B_DIR_PIN    33
#define B_EN_PIN     14
#define DRIVER_ENABLE_ACTIVE_LOW   1

// Switches wired between the pin and GND; internal pull-ups → LOW when pressed.
#define A_LIMIT_PIN  21
#define B_LIMIT_PIN  22
#define LIMIT_ACTIVE_LOW   1

// ── Jog motion profile ─────────────────────────────────────────────────────────
#define JOG_MAX_SPEED   2000.0f   // steps/s (slow enough to catch the switch)
#define JOG_ACCEL       4000.0f   // steps/s²

// ─────────────────────────────────────────────────────────────────────────────

AccelStepper motorA(AccelStepper::DRIVER, A_STEP_PIN, A_DIR_PIN);
AccelStepper motorB(AccelStepper::DRIVER, B_STEP_PIN, B_DIR_PIN);
WebServer    server(HTTP_PORT);
Preferences  prefs;

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
  int  sign  = (direction == "cw" ? +1 : -1) * arm.dirSign;
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

// ── HTTP handlers ──────────────────────────────────────────────────────────────

static void send_json(int code, const String& body) {
  server.send(code, "application/json", body);
}

static void handle_motor() {
  if (server.method() != HTTP_POST) { send_json(405, "{\"ok\":false,\"error\":\"POST only\"}"); return; }

  StaticJsonDocument<192> doc;
  if (deserializeJson(doc, server.arg("plain"))) {
    send_json(400, "{\"ok\":false,\"error\":\"bad JSON\"}");
    return;
  }

  String motor = doc["motor"] | "";
  String dir   = doc["direction"] | "";
  float  deg   = doc["degrees"] | 0.0f;
  motor.toUpperCase();
  dir.toLowerCase();

  Arm* arm = (motor == "A") ? &armA : (motor == "B") ? &armB : nullptr;
  if (!arm)                        { send_json(400, "{\"ok\":false,\"error\":\"motor must be A or B\"}"); return; }
  if (dir != "cw" && dir != "ccw"){ send_json(400, "{\"ok\":false,\"error\":\"direction must be cw or ccw\"}"); return; }
  if (deg <= 0 || deg > 360)      { send_json(400, "{\"ok\":false,\"error\":\"degrees out of range\"}"); return; }

  bool hit = jog_arm(*arm, dir, deg);

  String posStr = arm->homed ? String(arm_position_deg(*arm), 3) : String("null");
  String body = String("{\"ok\":true,\"motor\":\"") + motor +
                "\",\"limit\":" + (hit ? "true" : "false") +
                ",\"homed\":" + (arm->homed ? "true" : "false") +
                ",\"position_deg\":" + posStr + "}";
  send_json(200, body);
}

static void handle_position() {
  String aPos = armA.homed ? String(arm_position_deg(armA), 3) : String("null");
  String bPos = armB.homed ? String(arm_position_deg(armB), 3) : String("null");
  String body = String("{\"ok\":true,\"a\":{\"homed\":") + (armA.homed ? "true" : "false") +
                ",\"deg\":" + aPos + "},\"b\":{\"homed\":" + (armB.homed ? "true" : "false") +
                ",\"deg\":" + bPos + "}}";
  send_json(200, body);
}

static void handle_root() {
  server.send(200, "text/plain", "5-Bar SCARA plotter motor server. POST /motor to jog.");
}

// ── Arduino entry points ────────────────────────────────────────────────────────

static void setup_arm(Arm& arm) {
  pinMode(arm.enPin, OUTPUT);
  digitalWrite(arm.enPin, DRIVER_ENABLE_ACTIVE_LOW ? LOW : HIGH);  // enable driver
  pinMode(arm.limitPin, INPUT_PULLUP);
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

  Serial.printf("[geom] steps/rev=%.0f  steps/deg=%.3f\n", (float)STEPS_PER_REV, STEPS_PER_DEG);
  Serial.printf("[cal] A homed=%d %.2f°  B homed=%d %.2f°\n",
                armA.homed, arm_position_deg(armA), armB.homed, arm_position_deg(armB));

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[wifi] connecting");
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print('.'); }
  Serial.printf("\n[wifi] connected — http://%s:%u\n", WiFi.localIP().toString().c_str(), HTTP_PORT);

  server.on("/", HTTP_GET, handle_root);
  server.on("/motor", HTTP_POST, handle_motor);
  server.on("/position", HTTP_GET, handle_position);
  server.begin();
  Serial.println("[http] server started");
}

void loop() {
  server.handleClient();
}
