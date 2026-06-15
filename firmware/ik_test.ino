/**
 * ik_test — ESP32 inverse-kinematics bench test for the 5-Bar SCARA plotter
 *
 * Drives the two NEMA-17 motors (each behind a 32:1 harmonic-drive gearbox) so
 * the shared end effector tracks an X/Y target.  The target is steered live with
 * a PS4 controller over the ESP32's built-in Bluetooth, so you can feel out the
 * reachable workspace and verify the IK math before wiring up the full plotter.
 *
 * Geometry mirrors sim/simulator.py:
 *
 *        E1 ●───────────● E2          E1, E2 : elbows (proximal/distal joints)
 *           \           /
 *            \         /               P : end effector (pen)
 *             \       /
 *              ●─────●                 A, B : motor bases (LINK_BASE apart)
 *              A     B
 *
 *   • Motor A sits at (-LINK_BASE/2, 0), motor B at (+LINK_BASE/2, 0).
 *   • theta is measured CCW from +X.  The left elbow bends left, the right
 *     elbow bends right (the "upper" assembly), matching FiveBar.inverse().
 *
 * Controls (PS4)
 *   Left stick X/Y .... steer the end-effector target across the workspace
 *   R2 (analog) ....... speed boost (hold for faster travel)
 *   Cross (X) ......... re-home the target to the workspace center
 *   Circle (O) ........ toggle motor outputs enabled / disabled
 *
 * Dependencies (Arduino IDE → Library Manager):
 *   • AccelStepper     ≥ 1.64   (smooth, coordinated step generation)
 *   • PS4Controller    ≥ 0.0.1  (NURDspace/ESP32-PS4Controller — built-in BT)
 *
 * Pairing the PS4 controller
 *   Flash once, open Serial Monitor, note the ESP32 Bluetooth MAC printed at
 *   boot, then use a tool like SixaxisPairTool to set the controller's host MAC
 *   to that value.  Alternatively set PS4_PAIR_MAC below to a MAC the controller
 *   already trusts.  Hold PS + Share to put the pad in pairing mode.
 */

#include <math.h>
#include <AccelStepper.h>
#include <PS4Controller.h>

// ── Machine geometry (robot units — use whatever you like, e.g. cm) ────────────
// Placeholder values until the real linkage is measured.  These mirror the
// simulator defaults so the reachable workspace looks the same on the bench.
#define LINK_BASE   8.0f    // distance between the two motor shafts  (sim: d)
#define LINK_PROX   6.0f    // proximal link length, motor → elbow    (sim: l1)
#define LINK_DIST   9.0f    // distal link length,  elbow → pen       (sim: l2)

// ── Motor / drivetrain configuration ──────────────────────────────────────────
#define MOTOR_FULL_STEPS   200      // NEMA-17, 1.8° per full step
#define MICROSTEPS         16       // driver microstep setting (match DIP/jumpers)
#define GEAR_RATIO         32.0f    // harmonic drive reduction (output : motor)

// Output-shaft steps per full revolution, and the radians↔steps conversion.
#define STEPS_PER_REV   (MOTOR_FULL_STEPS * MICROSTEPS * GEAR_RATIO)
#define STEPS_PER_RAD   (STEPS_PER_REV / (2.0f * (float)M_PI))

// Physical joint angle (CCW from +X) that corresponds to motor "zero steps".
// Set these to whatever pose you home the arms to before powering on.
#define THETA_A_HOME_DEG   90.0f    // left motor home angle
#define THETA_B_HOME_DEG   90.0f    // right motor home angle

// If a motor turns the wrong way, flip its sign here (+1 or -1).
#define DIR_A   (+1)
#define DIR_B   (+1)

// ── Stepper driver pin map (STEP/DIR/ENABLE — e.g. A4988 / DRV8825 / TMC2208) ──
#define A_STEP_PIN   25
#define A_DIR_PIN    26
#define A_EN_PIN     27
#define B_STEP_PIN   32
#define B_DIR_PIN    33
#define B_EN_PIN     14
// Most step/dir drivers enable on a LOW level on EN.
#define DRIVER_ENABLE_ACTIVE_LOW   1

// ── Motion limits ──────────────────────────────────────────────────────────────
#define MOTOR_MAX_SPEED   4000.0f   // steps/s ceiling per motor (AccelStepper)
#define MOTOR_ACCEL       8000.0f   // steps/s² ramp
#define TARGET_SPEED      6.0f      // end-effector travel, robot units / second
#define TARGET_BOOST      2.5f      // R2 multiplier on TARGET_SPEED
#define STICK_DEADZONE    12        // ignore |stick| below this (raw ±128)

// Optional: bind to a controller MAC the pad already trusts.  Leave empty to use
// the ESP32's own factory Bluetooth MAC (printed at boot).
#define PS4_PAIR_MAC   ""           // e.g. "01:02:03:04:05:06"

// ─────────────────────────────────────────────────────────────────────────────

AccelStepper motorA(AccelStepper::DRIVER, A_STEP_PIN, A_DIR_PIN);
AccelStepper motorB(AccelStepper::DRIVER, B_STEP_PIN, B_DIR_PIN);

// Current end-effector target in robot coordinates.
static float target_x = 0.0f;
static float target_y = 0.0f;

static bool  motors_enabled = true;
static bool  last_circle    = false;   // edge-detect the O button
static uint32_t last_update_us = 0;

// ── Geometry helpers (ported from sim/simulator.py) ────────────────────────────

// Intersections of two circles centred at c0,c1 with radii r0,r1.
// Returns the number of solutions (0, 1, or 2) and writes them into out[2][2].
static int circle_intersections(float x0, float y0, float r0,
                                float x1, float y1, float r1,
                                float out[2][2]) {
    float dx = x1 - x0, dy = y1 - y0;
    float d  = sqrtf(dx * dx + dy * dy);
    if (d == 0.0f)               return 0;   // concentric
    if (d > r0 + r1 + 1e-6f)     return 0;   // too far apart
    if (d < fabsf(r0 - r1) - 1e-6f) return 0; // one inside the other

    float a  = (r0 * r0 - r1 * r1 + d * d) / (2.0f * d);
    float h2 = r0 * r0 - a * a;
    float h  = sqrtf(h2 > 0.0f ? h2 : 0.0f);
    float xm = x0 + a * dx / d;
    float ym = y0 + a * dy / d;
    float ux = -dy / d, uy = dx / d;         // unit normal to the centre line

    out[0][0] = xm + h * ux;  out[0][1] = ym + h * uy;
    out[1][0] = xm - h * ux;  out[1][1] = ym - h * uy;
    return (h < 1e-6f) ? 1 : 2;
}

// Inverse kinematics: end effector (px,py) → joint angles (rad).
// Returns true and fills *th_a/*th_b on success, false if unreachable.
static bool inverse_kinematics(float px, float py, float *th_a, float *th_b) {
    const float ax = -LINK_BASE / 2.0f, ay = 0.0f;   // left base
    const float bx =  LINK_BASE / 2.0f, by = 0.0f;   // right base

    float left[2][2], right[2][2];
    int nl = circle_intersections(ax, ay, LINK_PROX, px, py, LINK_DIST, left);
    int nr = circle_intersections(bx, by, LINK_PROX, px, py, LINK_DIST, right);
    if (nl == 0 || nr == 0)
        return false;

    // Left elbow bends outward to the left  → smallest X solution.
    float e1x = left[0][0], e1y = left[0][1];
    if (nl == 2 && left[1][0] < e1x) { e1x = left[1][0]; e1y = left[1][1]; }

    // Right elbow bends outward to the right → largest X solution.
    float e2x = right[0][0], e2y = right[0][1];
    if (nr == 2 && right[1][0] > e2x) { e2x = right[1][0]; e2y = right[1][1]; }

    *th_a = atan2f(e1y - ay, e1x - ax);
    *th_b = atan2f(e2y - by, e2x - bx);
    return true;
}

// Convert a joint angle (rad, CCW from +X) into an absolute step position,
// relative to the configured home angle and accounting for direction sign.
static long angle_to_steps(float theta_rad, float home_deg, int dir_sign) {
    float delta = theta_rad - (home_deg * (float)M_PI / 180.0f);
    return (long)lroundf(dir_sign * delta * STEPS_PER_RAD);
}

// ── Driver enable/disable ──────────────────────────────────────────────────────
static void set_motors_enabled(bool en) {
    motors_enabled = en;
    int level;
#if DRIVER_ENABLE_ACTIVE_LOW
    level = en ? LOW : HIGH;
#else
    level = en ? HIGH : LOW;
#endif
    digitalWrite(A_EN_PIN, level);
    digitalWrite(B_EN_PIN, level);
    Serial.printf("[motors] %s\n", en ? "ENABLED" : "disabled");
}

// Push the current target through IK and command both steppers.
static void command_target() {
    float th_a, th_b;
    if (!inverse_kinematics(target_x, target_y, &th_a, &th_b)) {
        // Unreachable — hold position so a bad target can't fling the arms.
        return;
    }
    motorA.moveTo(angle_to_steps(th_a, THETA_A_HOME_DEG, DIR_A));
    motorB.moveTo(angle_to_steps(th_b, THETA_B_HOME_DEG, DIR_B));
}

// Snap the target to a safe, reachable spot near the workspace centre.
static void home_target() {
    target_x = 0.0f;
    target_y = (LINK_PROX + LINK_DIST) * 0.6f;   // comfortably inside reach
    command_target();
    Serial.printf("[target] home -> (%.2f, %.2f)\n", target_x, target_y);
}

// ── PS4 input → target velocity integration ────────────────────────────────────
static void update_target_from_ps4(float dt) {
    if (!PS4.isConnected())
        return;

    // Toggle motor enable on the rising edge of Circle.
    bool circle = PS4.Circle();
    if (circle && !last_circle)
        set_motors_enabled(!motors_enabled);
    last_circle = circle;

    // Re-home on Cross.
    if (PS4.Cross())
        home_target();

    // Left stick: raw range roughly -128..127, +Y is down on the pad.
    int sx = PS4.LStickX();
    int sy = PS4.LStickY();
    if (abs(sx) < STICK_DEADZONE) sx = 0;
    if (abs(sy) < STICK_DEADZONE) sy = 0;

    float nx = sx / 128.0f;          // normalise to ~-1..1
    float ny = sy / 128.0f;

    // R2 analog (0..255) blends in the speed boost.
    float boost = 1.0f + (PS4.R2Value() / 255.0f) * (TARGET_BOOST - 1.0f);
    float speed = TARGET_SPEED * boost;

    float nx_new = target_x + nx * speed * dt;
    float ny_new = target_y + ny * speed * dt;   // stick +Y (down) → -world Y

    // Only commit the new target if it stays inside the reachable workspace,
    // so the arms can't be driven past a singularity into the unreachable zone.
    float th_a, th_b;
    if (inverse_kinematics(nx_new, ny_new, &th_a, &th_b)) {
        target_x = nx_new;
        target_y = ny_new;
        command_target();
    }
}

// ── Arduino entry points ────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    Serial.println("\n[ik_test] booting …");

    pinMode(A_EN_PIN, OUTPUT);
    pinMode(B_EN_PIN, OUTPUT);

    motorA.setMaxSpeed(MOTOR_MAX_SPEED);
    motorA.setAcceleration(MOTOR_ACCEL);
    motorB.setMaxSpeed(MOTOR_MAX_SPEED);
    motorB.setAcceleration(MOTOR_ACCEL);

    set_motors_enabled(true);

    if (strlen(PS4_PAIR_MAC) > 0)
        PS4.begin(PS4_PAIR_MAC);
    else
        PS4.begin();             // advertise using the ESP32's own BT MAC
    Serial.printf("[ps4] waiting for controller … (host MAC %s)\n",
                  PS4.getAddress().c_str());

    Serial.printf("[geom] base=%.1f  L1=%.1f  L2=%.1f  steps/rad=%.1f\n",
                  LINK_BASE, LINK_PROX, LINK_DIST, STEPS_PER_RAD);

    home_target();
    last_update_us = micros();
}

void loop() {
    // Time-step for velocity integration (seconds since last input poll).
    uint32_t now = micros();
    float dt = (now - last_update_us) * 1e-6f;
    if (dt >= 0.005f) {                  // poll input at ~200 Hz
        last_update_us = now;
        update_target_from_ps4(dt);
    }

    // AccelStepper must be run as often as possible for smooth stepping.
    if (motors_enabled) {
        motorA.run();
        motorB.run();
    }
}
