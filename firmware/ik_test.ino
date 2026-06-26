/**
 * ik_test — ESP32 inverse-kinematics bench test for the 5-Bar SCARA plotter
 *
 * Drives the two NEMA-17 motors (each behind a 32:1 harmonic-drive gearbox) so
 * the shared end effector tracks an X/Y target.  The target is steered live with
 * arrow keys sent over USB-Serial from a laptop terminal (e.g. minicom/screen/PuTTY
 * with raw arrow-key mode enabled, or any terminal that sends ANSI escape codes).
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
 * Controls (arrow keys over Serial at 115200 baud)
 *   ↑ / ↓ ........... move end effector +Y / -Y
 *   ← / → ........... move end effector -X / +X
 *
 * Dependencies (Arduino IDE → Library Manager):
 *   • AccelStepper     ≥ 1.64   (smooth, coordinated step generation)
 */

#include <math.h>
#include <AccelStepper.h>

// ── Machine geometry (robot units — use whatever you like, e.g. cm) ────────────
#define LINK_BASE   8.0f    // distance between the two motor shafts  (sim: d)
#define LINK_PROX   6.0f    // proximal link length, motor → elbow    (sim: l1)
#define LINK_DIST   9.0f    // distal link length,  elbow → pen       (sim: l2)

// ── Motor / drivetrain configuration ──────────────────────────────────────────
#define MOTOR_FULL_STEPS   200      // NEMA-17, 1.8° per full step
#define MICROSTEPS         16       // driver microstep setting (match DIP/jumpers)
#define GEAR_RATIO         32.0f    // harmonic drive reduction (output : motor)

#define STEPS_PER_REV   (MOTOR_FULL_STEPS * MICROSTEPS * GEAR_RATIO)
#define STEPS_PER_RAD   (STEPS_PER_REV / (2.0f * (float)M_PI))

#define THETA_A_HOME_DEG   90.0f    // left motor home angle (CCW from +X)
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
#define DRIVER_ENABLE_ACTIVE_LOW   1

// ── Motion limits ──────────────────────────────────────────────────────────────
#define MOTOR_MAX_SPEED   4000.0f   // steps/s ceiling per motor
#define MOTOR_ACCEL       8000.0f   // steps/s² ramp

// Distance the end effector moves per arrow-key press (robot units).
#define KEY_STEP   0.2f

// ─────────────────────────────────────────────────────────────────────────────

AccelStepper motorA(AccelStepper::DRIVER, A_STEP_PIN, A_DIR_PIN);
AccelStepper motorB(AccelStepper::DRIVER, B_STEP_PIN, B_DIR_PIN);

static float target_x = 0.0f;
static float target_y = 0.0f;

// ── Geometry helpers (ported from sim/simulator.py) ────────────────────────────

static int circle_intersections(float x0, float y0, float r0,
                                float x1, float y1, float r1,
                                float out[2][2]) {
    float dx = x1 - x0, dy = y1 - y0;
    float d  = sqrtf(dx * dx + dy * dy);
    if (d == 0.0f)                    return 0;
    if (d > r0 + r1 + 1e-6f)         return 0;
    if (d < fabsf(r0 - r1) - 1e-6f)  return 0;

    float a  = (r0 * r0 - r1 * r1 + d * d) / (2.0f * d);
    float h2 = r0 * r0 - a * a;
    float h  = sqrtf(h2 > 0.0f ? h2 : 0.0f);
    float xm = x0 + a * dx / d;
    float ym = y0 + a * dy / d;
    float ux = -dy / d, uy = dx / d;

    out[0][0] = xm + h * ux;  out[0][1] = ym + h * uy;
    out[1][0] = xm - h * ux;  out[1][1] = ym - h * uy;
    return (h < 1e-6f) ? 1 : 2;
}

static bool inverse_kinematics(float px, float py, float *th_a, float *th_b) {
    const float ax = -LINK_BASE / 2.0f, ay = 0.0f;
    const float bx =  LINK_BASE / 2.0f, by = 0.0f;

    float left[2][2], right[2][2];
    int nl = circle_intersections(ax, ay, LINK_PROX, px, py, LINK_DIST, left);
    int nr = circle_intersections(bx, by, LINK_PROX, px, py, LINK_DIST, right);
    if (nl == 0 || nr == 0)
        return false;

    // Left elbow bends outward to the left → smallest X solution.
    float e1x = left[0][0], e1y = left[0][1];
    if (nl == 2 && left[1][0] < e1x) { e1x = left[1][0]; e1y = left[1][1]; }

    // Right elbow bends outward to the right → largest X solution.
    float e2x = right[0][0], e2y = right[0][1];
    if (nr == 2 && right[1][0] > e2x) { e2x = right[1][0]; e2y = right[1][1]; }

    *th_a = atan2f(e1y - ay, e1x - ax);
    *th_b = atan2f(e2y - by, e2x - bx);
    return true;
}

static long angle_to_steps(float theta_rad, float home_deg, int dir_sign) {
    float delta = theta_rad - (home_deg * (float)M_PI / 180.0f);
    return (long)lroundf(dir_sign * delta * STEPS_PER_RAD);
}

static void command_target() {
    float th_a, th_b;
    if (!inverse_kinematics(target_x, target_y, &th_a, &th_b))
        return;
    motorA.moveTo(angle_to_steps(th_a, THETA_A_HOME_DEG, DIR_A));
    motorB.moveTo(angle_to_steps(th_b, THETA_B_HOME_DEG, DIR_B));
}

static void home_target() {
    target_x = 0.0f;
    target_y = (LINK_PROX + LINK_DIST) * 0.6f;
    command_target();
    Serial.printf("[target] home -> (%.2f, %.2f)\n", target_x, target_y);
}

// ── Arrow-key parsing (ANSI escape sequences: ESC [ A/B/C/D) ──────────────────
// State machine: idle → got_esc → got_bracket → dispatch
static void poll_serial_keys() {
    static uint8_t state = 0;   // 0=idle, 1=got ESC, 2=got '['

    while (Serial.available()) {
        int c = Serial.read();

        if (state == 0) {
            if (c == 0x1b) { state = 1; continue; }
            // Non-escape characters ignored here; add more bindings as needed.
        } else if (state == 1) {
            state = (c == '[') ? 2 : 0;
            continue;
        } else if (state == 2) {
            state = 0;
            float nx = target_x, ny = target_y;
            switch (c) {
                case 'A': ny += KEY_STEP; break;   // ↑
                case 'B': ny -= KEY_STEP; break;   // ↓
                case 'C': nx += KEY_STEP; break;   // →
                case 'D': nx -= KEY_STEP; break;   // ←
                default:  continue;
            }
            // Only commit if the new position is reachable.
            float th_a, th_b;
            if (inverse_kinematics(nx, ny, &th_a, &th_b)) {
                target_x = nx;
                target_y = ny;
                command_target();
                Serial.printf("[target] (%.2f, %.2f)\n", target_x, target_y);
            } else {
                Serial.println("[target] unreachable — ignored");
            }
        }
    }
}

// ── Arduino entry points ────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    Serial.println("\n[ik_test] booting …");

    pinMode(A_EN_PIN, OUTPUT);
    pinMode(B_EN_PIN, OUTPUT);

    // Enable drivers (active-low).
    digitalWrite(A_EN_PIN, DRIVER_ENABLE_ACTIVE_LOW ? LOW : HIGH);
    digitalWrite(B_EN_PIN, DRIVER_ENABLE_ACTIVE_LOW ? LOW : HIGH);

    motorA.setMaxSpeed(MOTOR_MAX_SPEED);
    motorA.setAcceleration(MOTOR_ACCEL);
    motorB.setMaxSpeed(MOTOR_MAX_SPEED);
    motorB.setAcceleration(MOTOR_ACCEL);

    Serial.printf("[geom] base=%.1f  L1=%.1f  L2=%.1f  steps/rad=%.1f\n",
                  LINK_BASE, LINK_PROX, LINK_DIST, STEPS_PER_RAD);
    Serial.println("[keys] arrow keys -> move end effector");

    home_target();
}

void loop() {
    poll_serial_keys();
    motorA.run();
    motorB.run();
}
