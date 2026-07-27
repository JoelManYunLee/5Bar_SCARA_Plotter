#!/usr/bin/env python3
"""
jog_test_rpi.py — Raspberry Pi 3B manual motor + servo bench test for the 5-Bar SCARA.

Companion to ik_test_rpi.py.  Where the IK script asks you for an (x, y) pen
target and solves backwards for the two motor angles, this script does no
kinematics at all: it jogs each MOTOR independently so you can eyeball joint
limits, DIR signs, and the mechanics directly.  It also drives the end-effector
SERVO (the pen-lift) straight from the PS3 controller.

Same wiring, same drivers as ik_test_rpi.py — see that file's header for the full
wiring notes and one-time Pi setup.  Home both proximal links straight up (90°)
before running; the step counters start at zero and assume that orientation.

Controls
  Left stick  Y ... motor A: continuous jog while held (up = increase θ_a)
  Right stick Y ... motor B: continuous jog while held (up = increase θ_b)
  D-pad L / R ..... nudge motor A by KEY_STEP_DEG (one press = one nudge)
  D-pad U / D ..... nudge motor B by KEY_STEP_DEG
  L1 / R1 ......... end-effector servo: pen up / pen down preset
  Triangle / Cross  trim the servo by SERVO_NUDGE_US (live calibration)
  Start ........... return both motors to the home angle
  Select .......... disable motors and quit

Wire the pen-lift servo's signal to SERVO_PIN, power it from an external 5V
supply, and tie the servo ground to the Pi ground (see the pin block below).

Verify PS3 axis/button mapping if something feels wrong (varies by driver):
  jstest /dev/input/js0      # or: evtest /dev/input/eventN
  (left stick: ABS_X / ABS_Y;  right stick: ABS_RX / ABS_RY — or ABS_Z / ABS_RZ)
  (L1/R1: BTN_TL / BTN_TR;  Triangle/Cross: BTN_NORTH / BTN_SOUTH)
"""

import math
import sys
import threading
import time

try:
    import lgpio
except ImportError:
    sys.exit(
        "lgpio not found.\n"
        "  sudo apt install python3-lgpio"
    )

try:
    from inputs import get_gamepad, UnpluggedError
except ImportError:
    sys.exit("inputs not found — install with: pip3 install inputs")


# ── Motor / drivetrain ─────────────────────────────────────────────────────────
MOTOR_FULL_STEPS = 200        # NEMA-17: 1.8° / step → 200 steps/rev
MICROSTEPS       = 4          # TB6600 DIP switch setting — must match hardware
GEAR_RATIO       = 32.0       # harmonic-drive output : motor shaft

STEPS_PER_REV = MOTOR_FULL_STEPS * MICROSTEPS * GEAR_RATIO
STEPS_PER_RAD = STEPS_PER_REV / (2.0 * math.pi)

THETA_A_HOME_DEG = 90.0   # left  motor home angle (CCW from +X)
THETA_B_HOME_DEG = 90.0   # right motor home angle

# Joint travel limits (deg, CCW from +X).  Keep both proximal links in the upper
# half-plane so the arm can't fold through the base or hit a singularity.
THETA_MIN_DEG = 10.0
THETA_MAX_DEG = 170.0

DIR_A = +1    # flip to -1 if motor A runs backwards
DIR_B = +1

# ── RPi GPIO pin assignments (BCM numbering) ──────────────────────────────────
A_STEP = 17
A_DIR  = 27
A_ENA  = 22

B_STEP = 23
B_DIR  = 24
B_ENA  = 25

# End-effector servo (pen lift).  Direct 3.3V PWM signal on this GPIO — NOT an
# opto input, so it ignores the WIRING mode below.  Power the servo from an
# external 5V supply and tie its ground to the Pi's ground; GPIO18 is a safe,
# hardware-PWM-capable choice on the 40-pin header (physical pin 12).
SERVO_PIN = 18

WIRING = "common_anode"       # see ik_test_rpi.py header for the wiring modes

_ON  = 0 if WIRING == "common_anode" else 1   # GPIO level that turns an opto ON
_OFF = 1 - _ON                                # GPIO level that turns an opto OFF

# ── Step timing (mirrors ik_test_rpi.py) ──────────────────────────────────────
CRUISE_SPEED = 1600   # steps/s at full speed
MIN_SPEED    = 500    # steps/s at the start/end of each move (ramp endpoints)
ACCEL_STEPS  = 50     # steps used to ramp up to cruise (same count for ramp down)
PULSE_US     = 5      # PUL+ high time (µs); TB6600 spec is ≥ 2.5 µs

# ── Joint control ──────────────────────────────────────────────────────────────
KEY_STEP_DEG  = 1.0    # degrees per d-pad nudge
STICK_SPEED   = 25.0   # deg/s per joint when a stick is fully deflected
DEADZONE      = 0.15   # ignore stick deflections smaller than this (0–1)
LOOP_HZ       = 30     # main-loop polling rate (Hz)

# ── End-effector servo (pen lift) ─────────────────────────────────────────────
# Standard hobby servos accept a 1000–2000 µs pulse at 50 Hz (many take the wider
# 500–2500 µs range).  Tune the two presets to your linkage: SERVO_UP_US should
# lift the pen clear of the paper, SERVO_DOWN_US should press it down to draw.
SERVO_MIN_US   = 500     # safety clamp — never command a pulse below this
SERVO_MAX_US   = 2500    # safety clamp — never command a pulse above this
SERVO_UP_US    = 1000    # pen raised (travel)
SERVO_DOWN_US  = 2000    # pen lowered (drawing)
SERVO_NUDGE_US = 25      # µs per Triangle/Cross press, for live calibration
SERVO_FREQ_HZ  = 50      # servo refresh rate


# ─────────────────────────────────────────────────────────────────────────────
# Angle → step conversion
# ─────────────────────────────────────────────────────────────────────────────

def _angle_to_steps(theta_rad, home_deg, dir_sign):
    delta = theta_rad - math.radians(home_deg)
    return round(dir_sign * delta * STEPS_PER_RAD)


# ─────────────────────────────────────────────────────────────────────────────
# GPIO / motor driver  (identical to ik_test_rpi.py)
# ─────────────────────────────────────────────────────────────────────────────

# Pi 3B exposes the 40-pin header on gpiochip0.  (On a Pi 5 it is gpiochip4.)
GPIO_CHIP = 0

try:
    _h = lgpio.gpiochip_open(GPIO_CHIP)
except Exception as exc:
    sys.exit(
        f"Cannot open gpiochip{GPIO_CHIP}: {exc}\n"
        "  add your user to the 'gpio' group:  sudo usermod -aG gpio $USER  (then re-login)"
    )

# Step counters — home position = 0 for both motors.
_pos_a = 0
_pos_b = 0

# Current servo pulse width (µs); the pen starts raised.
_servo_us = SERVO_UP_US


def _setup_gpio():
    # Claim every opto line in the OFF (idle) state so no opto conducts at startup.
    for pin in (A_STEP, A_DIR, A_ENA, B_STEP, B_DIR, B_ENA):
        lgpio.gpio_claim_output(_h, pin, _OFF)
    # Claim the servo signal line and park the pen in the raised position.
    lgpio.gpio_claim_output(_h, SERVO_PIN, 0)
    _set_servo(SERVO_UP_US)


def _set_servo(us):
    """Command the end-effector servo to a pulse width (µs), clamped to the safe
    range.  Uses lgpio's background servo-pulse generator, so the signal keeps
    refreshing on its own without blocking the motion loop."""
    global _servo_us
    _servo_us = max(SERVO_MIN_US, min(SERVO_MAX_US, us))
    lgpio.tx_servo(_h, SERVO_PIN, int(round(_servo_us)), SERVO_FREQ_HZ)


def _enable(on: bool):
    # TB6600 ENA opto ON = driver outputs DISABLED (coils released).  So to
    # *enable* the motor we leave the ENA opto OFF, and vice-versa.
    level = _OFF if on else _ON
    lgpio.gpio_write(_h, A_ENA, level)
    lgpio.gpio_write(_h, B_ENA, level)


def _sleep_until(target):
    """Block until perf_counter() reaches `target` (seconds).

    Coarse-sleep the bulk of the interval (cheap, yields the CPU) and busy-wait
    only the final ~200 µs for accuracy — smooth timing without pegging a core.
    """
    remaining = target - time.perf_counter()
    if remaining > 250e-6:
        time.sleep(remaining - 200e-6)
    while time.perf_counter() < target:
        pass


def _pulse(pin):
    # One step = drive the opto ON for PULSE_US, then back to OFF (idle).
    lgpio.gpio_write(_h, pin, _ON)
    _sleep_until(time.perf_counter() + PULSE_US * 1e-6)
    lgpio.gpio_write(_h, pin, _OFF)


def _step_speed(i, total):
    """Trapezoidal speed at step i of total: ramp up → cruise → ramp down."""
    ramp = min(ACCEL_STEPS, max(total // 2, 1))
    if   i < ramp:            t = (i + 1) / ramp
    elif i >= total - ramp:   t = (total - i) / ramp
    else:                     t = 1.0
    return MIN_SPEED + t * (CRUISE_SPEED - MIN_SPEED)


def _move_to(target_a, target_b):
    """
    Coordinated Bresenham move to (target_a, target_b) in step space.
    Both motors reach their targets simultaneously.  Blocks until done.
    """
    global _pos_a, _pos_b

    delta_a = target_a - _pos_a
    delta_b = target_b - _pos_b
    if delta_a == 0 and delta_b == 0:
        return

    sign_a  = 1 if delta_a >= 0 else -1
    sign_b  = 1 if delta_b >= 0 else -1
    steps_a = abs(delta_a)
    steps_b = abs(delta_b)
    major   = max(steps_a, steps_b)

    # Set direction — TB6600 requires DIR stable ≥ 5 µs before first pulse.
    lgpio.gpio_write(_h, A_DIR, _ON if sign_a * DIR_A > 0 else _OFF)
    lgpio.gpio_write(_h, B_DIR, _ON if sign_b * DIR_B > 0 else _OFF)
    time.sleep(20e-6)

    err_a = major // 2
    err_b = major // 2

    # Accumulate step deadlines against a single clock so timing can't drift.
    t_next = time.perf_counter()

    for i in range(major):
        t_next += 1.0 / _step_speed(i, major)

        err_a += steps_a
        if err_a >= major:
            err_a -= major
            _pulse(A_STEP)
            _pos_a += sign_a

        err_b += steps_b
        if err_b >= major:
            err_b -= major
            _pulse(B_STEP)
            _pos_b += sign_b

        _sleep_until(t_next)


# ─────────────────────────────────────────────────────────────────────────────
# PS3 controller input (background thread)
# ─────────────────────────────────────────────────────────────────────────────

_state = {
    'dpad_x':   0,
    'dpad_y':   0,
    'stick_a':  0.0,   # left  stick Y → motor A
    'stick_b':  0.0,   # right stick Y → motor B
    'pen_up':   False, # L1 → raise servo to SERVO_UP_US
    'pen_down': False, # R1 → lower servo to SERVO_DOWN_US
    'servo_d':  0,     # Triangle/Cross nudges, accumulated (± per press)
    'home':     False,
    'quit':     False,
}
_state_lock = threading.Lock()


def _norm_axis(raw, lo=0, hi=255):
    """Map a raw axis value [lo, hi] to the range [−1.0, +1.0]."""
    mid  = (lo + hi) / 2.0
    span = (hi - lo) / 2.0
    return max(-1.0, min(1.0, (raw - mid) / span))


def _input_thread():
    print("[input] waiting for PS3 controller …")
    while True:
        try:
            events = get_gamepad()
        except UnpluggedError:
            print("[input] controller unplugged, retrying …")
            time.sleep(1.0)
            continue
        except Exception as exc:
            print(f"[input] {exc}")
            time.sleep(0.5)
            continue

        with _state_lock:
            for ev in events:
                if ev.ev_type == 'Absolute':
                    if   ev.code == 'ABS_HAT0X': _state['dpad_x']  = ev.state
                    elif ev.code == 'ABS_HAT0Y': _state['dpad_y']  = ev.state
                    elif ev.code == 'ABS_Y':     _state['stick_a'] = _norm_axis(ev.state)
                    elif ev.code == 'ABS_RY':    _state['stick_b'] = _norm_axis(ev.state)
                elif ev.ev_type == 'Key' and ev.state == 1:
                    if   ev.code == 'BTN_START':  _state['home']     = True
                    elif ev.code == 'BTN_SELECT': _state['quit']     = True
                    elif ev.code == 'BTN_TL':     _state['pen_up']   = True   # L1
                    elif ev.code == 'BTN_TR':     _state['pen_down'] = True   # R1
                    elif ev.code == 'BTN_NORTH':  _state['servo_d'] += 1      # Triangle
                    elif ev.code == 'BTN_SOUTH':  _state['servo_d'] -= 1      # Cross


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clamp_deg(deg):
    return max(THETA_MIN_DEG, min(THETA_MAX_DEG, deg))


def _command(th_a_deg, th_b_deg):
    """Drive both motors to the given joint angles (deg) and report them."""
    ta = _angle_to_steps(math.radians(th_a_deg), THETA_A_HOME_DEG, DIR_A)
    tb = _angle_to_steps(math.radians(th_b_deg), THETA_B_HOME_DEG, DIR_B)
    _move_to(ta, tb)
    print(f"[joints] A={th_a_deg:6.2f}°  B={th_b_deg:6.2f}°")


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _setup_gpio()
    _enable(True)

    threading.Thread(target=_input_thread, daemon=True).start()

    th_a = THETA_A_HOME_DEG
    th_b = THETA_B_HOME_DEG
    print(f"[jog_test] homing → A={th_a:.1f}°  B={th_b:.1f}°")
    _command(th_a, th_b)
    print("[jog_test] ready — sticks/d-pad jog each motor  |  L1/R1 = pen up/down  |  "
          "Triangle/Cross = servo trim  |  Start = home  |  Select = quit")

    tick      = 1.0 / LOOP_HZ
    prev_dpad = {'x': 0, 'y': 0}

    try:
        while True:
            t0 = time.monotonic()

            with _state_lock:
                s = dict(_state)
                # consume one-shot flags
                _state['home']     = False
                _state['pen_up']   = False
                _state['pen_down'] = False
                _state['servo_d']  = 0

            if s['quit']:
                print("[jog_test] quit")
                break

            # End-effector servo — presets and live trim.
            if s['pen_up']:
                _set_servo(SERVO_UP_US)
                print(f"[servo] pen up   → {_servo_us:.0f} µs")
            if s['pen_down']:
                _set_servo(SERVO_DOWN_US)
                print(f"[servo] pen down → {_servo_us:.0f} µs")
            if s['servo_d']:
                _set_servo(_servo_us + s['servo_d'] * SERVO_NUDGE_US)
                print(f"[servo] trim     → {_servo_us:.0f} µs")

            if s['home']:
                th_a, th_b = THETA_A_HOME_DEG, THETA_B_HOME_DEG
                print("[joints] home")
                _command(th_a, th_b)
                prev_dpad = {'x': 0, 'y': 0}
                time.sleep(tick)
                continue

            moved = False
            dx, dy = s['dpad_x'], s['dpad_y']

            # D-pad: fire once per press (rising-edge detection).
            if dx != 0 and dx != prev_dpad['x']:
                th_a = _clamp_deg(th_a + KEY_STEP_DEG * dx)   # right = +θ_a
                moved = True
            if dy != 0 and dy != prev_dpad['y']:
                # Hat Y: −1 = up → +θ_b;  +1 = down → −θ_b.
                th_b = _clamp_deg(th_b - KEY_STEP_DEG * dy)
                moved = True

            prev_dpad = {'x': dx, 'y': dy}

            # Sticks: continuous joint rate, only when the d-pad is neutral.
            if not moved and dx == 0 and dy == 0:
                sa, sb = s['stick_a'], s['stick_b']
                # Stick up = negative raw Y → positive joint rate.
                if abs(sa) > DEADZONE:
                    th_a = _clamp_deg(th_a - sa * STICK_SPEED * tick)
                    moved = True
                if abs(sb) > DEADZONE:
                    th_b = _clamp_deg(th_b - sb * STICK_SPEED * tick)
                    moved = True

            if moved:
                _command(th_a, th_b)

            elapsed = time.monotonic() - t0
            if elapsed < tick:
                time.sleep(tick - elapsed)

    except KeyboardInterrupt:
        print("\n[jog_test] interrupted")
    finally:
        _enable(False)
        lgpio.tx_servo(_h, SERVO_PIN, 0)   # stop servo pulses
        lgpio.gpiochip_close(_h)
        print("[jog_test] motors disabled, servo released, GPIO released")


if __name__ == '__main__':
    main()
