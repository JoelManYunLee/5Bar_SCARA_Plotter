#!/usr/bin/env python3
"""
ik_test_rpi.py — Raspberry Pi 3B IK bench test for the 5-Bar SCARA plotter.

Replaces ik_test.ino for initial IK verification on the RPi.
A PS3 DualShock 3 plugged in over USB steers the end effector;
two TB6600 drivers (STEP / DIR / ENA) drive the NEMA-17 motors
via the RPi 40-pin GPIO header.

Controls
  D-pad ........... move end effector one KEY_STEP in X or Y
                    (same feel as arrow keys in the ESP32 version)
  Left stick ...... continuous movement while held
  Start ........... return to home position
  Select .......... disable motors and quit

Geometry is identical to ik_test.ino and sim/simulator.py.

─── One-time setup on the Pi (Raspberry Pi OS Bookworm) ──────────────────────
  sudo apt install python3-lgpio        # usually already present on Bookworm
  pip3 install inputs                   # if "externally-managed", add --break-system-packages
  sudo usermod -aG input,gpio $USER     # /dev/input + /dev/gpiochip access; re-login after
─────────────────────────────────────────────────────────────────────────────

Note: this uses lgpio, not pigpio.  pigpio was dropped from Bookworm because its
DMA daemon is incompatible with the new GPIO character-device kernel interface.
lgpio is software-timed from Python, so step pulses have more jitter than pigpio's
DMA engine — fine for a low-speed bench test (≤ ~1 kHz), but for production motion
move to lgpio waves (tx_wave) or back to an MCU.

Wiring — COMMON-ANODE (set WIRING = "common_anode" below; this is the default)
  Tie all "+" pins to the Pi's 3.3V rail, sink the "−" pins with the GPIOs:
        PUL+ ┐                 PUL− → RPi GPIO  (A_STEP / B_STEP)
        DIR+ ├─ RPi 3.3V       DIR− → RPi GPIO  (A_DIR  / B_DIR)
        ENA+ ┘  (pin 1 or 17)  ENA− → RPi GPIO  (A_ENA  / B_ENA)
  Use 3.3V — NOT 5V — for the anode rail: a 5V rail leaves ~1.7V across the
  opto when a 3.3V GPIO goes HIGH, which may not switch it fully off.

  For COMMON-CATHODE instead (GPIOs drive "+", all "−" to GND), set
  WIRING = "common_cathode" and the logic inverts automatically.

    BCM 17  = physical pin 11    BCM 23  = physical pin 16
    BCM 27  = physical pin 13    BCM 24  = physical pin 18
    BCM 22  = physical pin 15    BCM 25  = physical pin 22
    3.3V    = physical pin 1 or 17

IMPORTANT: before running, manually rotate both proximal links to the home
  angle (90° above horizontal — links pointing straight up).  The step
  counter starts at zero and assumes that physical orientation.

Verify PS3 axis mapping if movement feels wrong:
  jstest /dev/input/js0
  (D-pad: ABS_HAT0X / ABS_HAT0Y;  left stick: ABS_X / ABS_Y)
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


# ── Machine geometry (must match ik_test.ino and sim/simulator.py) ────────────
LINK_BASE = 10.0     # distance between the two motor shafts (cm)
LINK_PROX = 13.0     # proximal link length, motor → elbow  (cm)
LINK_DIST = 15.0     # distal link length,   elbow → pen    (cm)

# ── Motor / drivetrain ─────────────────────────────────────────────────────────
MOTOR_FULL_STEPS = 200        # NEMA-17: 1.8° / step → 200 steps/rev
MICROSTEPS       = 4          # TB6600 DIP switch setting — must match hardware
GEAR_RATIO       = 32.0       # harmonic-drive output : motor shaft

STEPS_PER_REV = MOTOR_FULL_STEPS * MICROSTEPS * GEAR_RATIO  # 200·8·32 = 51 200
STEPS_PER_RAD = STEPS_PER_REV / (2.0 * math.pi)             # ≈ 8 149

THETA_A_HOME_DEG = 90.0   # left  motor home angle (CCW from +X)
THETA_B_HOME_DEG = 90.0   # right motor home angle

DIR_A = +1    # flip to -1 if motor A runs backwards
DIR_B = +1

# ── RPi GPIO pin assignments (BCM numbering) ──────────────────────────────────
A_STEP = 17
A_DIR  = 27
A_ENA  = 22

B_STEP = 23
B_DIR  = 24
B_ENA  = 25

# TB6600 opto-input wiring mode:
#   "common_cathode" — all "−" pins to GND; GPIOs drive the "+" pins.
#                      An opto turns ON when its GPIO is HIGH (active-HIGH).
#   "common_anode"   — all "+" pins to the 3.3V rail; GPIOs sink the "−" pins.
#                      An opto turns ON when its GPIO is LOW (active-LOW).
# For a 3.3V Pi in common-anode, tie the "+" rail to 3.3V (header pin 1 or 17),
# NOT 5V — a 5V anode leaves ~1.7V across the opto at GPIO-HIGH and may not
# switch fully off, causing ghost steps.
WIRING = "common_anode"

_ON  = 0 if WIRING == "common_anode" else 1   # GPIO level that turns an opto ON
_OFF = 1 - _ON                                # GPIO level that turns an opto OFF

# ── Step timing ───────────────────────────────────────────────────────────────
# The hybrid sleep+busy-wait timer (see _sleep_until) holds accurate periods well
# past the old ~1 kHz soft-sleep ceiling.  If you push CRUISE_SPEED higher and the
# motors stall or lose steps, back it off — a stalled driver feels *slower* than a
# lower speed that never skips.  MIN_SPEED matters as much as CRUISE_SPEED here:
# short d-pad moves rarely reach cruise, so a brisk ramp floor keeps them snappy.
CRUISE_SPEED = 1600   # steps/s at full speed
MIN_SPEED    = 500    # steps/s at the start/end of each move (ramp endpoints)
ACCEL_STEPS  = 50     # steps used to ramp up to cruise (same count for ramp down)
PULSE_US     = 5      # PUL+ high time (µs); TB6600 spec is ≥ 2.5 µs

# ── End-effector control ──────────────────────────────────────────────────────
KEY_STEP    = 0.2    # robot units per d-pad press
STICK_SPEED = 0.8    # robot units/s when the stick is fully deflected
DEADZONE    = 0.15   # ignore stick deflections smaller than this (normalised 0–1)
LOOP_HZ     = 30     # main-loop polling rate (Hz)


# ─────────────────────────────────────────────────────────────────────────────
# IK — ported directly from ik_test.ino / sim/simulator.py
# ─────────────────────────────────────────────────────────────────────────────

def _circle_intersections(x0, y0, r0, x1, y1, r1):
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)
    if d == 0.0 or d > r0 + r1 + 1e-6 or d < abs(r0 - r1) - 1e-6:
        return []
    a  = (r0*r0 - r1*r1 + d*d) / (2.0 * d)
    h  = math.sqrt(max(r0*r0 - a*a, 0.0))
    xm = x0 + a*dx/d
    ym = y0 + a*dy/d
    ux, uy = -dy/d, dx/d
    pts = [(xm + h*ux, ym + h*uy), (xm - h*ux, ym - h*uy)]
    return pts[:1] if h < 1e-6 else pts


def inverse_kinematics(px, py):
    """Return (th_a, th_b) in radians, or None if the point is unreachable."""
    ax, ay = -LINK_BASE / 2.0, 0.0
    bx, by =  LINK_BASE / 2.0, 0.0
    left  = _circle_intersections(ax, ay, LINK_PROX, px, py, LINK_DIST)
    right = _circle_intersections(bx, by, LINK_PROX, px, py, LINK_DIST)
    if not left or not right:
        return None
    e1 = min(left,  key=lambda q: q[0])   # left elbow bends outward (min X)
    e2 = max(right, key=lambda q: q[0])   # right elbow bends outward (max X)
    th_a = math.atan2(e1[1] - ay, e1[0] - ax)
    th_b = math.atan2(e2[1] - by, e2[0] - bx)
    return th_a, th_b


def _angle_to_steps(theta_rad, home_deg, dir_sign):
    delta = theta_rad - math.radians(home_deg)
    return round(dir_sign * delta * STEPS_PER_RAD)


# ─────────────────────────────────────────────────────────────────────────────
# GPIO / motor driver
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


def _setup_gpio():
    # Claim every line in the OFF (idle) state so no opto conducts at startup.
    for pin in (A_STEP, A_DIR, A_ENA, B_STEP, B_DIR, B_ENA):
        lgpio.gpio_claim_output(_h, pin, _OFF)


def _enable(on: bool):
    # TB6600 ENA opto ON = driver outputs DISABLED (coils released).  So to
    # *enable* the motor we leave the ENA opto OFF, and vice-versa.
    level = _OFF if on else _ON
    lgpio.gpio_write(_h, A_ENA, level)
    lgpio.gpio_write(_h, B_ENA, level)


def _sleep_until(target):
    """Block until perf_counter() reaches `target` (seconds).

    Plain time.sleep() has ~50–100 µs of scheduler-wakeup jitter on Linux, which
    both blurs step spacing and caps the achievable rate.  Here we coarse-sleep the
    bulk of the interval (cheap, yields the CPU) and busy-wait only the final ~150 µs
    for accuracy — smooth timing without pegging a core at low step rates.
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
    # Positive direction drives the DIR opto ON; flip DIR_A / DIR_B if a motor
    # turns the wrong way.
    lgpio.gpio_write(_h, A_DIR, _ON if sign_a * DIR_A > 0 else _OFF)
    lgpio.gpio_write(_h, B_DIR, _ON if sign_b * DIR_B > 0 else _OFF)
    time.sleep(20e-6)

    err_a = major // 2
    err_b = major // 2

    # Accumulate step deadlines against a single clock so timing can't drift:
    # any overrun on one step is absorbed by the next, not compounded.
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
# Mapping for PS3 DualShock 3 via USB on Linux (hid-sony driver):
#   D-pad hat:   ABS_HAT0X  (−1 = left,  0 = centre, +1 = right)
#                ABS_HAT0Y  (−1 = up,    0 = centre, +1 = down)  ← Linux convention
#   Left stick:  ABS_X, ABS_Y  (raw 0–255, centre ≈ 128)
#   Buttons:     BTN_START (start), BTN_SELECT (select)

_state = {
    'dpad_x':  0,
    'dpad_y':  0,
    'stick_x': 0.0,
    'stick_y': 0.0,
    'home':    False,
    'quit':    False,
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
                    elif ev.code == 'ABS_X':     _state['stick_x'] = _norm_axis(ev.state)
                    elif ev.code == 'ABS_Y':     _state['stick_y'] = _norm_axis(ev.state)
                elif ev.ev_type == 'Key':
                    if   ev.code == 'BTN_START'  and ev.state == 1: _state['home'] = True
                    elif ev.code == 'BTN_SELECT' and ev.state == 1: _state['quit'] = True


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _home_xy():
    return 0.0, (LINK_PROX + LINK_DIST) * 0.6


def _command(x, y):
    """Run IK for (x, y) and drive motors there.  Returns True if reachable."""
    sol = inverse_kinematics(x, y)
    if sol is None:
        return False
    ta = _angle_to_steps(sol[0], THETA_A_HOME_DEG, DIR_A)
    tb = _angle_to_steps(sol[1], THETA_B_HOME_DEG, DIR_B)
    _move_to(ta, tb)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _setup_gpio()
    _enable(True)

    threading.Thread(target=_input_thread, daemon=True).start()
    print("[ik_test] ready — D-pad to move  |  Start = home  |  Select = quit")

    tick      = 1.0 / LOOP_HZ
    prev_dpad = {'x': 0, 'y': 0}

    try:
        while True:
            t0 = time.monotonic()

            with _state_lock:
                s = dict(_state)
                _state['home'] = False   # consume one-shot flags

            if s['quit']:
                print("[ik_test] quit")
                break

            if s['home']:
                tx, ty = _home_xy()
                print(f"[target] home → ({tx:.2f}, {ty:.2f})")
                _command(tx, ty)
                prev_dpad = {'x': 0, 'y': 0}
                time.sleep(tick)
                continue

            moved = False
            dx, dy = s['dpad_x'], s['dpad_y']

            # D-pad: fire once per press (rising-edge detection).
            if dx != 0 and dx != prev_dpad['x']:
                nx = tx + KEY_STEP * dx
                if inverse_kinematics(nx, ty):
                    tx = nx; moved = True
                else:
                    print("[target] unreachable — ignored")

            if dy != 0 and dy != prev_dpad['y']:
                # Hat Y: −1 = up → pen Y increases; +1 = down → pen Y decreases.
                ny = ty - KEY_STEP * dy
                if inverse_kinematics(tx, ny):
                    ty = ny; moved = True
                else:
                    print("[target] unreachable — ignored")

            prev_dpad = {'x': dx, 'y': dy}

            # Left stick: continuous movement, only active when d-pad is neutral.
            if not moved and dx == 0 and dy == 0:
                sx, sy = s['stick_x'], s['stick_y']
                if abs(sx) > DEADZONE or abs(sy) > DEADZONE:
                    nx = tx + sx * STICK_SPEED * tick
                    ny = ty - sy * STICK_SPEED * tick   # stick up = −Y raw → +Y robot
                    if inverse_kinematics(nx, ny):
                        tx, ty = nx, ny; moved = True

            if moved:
                print(f"[target] ({tx:.2f}, {ty:.2f})")
                _command(tx, ty)

            elapsed = time.monotonic() - t0
            if elapsed < tick:
                time.sleep(tick - elapsed)

    except KeyboardInterrupt:
        print("\n[ik_test] interrupted")
    finally:
        _enable(False)
        lgpio.gpiochip_close(_h)
        print("[ik_test] motors disabled, GPIO released")


if __name__ == '__main__':
    main()
