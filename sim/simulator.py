"""
Five-bar parallel SCARA simulator.

A five-bar linkage has two base motors a fixed distance apart.  Each motor
drives a *proximal* link; a *distal* link hangs off each proximal link, and the
two distal links meet at the shared end effector (the pen).

        E1 ●───────────● E2          E1, E2 : elbows (proximal/distal joints)
           \\          /
            \\        /               P : end effector (pen)
             \\      /
              ●────●                  A, B : motor bases (distance `d` apart)
              A    B

Controls (interactive matplotlib window)
  Mode .......... Forward Kinematics  -> drive the two motor angles (theta1/2)
                  Inverse Kinematics  -> drive the end-effector X/Y
                  (in IK mode you can also drag the pen directly in the plot)
  d ............. distance between the two motors
  L1 ............ proximal link length (motor -> elbow)
  L2 ............ distal link length   (elbow -> pen)
  Trace ......... record the pen path as you move it (toggle)
  Clear ......... erase the recorded trace
  Workspace ..... flood the reachable area to show the robot's range

CLI (initial values; everything is still adjustable in the window)
  python simulator.py --mode ik --distance 8 --proximal 6 --distal 9
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button


# ── geometry helpers ──────────────────────────────────────────────────────────
def circle_intersections(c0, r0, c1, r1):
    """Return the (0, 1 or 2) intersection points of two circles."""
    x0, y0 = c0
    x1, y1 = c1
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)
    if d == 0.0:
        return []                       # concentric -> none / infinite
    if d > r0 + r1 + 1e-9:
        return []                       # too far apart
    if d < abs(r0 - r1) - 1e-9:
        return []                       # one circle inside the other
    a = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
    h2 = r0 * r0 - a * a
    h = math.sqrt(max(h2, 0.0))
    xm = x0 + a * dx / d
    ym = y0 + a * dy / d
    ux, uy = -dy / d, dx / d            # unit vector perpendicular to the centre line
    return [(xm + h * ux, ym + h * uy),
            (xm - h * ux, ym - h * uy)]


@dataclass
class FiveBar:
    """Pure geometry of the five-bar mechanism (no plotting)."""

    d: float          # distance between the two motors
    l1: float         # proximal link length
    l2: float         # distal link length

    @property
    def left_base(self):
        return (-self.d / 2.0, 0.0)

    @property
    def right_base(self):
        return (self.d / 2.0, 0.0)

    def forward(self, th1, th2):
        """Angles (rad) -> (E1, E2, P).  Returns None if unreachable."""
        ax, ay = self.left_base
        bx, by = self.right_base
        e1 = (ax + self.l1 * math.cos(th1), ay + self.l1 * math.sin(th1))
        e2 = (bx + self.l1 * math.cos(th2), by + self.l1 * math.sin(th2))
        pts = circle_intersections(e1, self.l2, e2, self.l2)
        if not pts:
            return None
        p = max(pts, key=lambda q: q[1])        # take the upper assembly
        return e1, e2, p

    def inverse(self, p):
        """End effector (x, y) -> (th1, th2, E1, E2).  Returns None if unreachable."""
        a = self.left_base
        b = self.right_base
        left = circle_intersections(a, self.l1, p, self.l2)
        right = circle_intersections(b, self.l1, p, self.l2)
        if not left or not right:
            return None
        e1 = min(left, key=lambda q: q[0])      # left elbow bends outward (left)
        e2 = max(right, key=lambda q: q[0])     # right elbow bends outward (right)
        th1 = math.atan2(e1[1] - a[1], e1[0] - a[0])
        th2 = math.atan2(e2[1] - b[1], e2[0] - b[0])
        return th1, th2, e1, e2

    def workspace(self, samples=120):
        """Scatter of reachable pen positions, sweeping both motor angles."""
        angles = np.linspace(0.0, math.pi, samples)
        xs, ys = [], []
        for th1 in angles:
            for th2 in angles:
                sol = self.forward(th1, th2)
                if sol is not None:
                    _, _, p = sol
                    xs.append(p[0])
                    ys.append(p[1])
        return np.array(xs), np.array(ys)


# ── interactive simulator ──────────────────────────────────────────────────────
class Simulator:
    MODE_FK = "Forward (theta)"
    MODE_IK = "Inverse (X/Y)"

    def __init__(self, mode, d, l1, l2):
        self.bot = FiveBar(d=d, l1=l1, l2=l2)
        self.mode = self.MODE_FK if mode == "fk" else self.MODE_IK
        self._guard = False             # prevents slider call-back recursion
        self.tracing = False
        self._dragging = False          # True while the pen is being dragged in IK mode
        self._last_draw_time = 0.0
        self.trace_x: list[float] = []
        self.trace_y: list[float] = []

        # seed a valid pose
        self.th1 = math.radians(60.0)
        self.th2 = math.radians(120.0)
        sol = self.bot.forward(self.th1, self.th2)
        self.ee = sol[2] if sol else (0.0, l1 + l2)

        self._build_ui()
        self._autoscale()
        self._redraw()

    # ---- UI construction -------------------------------------------------------
    def _build_ui(self):
        self.fig = plt.figure(figsize=(11, 7))
        self.fig.canvas.manager.set_window_title("Five-Bar SCARA Simulator")

        self.ax = self.fig.add_axes([0.05, 0.30, 0.62, 0.65])
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, ls=":", alpha=0.5)
        self.ax.set_title("Five-Bar Parallel SCARA")

        # drawable artists (updated in place)
        (self.link_left_prox,) = self.ax.plot([], [], "-o", lw=3, color="#1f77b4")
        (self.link_right_prox,) = self.ax.plot([], [], "-o", lw=3, color="#ff7f0e")
        (self.link_left_dist,) = self.ax.plot([], [], "-o", lw=3, color="#2ca02c")
        (self.link_right_dist,) = self.ax.plot([], [], "-o", lw=3, color="#d62728")
        (self.pen,) = self.ax.plot([], [], "o", ms=10, color="black", zorder=5)
        (self.bases,) = self.ax.plot([], [], "s", ms=10, color="dimgray", zorder=5)
        (self.trace_line,) = self.ax.plot([], [], "-", lw=1, color="purple", alpha=0.8)
        self.workspace_scatter = None

        # mode selector
        mode_ax = self.fig.add_axes([0.72, 0.74, 0.25, 0.16])
        mode_ax.set_title("Mode", fontsize=10)
        self.mode_radio = RadioButtons(
            mode_ax, (self.MODE_FK, self.MODE_IK),
            active=0 if self.mode == self.MODE_FK else 1,
        )
        self.mode_radio.on_clicked(self._on_mode)

        # geometry sliders (always active)
        self.s_d = self._slider([0.72, 0.66, 0.23, 0.03], "d (motor gap)", 1, 30, self.bot.d)
        self.s_l1 = self._slider([0.72, 0.60, 0.23, 0.03], "L1 (proximal)", 1, 20, self.bot.l1)
        self.s_l2 = self._slider([0.72, 0.54, 0.23, 0.03], "L2 (distal)", 1, 25, self.bot.l2)
        for s in (self.s_d, self.s_l1, self.s_l2):
            s.on_changed(self._on_geometry)

        # forward-kinematics sliders
        self.s_t1 = self._slider([0.10, 0.18, 0.50, 0.03], "theta1 (deg)", 0, 180,
                                 math.degrees(self.th1))
        self.s_t2 = self._slider([0.10, 0.13, 0.50, 0.03], "theta2 (deg)", 0, 180,
                                 math.degrees(self.th2))
        self.s_t1.on_changed(self._on_fk)
        self.s_t2.on_changed(self._on_fk)

        # inverse-kinematics sliders
        self.s_x = self._slider([0.10, 0.18, 0.50, 0.03], "pen X", -25, 25, self.ee[0])
        self.s_y = self._slider([0.10, 0.13, 0.50, 0.03], "pen Y", 0, 40, self.ee[1])
        self.s_x.on_changed(self._on_ik)
        self.s_y.on_changed(self._on_ik)

        # action buttons
        self.b_trace = Button(self.fig.add_axes([0.72, 0.44, 0.11, 0.05]), "Trace: off")
        self.b_clear = Button(self.fig.add_axes([0.84, 0.44, 0.11, 0.05]), "Clear")
        self.b_work = Button(self.fig.add_axes([0.72, 0.37, 0.23, 0.05]), "Show workspace")
        self.b_trace.on_clicked(self._on_trace_toggle)
        self.b_clear.on_clicked(self._on_clear)
        self.b_work.on_clicked(self._on_workspace)

        self.status = self.fig.text(0.05, 0.02, "", fontsize=9, family="monospace")

        # drag the pen directly in IK mode
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

        self._sync_slider_visibility()

    def _slider(self, rect, label, vmin, vmax, val):
        return Slider(self.fig.add_axes(rect), label, vmin, vmax, valinit=val)

    # ---- visibility of mode-specific sliders -----------------------------------
    def _sync_slider_visibility(self):
        # The FK and IK sliders share the same screen rectangles, so the hidden
        # pair must also be *deactivated* — set_visible() alone does not stop a
        # widget from receiving mouse events, which causes overlapping sliders to
        # fight over the mouse grab ("Another Axes already grabs mouse input").
        fk = self.mode == self.MODE_FK
        for s, on in ((self.s_t1, fk), (self.s_t2, fk),
                      (self.s_x, not fk), (self.s_y, not fk)):
            s.ax.set_visible(on)
            s.set_active(on)
        self.fig.canvas.draw_idle()

    # ---- callbacks -------------------------------------------------------------
    def _on_mode(self, label):
        self.mode = label
        self._sync_slider_visibility()
        self._autoscale()
        self._redraw()

    def _on_geometry(self, _):
        if self._guard:
            return
        self.bot.d = self.s_d.val
        self.bot.l1 = self.s_l1.val
        self.bot.l2 = self.s_l2.val
        self._autoscale()
        self._redraw()

    def _on_fk(self, _):
        if self._guard or self.mode != self.MODE_FK:
            return
        self.th1 = math.radians(self.s_t1.val)
        self.th2 = math.radians(self.s_t2.val)
        self._redraw()

    def _on_ik(self, _):
        if self._guard or self.mode != self.MODE_IK:
            return
        self.ee = (self.s_x.val, self.s_y.val)
        self._redraw()

    # ---- pen dragging (IK mode) ------------------------------------------------
    def _on_press(self, event):
        # only grab the pen with the left button, in IK mode, inside the plot
        if (self.mode != self.MODE_IK or event.inaxes is not self.ax
                or event.button != 1 or event.xdata is None):
            return
        reach = self.bot.l1 + self.bot.l2 + self.bot.d
        if math.hypot(event.xdata - self.ee[0], event.ydata - self.ee[1]) < reach * 0.06:
            self._dragging = True
            self._move_pen(event.xdata, event.ydata)

    _DRAG_INTERVAL = 1 / 60  # cap redraws at ~30 fps during drag

    def _on_motion(self, event):
        if not self._dragging or event.inaxes is not self.ax or event.xdata is None:
            return
        now = time.monotonic()
        if now - self._last_draw_time < self._DRAG_INTERVAL:
            return
        self._last_draw_time = now
        self._move_pen(event.xdata, event.ydata)

    def _on_release(self, _):
        self._dragging = False

    def _move_pen(self, x, y):
        """Drive the pen to (x, y) if reachable, keeping the IK sliders in sync."""
        if self.bot.inverse((x, y)) is None:
            self.pen.set_visible(False)
            self.status.set_text("  pose unreachable — outside workspace")
            self.fig.canvas.draw_idle()
            return
        self.pen.set_visible(True)
        self.ee = (x, y)
        self._guard = True              # the slider writes below must not re-enter
        events_x, draws_x = self.s_x.eventson, self.s_x.drawon
        events_y, draws_y = self.s_y.eventson, self.s_y.drawon
        self.s_x.eventson = self.s_x.drawon = False
        self.s_y.eventson = self.s_y.drawon = False
        try:
            self.s_x.set_val(x)
            self.s_y.set_val(y)
        finally:
            self.s_x.eventson, self.s_x.drawon = events_x, draws_x
            self.s_y.eventson, self.s_y.drawon = events_y, draws_y
            self._guard = False
        self._redraw()

    def _on_trace_toggle(self, _):
        self.tracing = not self.tracing
        self.b_trace.label.set_text(f"Trace: {'on' if self.tracing else 'off'}")
        self.fig.canvas.draw_idle()

    def _on_clear(self, _):
        self.trace_x.clear()
        self.trace_y.clear()
        self.trace_line.set_data([], [])
        self.fig.canvas.draw_idle()

    def _on_workspace(self, _):
        xs, ys = self.bot.workspace()
        if self.workspace_scatter is not None:
            self.workspace_scatter.remove()
        self.workspace_scatter = self.ax.scatter(
            xs, ys, s=2, color="lightgray", alpha=0.5, zorder=0
        )
        self.fig.canvas.draw_idle()

    # ---- core update -----------------------------------------------------------
    def _redraw(self):
        if self.mode == self.MODE_FK:
            sol = self.bot.forward(self.th1, self.th2)
            if sol is None:
                self._show_unreachable()
                return
            e1, e2, p = sol
            self.ee = p
        else:
            sol = self.bot.inverse(self.ee)
            if sol is None:
                self._show_unreachable()
                return
            self.th1, self.th2, e1, e2 = sol[0], sol[1], sol[2], sol[3]
            p = self.ee

        a = self.bot.left_base
        b = self.bot.right_base

        self.link_left_prox.set_data([a[0], e1[0]], [a[1], e1[1]])
        self.link_right_prox.set_data([b[0], e2[0]], [b[1], e2[1]])
        self.link_left_dist.set_data([e1[0], p[0]], [e1[1], p[1]])
        self.link_right_dist.set_data([e2[0], p[0]], [e2[1], p[1]])
        self.pen.set_data([p[0]], [p[1]])
        self.bases.set_data([a[0], b[0]], [a[1], b[1]])

        if self.tracing:
            self.trace_x.append(p[0])
            self.trace_y.append(p[1])
            self.trace_line.set_data(self.trace_x, self.trace_y)

        self._sync_readback_sliders()
        self.status.set_text(
            f"theta1={math.degrees(self.th1):6.1f}  "
            f"theta2={math.degrees(self.th2):6.1f}   "
            f"pen=({p[0]:6.2f}, {p[1]:6.2f})   "
            f"d={self.bot.d:.1f}  L1={self.bot.l1:.1f}  L2={self.bot.l2:.1f}"
        )
        self.fig.canvas.draw_idle()

    def _sync_readback_sliders(self):
        """Push computed values into the *inactive* sliders without re-triggering.

        Each Slider.set_val() normally fires its observers *and* schedules a full
        figure redraw.  Doing that here — once per drag event, for two sliders —
        stacked extra redraws on top of the one we already issue in _redraw(),
        which made dragging unbearably slow and eventually wedged the event loop.
        We silence both by toggling the sliders' eventson/drawon flags.
        """
        if self.mode == self.MODE_FK:
            targets = ((self.s_x, self.ee[0]), (self.s_y, self.ee[1]))
        else:
            targets = ((self.s_t1, math.degrees(self.th1)),
                       (self.s_t2, math.degrees(self.th2)))
        for slider, value in targets:
            events, draws = slider.eventson, slider.drawon
            slider.eventson = slider.drawon = False
            try:
                slider.set_val(value)
            finally:
                slider.eventson, slider.drawon = events, draws

    def _autoscale(self):
        reach = self.bot.l1 + self.bot.l2 + self.bot.d
        self.ax.set_xlim(-reach * 0.7, reach * 0.7)
        self.ax.set_ylim(-self.bot.l1 * 1.2, reach)

    def _show_unreachable(self):
        self.status.set_text("  pose unreachable with current geometry")
        self.pen.set_color("red")
        self.fig.canvas.draw_idle()
        self.pen.set_color("black")


# ── entry point ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Five-bar parallel SCARA simulator")
    parser.add_argument("--mode", choices=["fk", "ik"], default="fk",
                        help="initial control mode: fk = drive angles, ik = drive pen")
    parser.add_argument("--distance", type=float, default=8.0,
                        help="distance between the two motors")
    parser.add_argument("--proximal", type=float, default=6.0,
                        help="proximal link length (L1)")
    parser.add_argument("--distal", type=float, default=9.0,
                        help="distal link length (L2)")
    args = parser.parse_args()

    # Keep a reference to the Simulator alive for the lifetime of the window.
    # matplotlib widgets stop responding if the object owning them is garbage
    # collected, and plt.show() only keeps the *figure* alive, not this instance.
    sim = Simulator(mode=args.mode, d=args.distance, l1=args.proximal, l2=args.distal)
    plt.show()
    return sim


if __name__ == "__main__":
    main()
