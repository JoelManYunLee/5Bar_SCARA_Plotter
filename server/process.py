"""
Step 2 — photo to plottable strokes.

Reads an image (anything Pillow can open, including iPhone HEIC if pillow-heif
is installed) and converts it into a set of 2D polylines a pen can draw. Outputs:

  * an SVG  (standard vector format; opens in a browser, Inkscape, or vpype)
  * a PNG   (quick preview; stroke darkness reflects pencil pressure)
  * a JSON  (raw polylines + per-stroke weight for step 3 to consume directly)

Coordinates are in processed-image pixel space, origin top-left, y down. Step 3
handles fitting these into the five-bar workspace, flipping Y, the inverse
kinematics, and streaming moves — so only the shape/aspect ratio matters here.

Each path in the output carries a **weight** (0.0 - 1.0) that maps to physical
pencil pressure: 1.0 = full press (main contours), 0.5 = half press (texture),
≤ 0.7 = light touch (shading hatch lines, proportional to local darkness).

Processing layers (all combinable):
  edges / threshold  — primary contour extraction (always active)
  --multiscale       — second fine-detail Canny pass for hair / fabric texture
  --shade            — luminance hatching to convey tone and shadow depth

CLI:
    python process.py photo.jpg --preview out.png --svg out.svg
    python process.py photo.jpg --shade --shade-layers 2 --multiscale
    python process.py photo.jpg --mode threshold --simplify 2.0
"""

import argparse
import json
import math
import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_OK = True
except Exception:
    HEIC_OK = False


@dataclass
class Params:
    # ── primary edge detection ─────────────────────────────────────────────────
    mode: str = "edges"         # "edges" (sketch) | "threshold" (stencil)
    max_dim: int = 1200         # longest side (px); controls speed vs detail
    blur: int = 5               # gaussian kernel, odd; <3 disables
    canny_low: int = 0          # 0,0 => auto-threshold from median brightness
    canny_high: int = 0
    min_length: float = 14.0    # drop strokes shorter than this (px)
    simplify: float = 1.4       # douglas-peucker epsilon (px); 0 disables
    invert: bool = False        # threshold mode: trace light areas not dark
    order: bool = True          # greedy reorder to minimise pen-up travel
    order_cap: int = 3000       # skip ordering above this many strokes

    # ── fine detail / texture (hair, fabric) ──────────────────────────────────
    multiscale: bool = False    # add a second Canny pass with minimal blur
    fine_blur: int = 1          # blur for fine pass (1 = off, 3 = very light)
    fine_min_length: float = 5.0  # min length for fine-detail strokes (px)

    # ── luminance hatching (shading / depth) ──────────────────────────────────
    shade: bool = False         # add scan-line hatching in dark / mid-tone regions
    shade_angle: float = 45.0   # primary hatch angle (degrees)
    shade_spacing: float = 6.0  # distance between hatch scan lines (px)
    shade_layers: int = 1       # 1 = hatch, 2 = crosshatch (+90°), etc.
    shade_dark: int = 180       # only shade pixels darker than this (0–255)
    shade_min_length: float = 8.0  # min hatch segment length (px)


# ── image loading ─────────────────────────────────────────────────────────────

def load_gray(path: str, max_dim: int) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("L")
    w, h = img.size
    if max_dim and max(w, h) > max_dim:
        s = max_dim / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return np.asarray(img)


# ── edge / contour extraction ─────────────────────────────────────────────────

def _blur(gray: np.ndarray, k: int) -> np.ndarray:
    if k and k >= 3:
        return cv2.GaussianBlur(gray, (k | 1, k | 1), 0)
    return gray


def _auto_canny(gray: np.ndarray, sigma: float = 0.33):
    v = float(np.median(gray))
    return int(max(0, (1 - sigma) * v)), int(min(255, (1 + sigma) * v))


def _edges(gray: np.ndarray, p: Params):
    g = _blur(gray, p.blur)
    lo, hi = (p.canny_low, p.canny_high) if (p.canny_low and p.canny_high) else _auto_canny(g)
    edges = cv2.Canny(g, lo, hi)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _threshold(gray: np.ndarray, p: Params):
    g = _blur(gray, p.blur)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if p.invert:
        bw = cv2.bitwise_not(bw)
    contours, _ = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _to_paths(contours, min_length: float, simplify: float) -> list:
    out = []
    for c in contours:
        if len(c) < 2 or cv2.arcLength(c, False) < min_length:
            continue
        if simplify > 0:
            c = cv2.approxPolyDP(c, simplify, False)
        pts = c.reshape(-1, 2).astype(float)
        if len(pts) >= 2:
            out.append(pts)
    return out


def _fine_edges(gray: np.ndarray, p: Params) -> list:
    """Second Canny pass with minimal blur to capture fine texture (hair, fabric)."""
    g = _blur(gray, p.fine_blur)
    lo, hi = _auto_canny(g, sigma=0.5)   # wider sigma -> catches weaker edges
    edges = cv2.Canny(g, lo, hi)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return _to_paths(contours, p.fine_min_length, simplify=0.5)


# ── luminance hatching ────────────────────────────────────────────────────────

def _hatch_layer(gray: np.ndarray, angle_deg: float, spacing: float,
                 dark_thresh: int, min_len: float):
    """
    Scan-line hatching: sweep lines at `angle_deg` across the image, drawing
    segments wherever pixel brightness < dark_thresh.  Returns (paths, weights)
    where weight ∈ [0.1, 1.0] reflects how dark the local region is.
    """
    h, w = gray.shape
    angle = math.radians(angle_deg)
    ca, sa = math.cos(angle), math.sin(angle)
    # perpendicular unit vector (scan lines run along ca/sa, spaced by perp)
    cp, sp = -sa, ca

    diag = math.hypot(w, h)
    cx, cy = w / 2.0, h / 2.0
    n_lines = int(diag / spacing) + 2

    paths: list = []
    weights: list = []

    for i in range(-n_lines, n_lines + 1):
        # origin of this scan line (perpendicular offset from image centre)
        ox = cx + i * spacing * cp
        oy = cy + i * spacing * sp

        # clip parametric range so samples stay inside the image
        t_min, t_max = -diag, diag
        if abs(ca) > 1e-9:
            ts = sorted([-ox / ca, (w - 1 - ox) / ca])
            t_min = max(t_min, ts[0])
            t_max = min(t_max, ts[1])
        if abs(sa) > 1e-9:
            ts = sorted([-oy / sa, (h - 1 - oy) / sa])
            t_min = max(t_min, ts[0])
            t_max = min(t_max, ts[1])

        if t_max <= t_min + 1:
            continue

        # vectorised: sample all pixels along this scan line at 1-px steps
        t_vals = np.arange(t_min, t_max, 1.0)
        xs = np.round(ox + t_vals * ca).astype(int)
        ys = np.round(oy + t_vals * sa).astype(int)

        brightness = np.full(len(t_vals), 255, dtype=np.uint16)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        brightness[valid] = gray[ys[valid], xs[valid]]

        is_dark = brightness < dark_thresh

        # find contiguous dark runs via edge detection on the boolean mask
        padded = np.concatenate([[False], is_dark, [False]])
        starts = np.where(~padded[:-1] & padded[1:])[0]
        ends   = np.where( padded[:-1] & ~padded[1:])[0]

        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            seg_x = ox + t_vals[s:e] * ca
            seg_y = oy + t_vals[s:e] * sa
            length = math.hypot(seg_x[-1] - seg_x[0], seg_y[-1] - seg_y[0])
            if length < min_len:
                continue
            avg_v = float(brightness[s:e].mean())
            w_val = (dark_thresh - avg_v) / dark_thresh
            paths.append(np.column_stack([seg_x, seg_y]))
            weights.append(max(0.1, min(1.0, w_val)))

    return paths, weights


# ── path ordering ─────────────────────────────────────────────────────────────

def _order(paths: list, weights: list, cap: int):
    """Greedy nearest-neighbour reorder to minimise pen-up travel."""
    if not paths or len(paths) > cap:
        return paths, weights
    pairs = list(zip(paths, weights))
    ordered_p: list = []
    ordered_w: list = []
    cur = np.array([0.0, 0.0])
    while pairs:
        best_i, flip, best_d = 0, False, float("inf")
        for i, (pth, _) in enumerate(pairs):
            ds = float(np.hypot(*(pth[0] - cur)))
            de = float(np.hypot(*(pth[-1] - cur)))
            if ds < best_d:
                best_d, best_i, flip = ds, i, False
            if de < best_d:
                best_d, best_i, flip = de, i, True
        pth, wt = pairs.pop(best_i)
        if flip:
            pth = pth[::-1]
        ordered_p.append(pth)
        ordered_w.append(wt)
        cur = pth[-1]
    return ordered_p, ordered_w


# ── top-level pipeline ────────────────────────────────────────────────────────

def photo_to_paths(path: str, p: Params):
    """
    Convert an image file into plottable strokes.

    Returns (paths, weights, (w, h)) where:
      paths   — list of Nx2 float arrays (pixel coords, origin top-left, y down)
      weights — parallel list of float in [0.1, 1.0] representing pencil pressure
      (w, h)  — processed image dimensions in pixels
    """
    gray = load_gray(path, p.max_dim)
    h, w = gray.shape

    # Layer 1: primary contours (full pressure)
    contours = _threshold(gray, p) if p.mode == "threshold" else _edges(gray, p)
    all_paths = _to_paths(contours, p.min_length, p.simplify)
    all_weights = [1.0] * len(all_paths)

    # Layer 2: fine texture (hair, fabric) — half pressure so it reads lighter
    if p.multiscale:
        fine = _fine_edges(gray, p)
        all_paths.extend(fine)
        all_weights.extend([0.5] * len(fine))

    # Layer 3: luminance hatching for depth / shading
    if p.shade:
        for k in range(p.shade_layers):
            angle = p.shade_angle + k * 90.0
            h_paths, h_weights = _hatch_layer(
                gray, angle, p.shade_spacing, p.shade_dark, p.shade_min_length
            )
            all_paths.extend(h_paths)
            # shading is always lighter than contour lines (cap at 0.7)
            all_weights.extend([min(0.7, hw) for hw in h_weights])

    if p.order:
        all_paths, all_weights = _order(all_paths, all_weights, p.order_cap)

    return all_paths, all_weights, (w, h)


# ── statistics ────────────────────────────────────────────────────────────────

def path_stats(paths, weights=None):
    pts = sum(len(p) for p in paths)
    draw = travel = 0.0
    cur = np.array([0.0, 0.0])
    for p in paths:
        travel += float(np.hypot(*(p[0] - cur)))
        d = np.diff(p, axis=0)
        draw += float(np.hypot(d[:, 0], d[:, 1]).sum())
        cur = p[-1]
    stats = {
        "strokes": len(paths), "points": pts,
        "draw_px": round(draw, 1), "penup_px": round(travel, 1),
    }
    if weights:
        stats["heavy_strokes"] = sum(1 for wt in weights if wt >= 0.7)
        stats["light_strokes"]  = sum(1 for wt in weights if wt < 0.4)
    return stats


# ── output writers ────────────────────────────────────────────────────────────

def write_svg(paths, weights, w, h, path):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g fill="none" stroke="black" stroke-linecap="round" stroke-linejoin="round">',
    ]
    for p, wt in zip(paths, weights):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in p)
        sw  = max(0.3, round(wt * 1.5, 1))
        op  = round(0.3 + wt * 0.7, 2)
        parts.append(f'<polyline points="{pts}" stroke-width="{sw}" opacity="{op}"/>')
    parts.append("</g></svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts))


def write_preview(paths, weights, w, h, path):
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    for p, wt in zip(paths, weights):
        if len(p) >= 2:
            v = max(20, int(255 * (1.0 - wt)))   # darker pixel = heavier stroke
            d.line([(float(x), float(y)) for x, y in p], fill=(v, v, v), width=1)
    img.save(path)


def write_json(paths, weights, w, h, path):
    data = {
        "width": w,
        "height": h,
        "paths":   [[[round(float(x), 2), round(float(y), 2)] for x, y in p] for p in paths],
        "weights": [round(float(wt), 3) for wt in weights],
    }
    with open(path, "w") as f:
        json.dump(data, f)


def send_to_device(paths, weights, w: int, h: int, ip: str,
                   port: int = 9000, timeout: float = 5.0) -> bool:
    """POST stroke JSON (with weights) to http://<ip>:<port>/plot."""
    import urllib.request
    import urllib.error
    body = json.dumps({
        "width":   w,
        "height":  h,
        "paths":   [[[round(float(x), 2), round(float(y), 2)] for x, y in p] for p in paths],
        "weights": [round(float(wt), 3) for wt in weights],
    }).encode()
    req = urllib.request.Request(
        f"http://{ip}:{port}/plot",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status < 400
    except urllib.error.URLError:
        return False


def send_motor_command(motor: str, direction: str, degrees: float, ip: str,
                       port: int = 9000, timeout: float = 5.0):
    """Jog a single motor for calibration.

    POSTs to http://<ip>:<port>/motor so the firmware can turn one motor a fixed
    amount. ``motor`` is "A" or "B", ``direction`` is "cw" or "ccw", ``degrees``
    is the output-shaft rotation. The firmware stops early if that arm's limit
    switch engages. Returns ``{"limit": bool}`` on success, or ``None`` if the
    device is unreachable.
    """
    import urllib.request
    import urllib.error
    body = json.dumps({
        "motor":     motor,
        "direction": direction,
        "degrees":   round(float(degrees), 3),
    }).encode()
    req = urllib.request.Request(
        f"http://{ip}:{port}/motor",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status >= 400:
                return None
            try:
                info = json.loads(resp.read().decode())
            except (ValueError, UnicodeDecodeError):
                info = {}
            return {"limit": bool(info.get("limit", False))}
    except urllib.error.URLError:
        return None


def get_device_status(ip: str, port: int = 9000, timeout: float = 3.0):
    """GET http://<ip>:<port>/position — the firmware's live motor state.

    Returns {"ready", "plotting", "a": {"homed", "deg"}, "b": {"homed", "deg"}}
    on success, or None if the device is unreachable or replies unexpectedly.
    """
    import urllib.request
    import urllib.error
    req = urllib.request.Request(f"http://{ip}:{port}/position", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status >= 400:
                return None
            info = json.loads(resp.read().decode())
    except (urllib.error.URLError, ValueError, UnicodeDecodeError):
        return None
    if not info.get("ok"):
        return None
    return {
        "ready":    bool(info.get("ready", False)),
        "plotting": bool(info.get("plotting", False)),
        "a": info.get("a", {}),
        "b": info.get("b", {}),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Convert a photo into plottable strokes.")
    ap.add_argument("input")

    # primary
    ap.add_argument("--mode", choices=["edges", "threshold"], default="edges")
    ap.add_argument("--max-dim",    type=int,   default=1200)
    ap.add_argument("--blur",       type=int,   default=5)
    ap.add_argument("--canny-low",  type=int,   default=0)
    ap.add_argument("--canny-high", type=int,   default=0)
    ap.add_argument("--min-length", type=float, default=14.0)
    ap.add_argument("--simplify",   type=float, default=1.4)
    ap.add_argument("--invert",     action="store_true")
    ap.add_argument("--no-order",   action="store_true")

    # texture
    ap.add_argument("--multiscale",      action="store_true",
                    help="add fine-detail Canny pass for hair / texture")
    ap.add_argument("--fine-blur",       type=int,   default=1)
    ap.add_argument("--fine-min-length", type=float, default=5.0)

    # shading
    ap.add_argument("--shade",            action="store_true",
                    help="add luminance hatching for depth / shadow")
    ap.add_argument("--shade-angle",      type=float, default=45.0)
    ap.add_argument("--shade-spacing",    type=float, default=6.0)
    ap.add_argument("--shade-layers",     type=int,   default=1,
                    help="1=hatch, 2=crosshatch (+90°)")
    ap.add_argument("--shade-dark",       type=int,   default=180)
    ap.add_argument("--shade-min-length", type=float, default=8.0)

    # outputs
    ap.add_argument("--svg")
    ap.add_argument("--preview")
    ap.add_argument("--json")
    ap.add_argument("--send-to", metavar="IP[:PORT]",
                    help="forward stroke JSON to http://<ip>:<port>/plot")
    args = ap.parse_args()

    p = Params(
        mode=args.mode, max_dim=args.max_dim, blur=args.blur,
        canny_low=args.canny_low, canny_high=args.canny_high,
        min_length=args.min_length, simplify=args.simplify,
        invert=args.invert, order=not args.no_order,
        multiscale=args.multiscale,
        fine_blur=args.fine_blur, fine_min_length=args.fine_min_length,
        shade=args.shade, shade_angle=args.shade_angle,
        shade_spacing=args.shade_spacing, shade_layers=args.shade_layers,
        shade_dark=args.shade_dark, shade_min_length=args.shade_min_length,
    )

    paths, weights, (w, h) = photo_to_paths(args.input, p)

    base    = os.path.splitext(os.path.basename(args.input))[0]
    svg     = args.svg     or f"{base}_strokes.svg"
    preview = args.preview or f"{base}_preview.png"
    write_svg(paths, weights, w, h, svg)
    write_preview(paths, weights, w, h, preview)
    if args.json:
        write_json(paths, weights, w, h, args.json)

    s = path_stats(paths, weights)
    flags = []
    if p.multiscale:
        flags.append("multiscale")
    if p.shade:
        flags.append(f"shade×{p.shade_layers}")
    print(f"mode={p.mode}  size={w}×{h}  heic={'on' if HEIC_OK else 'off'}"
          + (f"  [{', '.join(flags)}]" if flags else ""))
    print(f"strokes={s['strokes']}  points={s['points']}  "
          f"draw={s['draw_px']:.0f}px  pen-up={s['penup_px']:.0f}px"
          + (f"  heavy={s.get('heavy_strokes',0)}  light={s.get('light_strokes',0)}"
             if weights else ""))
    print(f"svg={svg}  preview={preview}" + (f"  json={args.json}" if args.json else ""))

    if args.send_to:
        host, _, port_str = args.send_to.partition(":")
        port = int(port_str) if port_str else 9000
        ok = send_to_device(paths, weights, w, h, host, port)
        print(f"send-to {args.send_to}: {'ok' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
