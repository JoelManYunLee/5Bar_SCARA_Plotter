"""
Step 2 — photo to plottable strokes.

Reads an image (anything Pillow can open, including iPhone HEIC if pillow-heif
is installed) and converts it into a set of 2D polylines a pen can draw. Outputs:

  * an SVG  (standard vector format; opens in a browser, Inkscape, or vpype)
  * a PNG   (quick preview of exactly what the pen will trace)
  * a JSON  (optional; raw polylines for step 3 to consume directly)

Coordinates are in processed-image pixel space, origin top-left, y down. Step 3
handles fitting these into the five-bar workspace, flipping Y, the inverse
kinematics, and streaming moves — so only the shape/aspect ratio matters here.

CLI:
    python process.py uploads/PHOTO.jpg --preview out.png --svg out.svg
    python process.py uploads/PHOTO.jpg --mode threshold --simplify 2.0
"""

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_OK = True
except Exception:
    HEIC_OK = False


@dataclass
class Params:
    mode: str = "edges"        # "edges" (sketch-like) or "threshold" (stencil-like)
    max_dim: int = 1200        # longest side processed at this many px (speed + detail control)
    blur: int = 5              # gaussian kernel, odd; <3 disables
    canny_low: int = 0         # 0,0 => auto thresholds from image median
    canny_high: int = 0
    min_length: float = 14.0   # drop strokes shorter than this (px) -> kills speckle
    simplify: float = 1.4      # douglas-peucker epsilon (px); 0 disables
    invert: bool = False       # threshold mode: trace light regions instead of dark
    order: bool = True         # greedy reorder to cut pen-up travel
    order_cap: int = 3000      # skip ordering above this many strokes (keeps it fast)


def load_gray(path: str, max_dim: int) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)        # honor phone EXIF rotation
    img = img.convert("L")
    w, h = img.size
    if max_dim and max(w, h) > max_dim:
        s = max_dim / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return np.asarray(img)


def _blur(gray: np.ndarray, k: int) -> np.ndarray:
    if k and k >= 3:
        k |= 1
        return cv2.GaussianBlur(gray, (k, k), 0)
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


def _to_paths(contours, p: Params):
    out = []
    for c in contours:
        if len(c) < 2 or cv2.arcLength(c, False) < p.min_length:
            continue
        if p.simplify > 0:
            c = cv2.approxPolyDP(c, p.simplify, False)
        pts = c.reshape(-1, 2).astype(float)
        if len(pts) >= 2:
            out.append(pts)
    return out


def _order(paths, cap: int):
    """Greedy nearest-neighbour ordering from the origin to shrink pen-up travel."""
    if not paths or len(paths) > cap:
        return paths
    remaining = list(paths)
    ordered, cur = [], np.array([0.0, 0.0])
    while remaining:
        best_i, flip, best_d = 0, False, float("inf")
        for i, pth in enumerate(remaining):
            ds = np.hypot(*(pth[0] - cur))
            de = np.hypot(*(pth[-1] - cur))
            if ds < best_d:
                best_d, best_i, flip = ds, i, False
            if de < best_d:
                best_d, best_i, flip = de, i, True
        pth = remaining.pop(best_i)
        if flip:
            pth = pth[::-1]
        ordered.append(pth)
        cur = pth[-1]
    return ordered


def photo_to_paths(path: str, p: Params):
    gray = load_gray(path, p.max_dim)
    h, w = gray.shape
    contours = _threshold(gray, p) if p.mode == "threshold" else _edges(gray, p)
    paths = _to_paths(contours, p)
    if p.order:
        paths = _order(paths, p.order_cap)
    return paths, (w, h)


def path_stats(paths):
    pts = sum(len(p) for p in paths)
    draw = 0.0
    travel = 0.0
    cur = np.array([0.0, 0.0])
    for p in paths:
        travel += float(np.hypot(*(p[0] - cur)))
        d = np.diff(p, axis=0)
        draw += float(np.hypot(d[:, 0], d[:, 1]).sum())
        cur = p[-1]
    return {"strokes": len(paths), "points": pts,
            "draw_px": round(draw, 1), "penup_px": round(travel, 1)}


def write_svg(paths, w, h, path):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g fill="none" stroke="black" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">']
    for p in paths:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in p)
        parts.append(f'<polyline points="{pts}"/>')
    parts.append("</g></svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts))


def write_preview(paths, w, h, path):
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    for p in paths:
        if len(p) >= 2:
            d.line([(float(x), float(y)) for x, y in p], fill=(20, 20, 20), width=1)
    img.save(path)


def write_json(paths, w, h, path):
    data = {"width": w, "height": h,
            "paths": [[[round(float(x), 2), round(float(y), 2)] for x, y in p] for p in paths]}
    with open(path, "w") as f:
        json.dump(data, f)


def send_to_device(paths, w: int, h: int, ip: str, port: int = 9000, timeout: float = 5.0) -> bool:
    """POST stroke JSON to http://<ip>:<port>/plot. Returns True on HTTP 2xx."""
    import urllib.request
    import urllib.error
    body = json.dumps(
        {"width": w, "height": h,
         "paths": [[[round(float(x), 2), round(float(y), 2)] for x, y in p] for p in paths]}
    ).encode()
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


def main():
    ap = argparse.ArgumentParser(description="Convert a photo into plottable strokes.")
    ap.add_argument("input")
    ap.add_argument("--mode", choices=["edges", "threshold"], default="edges")
    ap.add_argument("--max-dim", type=int, default=1200)
    ap.add_argument("--blur", type=int, default=5)
    ap.add_argument("--canny-low", type=int, default=0)
    ap.add_argument("--canny-high", type=int, default=0)
    ap.add_argument("--min-length", type=float, default=14.0)
    ap.add_argument("--simplify", type=float, default=1.4)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--no-order", action="store_true")
    ap.add_argument("--svg")
    ap.add_argument("--preview")
    ap.add_argument("--json")
    ap.add_argument("--send-to", metavar="IP[:PORT]",
                    help="Forward stroke JSON to http://<ip>:<port>/plot after processing")
    args = ap.parse_args()

    p = Params(mode=args.mode, max_dim=args.max_dim, blur=args.blur,
               canny_low=args.canny_low, canny_high=args.canny_high,
               min_length=args.min_length, simplify=args.simplify,
               invert=args.invert, order=not args.no_order)

    paths, (w, h) = photo_to_paths(args.input, p)

    base = os.path.splitext(os.path.basename(args.input))[0]
    svg = args.svg or f"{base}_strokes.svg"
    preview = args.preview or f"{base}_preview.png"
    write_svg(paths, w, h, svg)
    write_preview(paths, w, h, preview)
    if args.json:
        write_json(paths, w, h, args.json)

    s = path_stats(paths)
    print(f"mode={p.mode}  size={w}x{h}  heic={'on' if HEIC_OK else 'off'}")
    print(f"strokes={s['strokes']}  points={s['points']}  "
          f"draw={s['draw_px']:.0f}px  pen-up={s['penup_px']:.0f}px")
    print(f"svg={svg}  preview={preview}" + (f"  json={args.json}" if args.json else ""))

    if args.send_to:
        host, _, port_str = args.send_to.partition(":")
        port = int(port_str) if port_str else 9000
        ok = send_to_device(paths, w, h, host, port)
        print(f"send-to {args.send_to}: {'ok' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
