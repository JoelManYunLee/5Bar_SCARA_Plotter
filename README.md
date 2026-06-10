# Photo Plotter — pipeline

Phone photo → server → plottable strokes → (later) five-bar inverse kinematics → pen.

```
step 1  app.py        capture page + upload server   -> saves to uploads/
step 2  process.py    photo -> strokes (SVG/PNG/JSON) <- reads from uploads/
step 3  (next)        fit to workspace, IK, stream to the WIO Lite
```

## Install (on the Pi)

```bash
pip install -r requirements.txt
```

`opencv-python-headless` is the right OpenCV build for a headless Pi (no GUI deps).
`pillow-heif` is optional but recommended — it lets the server-side processing read
`.heic` photos that iPhones produce.

## Step 1 — get a photo onto the Pi

```bash
python app.py
```

Listens on `0.0.0.0:5000`. On your phone (same Wi-Fi), open `http://<pi-ip>:5000`
(`hostname -I` gives the IP). **Take Photo** opens the camera; **Choose File** picks
from the gallery. Files land in `uploads/` as `YYYYMMDD-HHMMSS_<id>.<ext>`.

Plain HTTP is fine — the page uses a native file/camera input, which doesn't require
HTTPS (a live in-page camera feed would).

## Step 2 — turn a photo into strokes

```bash
python process.py uploads/PHOTO.jpg
```

Writes `PHOTO_strokes.svg` (vector) and `PHOTO_preview.png` (what the pen traces).
Add `--json out.json` for raw polylines that step 3 can read directly.

Output coordinates are processed-image pixels (origin top-left, y down). Only the
shape and aspect ratio matter; step 3 scales them into the physical drawing area,
flips Y, and runs the inverse kinematics.

### Two styles

- `--mode edges` (default) — Canny edge tracing. Sketch-like outlines. Best general
  choice for photos.
- `--mode threshold` — Otsu light/dark split. Bold, graphic, stencil-like. Good for
  high-contrast subjects and logos. Add `--invert` to trace light regions instead.

### Tuning knobs

| flag | what it does | when to reach for it |
|------|--------------|----------------------|
| `--blur N` | gaussian smoothing (odd, default 5) | raise to 7–11 to calm busy texture |
| `--min-length N` | drop strokes shorter than N px (default 14) | raise to 20–30 to kill speckle |
| `--simplify N` | path point reduction (default 1.4) | raise for fewer points / smoother |
| `--canny-low / --canny-high` | manual edge thresholds (default: auto) | when auto under/over-detects |
| `--max-dim N` | longest processed side (default 1200) | lower = faster, coarser detail |
| `--no-order` | skip pen-up travel optimization | rarely; ordering is usually a win |

High-contrast subjects look great at defaults. Low-contrast, heavily textured photos
benefit from something like `--blur 7 --min-length 22`. It's worth eyeballing the
preview PNG and adjusting before committing to a long plot.

The greedy stroke ordering reorders paths to shrink pen-up travel between strokes;
for global optimization you can also run the SVG through `vpype` (`linesort`,
`linemerge`, `linesimplify`) before step 3.

## What's next

Step 3 reads the strokes, fits them into the well-conditioned interior of the
five-bar workspace, converts each finely-segmented point to the two joint angles,
and streams step targets to the WIO Lite over serial.
