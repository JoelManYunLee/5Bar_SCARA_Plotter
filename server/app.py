"""
Plotter upload server

Serves a mobile capture page, receives photos from your phone, runs the
process pipeline to generate stroke vectors, and optionally forwards them
to a plotter device.

Run on the Pi:
    pip install -r requirements.txt
    python app.py
    # With plotter forwarding:
    PLOTTER_IP=192.168.1.50 python app.py
    PLOTTER_IP=192.168.1.50 PLOTTER_PORT=9000 python app.py

Then open http://<pi-lan-ip>:5000 on your phone (same Wi-Fi network).
"""

import argparse
import json
import os
import shutil
import sys
import threading
import time
import uuid

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from process import Params, photo_to_paths, write_svg, write_preview, write_json, path_stats, send_to_device, send_motor_command

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUTS_FOLDER = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

# Set PLOTTER_IP env var to enable forwarding: PLOTTER_IP=192.168.1.50 python app.py
PLOTTER_IP = os.environ.get("PLOTTER_IP")
PLOTTER_PORT = int(os.environ.get("PLOTTER_PORT", "9000"))

# Known output-shaft angle where each arm's limit switch engages. Until an arm
# trips its switch its angle is unknown; the trip fixes it to this reference and
# tracking begins from there.
LIMIT_A_DEG = float(os.environ.get("LIMIT_A_DEG", "90"))
LIMIT_B_DEG = float(os.environ.get("LIMIT_B_DEG", "90"))

# Overridden to True by --sim at startup; False when running on real hardware.
SIM_MODE = False

# Persisted motor position (output-shaft degrees) so calibration survives restarts.
# Each arm's angle becomes known once it's homed against its limit switch; the
# machine is calibrated when both arms are homed.
POSITION_FILE = os.path.join(BASE_DIR, "position.json")
_position_lock = threading.Lock()


def load_position() -> dict:
    try:
        with open(POSITION_FILE) as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    a_homed = bool(data.get("a_homed", False))
    b_homed = bool(data.get("b_homed", False))
    a_deg = data.get("motor_a_deg")
    b_deg = data.get("motor_b_deg")
    return {
        # Angle is None (unknown) until the arm has tripped its limit switch.
        "motor_a_deg": float(a_deg) if a_homed and a_deg is not None else None,
        "motor_b_deg": float(b_deg) if b_homed and b_deg is not None else None,
        "a_homed":     a_homed,
        "b_homed":     b_homed,
        "calibrated":  a_homed and b_homed,
        "updated":     data.get("updated"),
    }


def save_position(pos: dict) -> dict:
    pos["calibrated"] = bool(pos.get("a_homed")) and bool(pos.get("b_homed"))
    pos["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = POSITION_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(pos, f, indent=2)
    os.replace(tmp, POSITION_FILE)  # atomic write
    return pos


# Phone photos are big; allow generous headroom. HEIC/large JPEGs included.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".heic", ".heif"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def _params_from_request() -> Params:
    """Build Params from form fields sent with the upload, falling back to defaults."""
    f = request.form
    return Params(
        mode          = f.get("mode", "edges"),
        simplify      = float(f.get("simplify", 1.4)),
        multiscale    = f.get("multiscale") == "1",
        shade         = f.get("shade") == "1",
        shade_layers  = int(f.get("shade_layers", 1)),
        shade_spacing = float(f.get("shade_spacing", 6.0)),
        shade_dark    = int(f.get("shade_dark", 180)),
    )


def is_allowed(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


def unique_name(original: str) -> str:
    """A collision-proof, sortable filename: <timestamp>_<short-id><ext>."""
    _, ext = os.path.splitext(secure_filename(original).lower())
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".jpg"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}{ext}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "photo" not in request.files:
        return jsonify(ok=False, error="No photo field in the request."), 400

    file = request.files["photo"]
    if not file or file.filename == "":
        return jsonify(ok=False, error="No file was selected."), 400

    if not is_allowed(file.filename):
        return jsonify(
            ok=False,
            error="That file type isn't an image we accept (use JPG, PNG, WEBP, or HEIC).",
        ), 415

    name = unique_name(file.filename)
    path = os.path.join(UPLOAD_FOLDER, name)
    file.save(path)
    size = os.path.getsize(path)

    base = os.path.splitext(name)[0]
    svg_path     = os.path.join(OUTPUTS_FOLDER, f"{base}.svg")
    preview_path = os.path.join(OUTPUTS_FOLDER, f"{base}_preview.png")
    json_path    = os.path.join(OUTPUTS_FOLDER, f"{base}.json")

    try:
        paths, weights, (w, h) = photo_to_paths(path, _params_from_request())
        write_svg(paths, weights, w, h, svg_path)
        write_preview(paths, weights, w, h, preview_path)
        write_json(paths, weights, w, h, json_path)
        if SIM_MODE:
            shutil.copy2(json_path, os.path.join(OUTPUTS_FOLDER, "latest.json"))
        stats = path_stats(paths, weights)

        forwarded = None
        if PLOTTER_IP:
            forwarded = send_to_device(paths, weights, w, h, PLOTTER_IP, PLOTTER_PORT)

        processing = dict(
            svg=f"/outputs/{base}.svg",
            preview=f"/outputs/{base}_preview.png",
            json=f"/outputs/{base}.json",
            stats=stats,
            forwarded=forwarded,
        )
    except Exception as exc:
        processing = dict(error=str(exc))

    return jsonify(
        ok=True,
        filename=name,
        size_bytes=size,
        url=f"/uploads/{name}",
        processing=processing,
    )


@app.route("/motor", methods=["POST"])
def motor():
    """Jog one motor for limit-switch calibration.

    Expects JSON {motor: "A"|"B", direction: "cw"|"ccw", degrees: <number>}
    and forwards it to the plotter device when PLOTTER_IP is set.
    """
    data = request.get_json(silent=True) or {}
    motor_id  = str(data.get("motor", "")).upper()
    direction = str(data.get("direction", "")).lower()

    if motor_id not in ("A", "B"):
        return jsonify(ok=False, error="motor must be 'A' or 'B'."), 400
    if direction not in ("cw", "ccw"):
        return jsonify(ok=False, error="direction must be 'cw' or 'ccw'."), 400
    try:
        degrees = float(data.get("degrees", 5.0))
    except (TypeError, ValueError):
        return jsonify(ok=False, error="degrees must be a number."), 400
    if not 0 < degrees <= 360:
        return jsonify(ok=False, error="degrees must be between 0 and 360."), 400

    forwarded = False
    limit = False
    if PLOTTER_IP:
        result = send_motor_command(motor_id, direction, degrees, PLOTTER_IP, PLOTTER_PORT)
        if result is None:
            return jsonify(ok=False, error="Could not reach the plotter device."), 502
        forwarded = True
        limit = result["limit"]

    # A limit hit fixes the arm's known angle and starts tracking. Before that the
    # angle is unknown; only an already-homed arm accumulates the jog.
    delta = degrees if direction == "cw" else -degrees
    with _position_lock:
        pos = load_position()
        if motor_id == "A":
            if limit:
                pos["motor_a_deg"], pos["a_homed"] = LIMIT_A_DEG, True
            elif pos["a_homed"]:
                pos["motor_a_deg"] = round(pos["motor_a_deg"] + delta, 3)
        else:
            if limit:
                pos["motor_b_deg"], pos["b_homed"] = LIMIT_B_DEG, True
            elif pos["b_homed"]:
                pos["motor_b_deg"] = round(pos["motor_b_deg"] + delta, 3)
        pos = save_position(pos)

    return jsonify(ok=True, forwarded=forwarded, limit=limit, position=pos)


@app.route("/position")
def position():
    """Return the last known motor position and calibration state."""
    with _position_lock:
        return jsonify(ok=True, position=load_position())


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUTS_FOLDER, filename)


@app.errorhandler(RequestEntityTooLarge)
def too_large(_e):
    mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    return jsonify(ok=False, error=f"That photo is over the {mb} MB limit."), 413


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotter upload server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--sim", action="store_true",
                        help="write outputs/latest.json after each upload for the simulator")
    args = parser.parse_args()
    SIM_MODE = args.sim
    app.run(host="0.0.0.0", port=args.port, debug=False)