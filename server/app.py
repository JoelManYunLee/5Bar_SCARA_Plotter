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

import os
import sys
import time
import uuid

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from process import Params, photo_to_paths, write_svg, write_preview, write_json, path_stats, send_to_device

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUTS_FOLDER = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

# Set PLOTTER_IP env var to enable forwarding: PLOTTER_IP=192.168.1.50 python app.py
PLOTTER_IP = os.environ.get("PLOTTER_IP")
PLOTTER_PORT = int(os.environ.get("PLOTTER_PORT", "9000"))


# Phone photos are big; allow generous headroom. HEIC/large JPEGs included.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".heic", ".heif"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


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
        paths, (w, h) = photo_to_paths(path, Params())
        write_svg(paths, w, h, svg_path)
        write_preview(paths, w, h, preview_path)
        write_json(paths, w, h, json_path)
        stats = path_stats(paths)

        forwarded = None
        if PLOTTER_IP:
            forwarded = send_to_device(paths, w, h, PLOTTER_IP, PLOTTER_PORT)

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
    app.run(host="0.0.0.0", port=5000, debug=False)