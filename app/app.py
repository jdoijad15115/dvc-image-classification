"""Flask web app serving the DVC-trained handwritten-digit image classifier.

Routes
  GET  /          web UI: draw a digit or upload an image
  POST /predict   JSON {"image": "<data-URL or base64 PNG/JPG>"} or multipart file "file"
  GET  /health    liveness probe + model metadata
"""
import base64
import io
import json
import os
import pickle

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE, "..", "models", "model.pkl"))
METRICS_PATH = os.environ.get("METRICS_PATH", os.path.join(BASE, "..", "metrics.json"))

app = Flask(__name__)
with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)
try:
    METRICS = json.load(open(METRICS_PATH))
except (OSError, ValueError):
    METRICS = {}


def preprocess(img: Image.Image) -> np.ndarray:
    """Convert any image into the 8x8, 0..1 feature vector the model was trained on.

    The training digits are light strokes on a dark background that fill the full
    8-pixel height (no margin), so: grayscale -> invert if the background is light ->
    crop to the stroke bounding box -> centre on a square (aspect kept, no margin) ->
    area-average down to 8x8 -> scale to 0..1. Tuned on the real test images.
    """
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img.convert("RGBA"))
    g = ImageOps.grayscale(img)
    arr = np.asarray(g, dtype=np.float32)
    if arr.mean() > 127:                       # dark ink on light paper -> invert
        arr = 255.0 - arr
    mask = arr > 50
    if mask.any():
        ys, xs = np.where(mask)
        arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = arr.shape
    side = max(h, w)                           # square canvas, digit touches the edges
    canvas = np.zeros((side, side), dtype=np.float32)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = arr
    small = Image.fromarray(canvas.astype(np.uint8)).resize((8, 8), Image.BOX)
    x = np.asarray(small, dtype=np.float32) / 255.0
    if x.max() > 0:
        x = x / x.max()                        # match training contrast (max = 16/16)
    return x.reshape(1, -1)


def decode_request_image():
    if "file" in request.files:
        return Image.open(request.files["file"].stream)
    payload = request.get_json(silent=True) or {}
    data = payload.get("image", "")
    if "," in data:
        data = data.split(",", 1)[1]
    if not data:
        raise ValueError("no image supplied")
    return Image.open(io.BytesIO(base64.b64decode(data)))


@app.get("/")
def index():
    return render_template("index.html", metrics=METRICS)


@app.get("/health")
def health():
    return jsonify(status="ok", model=type(MODEL).__name__,
                   hidden_units=MODEL.hidden_layer_sizes, metrics=METRICS,
                   host=os.environ.get("HOSTNAME", ""))


@app.post("/predict")
def predict():
    try:
        x = preprocess(decode_request_image())
    except Exception as e:  # bad upload -> 400, not 500
        return jsonify(error=f"could not read image: {e}"), 400
    proba = MODEL.predict_proba(x)[0]
    order = np.argsort(proba)[::-1]
    return jsonify(
        prediction=int(order[0]),
        confidence=round(float(proba[order[0]]), 4),
        top3=[{"digit": int(i), "p": round(float(proba[i]), 4)} for i in order[:3]],
        pixels=x.reshape(8, 8).round(3).tolist(),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
