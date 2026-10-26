"""Automated tests for the DVC image-classification pipeline.

Run after `dvc repro`:
    pytest test.py -v
"""
import json
import os
import pickle

import numpy as np
import pytest
import yaml

PARAMS_FILE = "params.yaml"
TRAIN_FILE = "data/prepared/train.npz"
TEST_FILE = "data/prepared/test.npz"
MODEL_FILE = "models/model.pkl"
METRICS_FILE = "metrics.json"

N_FEATURES = 64          # 8x8 images flattened
N_CLASSES = 10
MIN_ACCURACY = 0.90      # quality gate: the build fails below this


@pytest.fixture(scope="module")
def params():
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def train_data():
    return np.load(TRAIN_FILE)


@pytest.fixture(scope="module")
def test_data():
    return np.load(TEST_FILE)


@pytest.fixture(scope="module")
def model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


# ---------- configuration ----------
def test_params_file_valid(params):
    assert "prepare" in params and "train" in params
    t = params["train"]
    assert t["hidden_units"] > 0
    assert t["activation"] in {"relu", "tanh", "logistic", "identity"}
    assert 0 < t["learning_rate"] < 1
    assert t["alpha"] >= 0
    assert t["batch_size"] > 0 and t["epochs"] > 0
    assert 0 < params["prepare"]["test_size"] < 1


# ---------- data stage ----------
def test_pipeline_outputs_exist():
    for path in (TRAIN_FILE, TEST_FILE, MODEL_FILE, METRICS_FILE):
        assert os.path.exists(path), f"missing pipeline output: {path}"


def test_data_shapes(train_data, test_data):
    assert train_data["X"].shape[1] == N_FEATURES
    assert test_data["X"].shape[1] == N_FEATURES
    assert len(train_data["X"]) == len(train_data["y"])
    assert len(test_data["X"]) == len(test_data["y"])


def test_split_ratio(params, train_data, test_data):
    n_train, n_test = len(train_data["y"]), len(test_data["y"])
    ratio = n_test / (n_train + n_test)
    assert abs(ratio - params["prepare"]["test_size"]) < 0.01


def test_pixels_normalised(train_data, test_data):
    for X in (train_data["X"], test_data["X"]):
        assert X.min() >= 0.0 and X.max() <= 1.0
        assert not np.isnan(X).any()


def test_all_classes_present(train_data, test_data):
    assert set(np.unique(train_data["y"])) == set(range(N_CLASSES))
    assert set(np.unique(test_data["y"])) == set(range(N_CLASSES))


# ---------- model stage ----------
def test_model_matches_params(model, params):
    t = params["train"]
    assert model.hidden_layer_sizes == (t["hidden_units"],)
    assert model.activation == t["activation"]
    assert model.learning_rate_init == t["learning_rate"]
    assert model.batch_size == t["batch_size"]


def test_prediction_shape_and_range(model, test_data):
    pred = model.predict(test_data["X"])
    assert pred.shape == test_data["y"].shape
    assert set(np.unique(pred)).issubset(set(range(N_CLASSES)))


def test_predict_proba_sums_to_one(model, test_data):
    proba = model.predict_proba(test_data["X"][:20])
    assert proba.shape == (20, N_CLASSES)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_single_image_prediction(model, test_data):
    img = test_data["X"][0].reshape(1, -1)
    assert model.predict(img)[0] in range(N_CLASSES)


# ---------- evaluation stage ----------
def test_metrics_file(test_data, model):
    with open(METRICS_FILE) as f:
        m = json.load(f)
    for key in ("accuracy", "f1_macro", "precision_macro", "recall_macro"):
        assert key in m and 0.0 <= m[key] <= 1.0
    # metrics.json must agree with the model actually on disk
    acc = float((model.predict(test_data["X"]) == test_data["y"]).mean())
    assert abs(acc - m["accuracy"]) < 1e-3


def test_accuracy_quality_gate():
    with open(METRICS_FILE) as f:
        m = json.load(f)
    assert m["accuracy"] >= MIN_ACCURACY, (
        f"accuracy {m['accuracy']} below gate {MIN_ACCURACY}"
    )


# ---------- web application (Flask) ----------
@pytest.fixture(scope="module")
def client():
    import sys
    sys.path.insert(0, "app")
    import app as webapp
    return webapp.app.test_client()


def _png_b64(arr8, invert=False):
    import base64, io
    from PIL import Image
    img = Image.fromarray((arr8 * 255).astype("uint8")).resize((280, 280), Image.NEAREST)
    if invert:
        img = Image.fromarray(255 - np.asarray(img))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_app_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json["status"] == "ok"


def test_app_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Digit Classifier" in r.data


def test_app_rejects_bad_image(client):
    assert client.post("/predict", json={"image": "not-an-image"}).status_code == 400


def test_app_predicts_real_images(client, test_data):
    """Upscaled real test images (half inverted) sent through the HTTP API."""
    n, ok = 100, 0
    for i in range(n):
        r = client.post("/predict", json={"image": _png_b64(test_data["X"][i].reshape(8, 8), invert=i % 2 == 1)})
        assert r.status_code == 200
        body = r.json
        assert 0.0 <= body["confidence"] <= 1.0 and len(body["top3"]) == 3
        ok += body["prediction"] == int(test_data["y"][i])
    assert ok / n >= 0.85, f"API accuracy {ok}/{n}"
