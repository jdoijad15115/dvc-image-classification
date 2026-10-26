"""Stage 3: evaluate the trained model on the held-out test images."""
import json
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

test = np.load("data/prepared/test.npz")
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

pred = model.predict(test["X"])
metrics = {
    "accuracy": round(accuracy_score(test["y"], pred), 4),
    "f1_macro": round(f1_score(test["y"], pred, average="macro"), 4),
    "precision_macro": round(precision_score(test["y"], pred, average="macro"), 4),
    "recall_macro": round(recall_score(test["y"], pred, average="macro"), 4),
    "train_loss": round(float(model.loss_), 4),
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))
