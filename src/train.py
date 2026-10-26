"""Stage 2: train an MLP image classifier with hyperparameters from params.yaml."""
import os
import pickle
import warnings

import numpy as np
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

p = yaml.safe_load(open("params.yaml"))["train"]
train = np.load("data/prepared/train.npz")

model = MLPClassifier(
    hidden_layer_sizes=(p["hidden_units"],),
    activation=p["activation"],
    learning_rate_init=p["learning_rate"],
    alpha=p["alpha"],
    batch_size=p["batch_size"],
    max_iter=p["epochs"],
    random_state=p["seed"],
    solver="adam",
)
model.fit(train["X"], train["y"])

os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"trained for {model.n_iter_} epochs, final loss {model.loss_:.4f}")
