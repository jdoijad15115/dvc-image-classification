"""Stage 1: load the 8x8 handwritten-digit image dataset, normalise and split."""
import os

import numpy as np
import yaml
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["prepare"]

digits = load_digits()                       # 1797 grayscale images, 8x8, 10 classes
X = digits.images.reshape(len(digits.images), -1) / 16.0   # pixel range 0..16 -> 0..1
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["seed"], stratify=y
)

os.makedirs("data/prepared", exist_ok=True)
np.savez("data/prepared/train.npz", X=X_train, y=y_train)
np.savez("data/prepared/test.npz", X=X_test, y=y_test)
print(f"train: {X_train.shape}  test: {X_test.shape}")
