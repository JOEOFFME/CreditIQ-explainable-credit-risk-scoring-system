from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

MODEL_PATH = Path("models") / "model.joblib"


def train_dummy_model() -> None:
    # Placeholder training using synthetic data.
    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(200, 5))
    y_train = (x_train[:, 0] + x_train[:, 1] > 0).astype(int)

    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)


if __name__ == "__main__":
    train_dummy_model()
