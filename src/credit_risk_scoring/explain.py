from __future__ import annotations

from pathlib import Path

import joblib
import shap
import numpy as np

MODEL_PATH = Path("models") / "model.joblib"


def explain(features: list[float]) -> list[float]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    explainer = shap.Explainer(model)
    values = explainer(np.array([features], dtype=float))
    return values.values[0].tolist()
