from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    name: str = "Explainable Credit Risk Scoring System"
    model_path: Path = Path("models") / "model.joblib"


DEFAULT_CONFIG = AppConfig()
