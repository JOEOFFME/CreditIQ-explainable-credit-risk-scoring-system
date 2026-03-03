from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

from credit_risk_scoring.explain import explain

app = FastAPI(title="Explainable Credit Risk Scoring System")

MODEL_PATH = Path("models") / "model.joblib"


class ScoreRequest(BaseModel):
    features: List[float]


class ScoreResponse(BaseModel):
    score: float


class ExplainResponse(BaseModel):
    attributions: List[float]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not available")

    model = joblib.load(MODEL_PATH)
    features = np.array([request.features], dtype=float)
    prediction = float(model.predict_proba(features)[0][1])
    return ScoreResponse(score=prediction)


@app.post("/explain", response_model=ExplainResponse)
def explain_score(request: ScoreRequest) -> ExplainResponse:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not available")

    attributions = explain(request.features)
    return ExplainResponse(attributions=attributions)
