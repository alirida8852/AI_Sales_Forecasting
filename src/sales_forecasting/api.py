"""FastAPI endpoints for the first sales analytics demo."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sales_forecasting.ml.win_prediction import CATEGORICAL_FEATURES, NUMERIC_FEATURES


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "win_probability.joblib"
DATA_PATH = (
    ROOT
    / "src"
    / "sales_forecasting"
    / "data"
    / "raw"
    / "CRM+Sales+Opportunities"
    / "sales_pipeline.csv"
)

app = FastAPI(title="Sales Forecasting API", version="0.1.0")


class WinProbabilityRequest(BaseModel):
    engage_year: int
    engage_month: int
    account_revenue: float
    account_employees: int
    product_price: float
    sales_agent: str
    product: str
    account: str
    sector: str
    manager: str
    regional_office: str


@app.get("/health")
def health() -> dict[str, str]:
    """Report whether the API is available."""
    return {"status": "ok"}


@app.get("/api/v1/pipeline/summary")
def pipeline_summary() -> dict[str, float | int]:
    """Return high-level counts and values for closed and open pipeline deals."""
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail="CRM pipeline dataset is unavailable")

    data = pd.read_csv(DATA_PATH)
    data["close_value"] = pd.to_numeric(data["close_value"], errors="coerce").fillna(0)
    return {
        "total_opportunities": int(len(data)),
        "won_opportunities": int((data["deal_stage"] == "Won").sum()),
        "lost_opportunities": int((data["deal_stage"] == "Lost").sum()),
        "open_opportunities": int(data["deal_stage"].isin(["Engaging", "Prospecting"]).sum()),
        "closed_revenue": float(data.loc[data["deal_stage"] == "Won", "close_value"].sum()),
    }


@app.post("/api/v1/predictions/win-probability")
def predict_win_probability(request: WinProbabilityRequest) -> dict[str, float]:
    """Predict the probability that a supplied opportunity will be won."""
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model is unavailable; run train_model.py first")

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    values = pd.DataFrame([request.model_dump()])[features]
    probability = float(joblib.load(MODEL_PATH).predict_proba(values)[0, 1])
    return {"win_probability": probability}