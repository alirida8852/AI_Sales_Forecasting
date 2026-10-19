"""FastAPI endpoints for the first sales analytics demo."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sales_forecasting.data.cleaning import normalize_product_names
from sales_forecasting.data.database import DEFAULT_DATABASE_PATH
from sales_forecasting.data.intelligence import route_question
from sales_forecasting.data.llm import refine_answer
from sales_forecasting.ml.win_prediction import CATEGORICAL_FEATURES, NUMERIC_FEATURES


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "win_probability.joblib"
INTELLIGENCE_DATABASE_PATH = DEFAULT_DATABASE_PATH
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class IntelligenceQuestionRequest(BaseModel):
    question: str


class IntelligenceQuestionResponse(BaseModel):
    intent: str
    status: str
    answer: str
    data: dict[str, object] | None
    sources: list[dict[str, object]]
    next_action: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    """Report whether the API is available."""
    return {"status": "ok"}


@app.post("/api/v1/intelligence/ask", response_model=IntelligenceQuestionResponse)
def ask_intelligence(request: IntelligenceQuestionRequest) -> IntelligenceQuestionResponse:
    """Answer a grounded analytics or company-knowledge question."""
    if not INTELLIGENCE_DATABASE_PATH.exists():
        raise HTTPException(status_code=503, detail="Intelligence database is unavailable")
    answer = refine_answer(
        request.question,
        route_question(request.question, INTELLIGENCE_DATABASE_PATH),
    )
    return IntelligenceQuestionResponse(**answer.as_dict())


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


def load_enriched_pipeline() -> pd.DataFrame:
    """Load pipeline rows with the dimensions required by the model and UI."""
    raw_path = DATA_PATH.parent
    pipeline = normalize_product_names(
        pd.read_csv(DATA_PATH, parse_dates=["engage_date", "close_date"])
    )
    accounts = pd.read_csv(raw_path / "accounts.csv").rename(
        columns={"revenue": "account_revenue", "employees": "account_employees"}
    )
    products = pd.read_csv(raw_path / "products.csv").rename(columns={"sales_price": "product_price"})
    teams = pd.read_csv(raw_path / "sales_teams.csv")
    data = pipeline.merge(accounts, on="account", how="left")
    data = data.merge(products, on="product", how="left")
    return data.merge(teams, on="sales_agent", how="left")


@app.get("/api/v1/pipeline/opportunities")
def pipeline_opportunities(
    stage: str | None = Query(default=None),
    search: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, int | list[dict[str, object]]]:
    """Return filterable opportunity rows with win probabilities when a model exists."""
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail="CRM pipeline dataset is unavailable")

    data = load_enriched_pipeline()
    if stage and stage != "All stages":
        data = data[data["deal_stage"] == stage]
    if search:
        search_text = search.lower()
        searchable = data[["opportunity_id", "account", "product", "sales_agent"]].fillna("").astype(str)
        data = data[searchable.apply(lambda column: column.str.lower().str.contains(search_text)).any(axis=1)]

    if MODEL_PATH.exists():
        features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        model_data = data.copy()
        model_data["engage_year"] = model_data["engage_date"].dt.year
        model_data["engage_month"] = model_data["engage_date"].dt.month
        data = data.copy()
        data["win_probability"] = joblib.load(MODEL_PATH).predict_proba(model_data[features])[:, 1]
    else:
        data = data.copy()
        data["win_probability"] = None

    data["close_value"] = pd.to_numeric(data["close_value"], errors="coerce").fillna(0)
    data = data.sort_values(["win_probability", "close_value"], ascending=False, na_position="last")
    selected = data.iloc[offset : offset + limit]
    rows = selected[
        ["opportunity_id", "account", "product", "sales_agent", "deal_stage", "close_value", "win_probability", "engage_date"]
    ].copy()
    rows["engage_date"] = rows["engage_date"].dt.strftime("%Y-%m-%d")
    return {"total": int(len(data)), "items": rows.where(pd.notna(rows), None).to_dict(orient="records")}


@app.get("/api/v1/forecast/revenue")
def revenue_forecast() -> dict[str, object]:
    """Return recent won revenue and a three-month weighted pipeline estimate."""
    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Dataset or model is unavailable")

    data = load_enriched_pipeline()
    data["close_value"] = pd.to_numeric(data["close_value"], errors="coerce").fillna(0)
    closed = data[(data["deal_stage"] == "Won") & data["close_date"].notna()].copy()
    closed["month"] = closed["close_date"].dt.to_period("M").astype(str)
    monthly = closed.groupby("month", as_index=False)["close_value"].sum()
    monthly = monthly.sort_values("month").tail(6)

    open_deals = data[data["deal_stage"].isin(["Prospecting", "Engaging"])].copy()
    open_deals["engage_year"] = open_deals["engage_date"].dt.year
    open_deals["engage_month"] = open_deals["engage_date"].dt.month
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    probabilities = joblib.load(MODEL_PATH).predict_proba(open_deals[features])[:, 1]
    weighted_pipeline = float((open_deals["close_value"] * probabilities).sum())
    recent_average = float(monthly["close_value"].tail(3).mean()) if not monthly.empty else 0
    monthly_forecast = recent_average + (weighted_pipeline / 3)
    latest_month = pd.Timestamp(data["engage_date"].max()).to_period("M").to_timestamp()
    forecast = [
        {
            "month": (latest_month + pd.DateOffset(months=index)).to_period("M").strftime("%b %Y"),
            "actual": None,
            "forecast": round(monthly_forecast, 2),
        }
        for index in range(1, 4)
    ]
    history = [
        {"month": pd.Period(row["month"]).strftime("%b %Y"), "actual": round(float(row["close_value"]), 2), "forecast": None}
        for _, row in monthly.iterrows()
    ]
    return {
        "history": history,
        "forecast": forecast,
        "weighted_open_pipeline": round(weighted_pipeline, 2),
        "expected_monthly_revenue": round(monthly_forecast, 2),
        "method": "Recent won-revenue baseline plus probability-weighted open pipeline",
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