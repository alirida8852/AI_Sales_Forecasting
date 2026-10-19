"""Load CRM tables and train the first win-probability model."""

import json
from pathlib import Path

import pandas as pd

from sales_forecasting.data.cleaning import normalize_product_names
from sales_forecasting.ml.win_prediction import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    save_model,
    train_and_evaluate,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "src" / "sales_forecasting" / "data" / "raw" / "CRM+Sales+Opportunities"
PIPELINE_PATH = RAW_PATH / "sales_pipeline.csv"
ACCOUNTS_PATH = RAW_PATH / "accounts.csv"
PRODUCTS_PATH = RAW_PATH / "products.csv"
TEAMS_PATH = RAW_PATH / "sales_teams.csv"
MODEL_PATH = ROOT / "models" / "win_probability.joblib"
METRICS_PATH = ROOT / "models" / "win_probability_metrics.json"


def load_sales_pipeline() -> pd.DataFrame:
    """Load the pipeline and join the account, product, and team dimensions."""
    pipeline = normalize_product_names(
        pd.read_csv(PIPELINE_PATH, parse_dates=["engage_date", "close_date"])
    )
    accounts = pd.read_csv(ACCOUNTS_PATH).rename(
        columns={"revenue": "account_revenue", "employees": "account_employees"}
    )
    products = pd.read_csv(PRODUCTS_PATH).rename(columns={"sales_price": "product_price"})
    teams = pd.read_csv(TEAMS_PATH)

    data = pipeline.merge(accounts, on="account", how="left")
    data = data.merge(products, on="product", how="left")
    data = data.merge(teams, on="sales_agent", how="left")
    data["engage_year"] = data["engage_date"].dt.year
    data["engage_month"] = data["engage_date"].dt.month
    data = data.rename(columns={"deal_stage": "deal_outcome"})
    return data[data["deal_outcome"].isin(["Won", "Lost"])].copy()


def main() -> None:
    required_paths = [PIPELINE_PATH, ACCOUNTS_PATH, PRODUCTS_PATH, TEAMS_PATH]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("Missing CRM files: " + ", ".join(str(path) for path in missing_paths))

    data = load_sales_pipeline()
    pipeline, metrics = train_and_evaluate(data)
    save_model(pipeline, MODEL_PATH)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Dataset: {PIPELINE_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()