"""Train and evaluate the opportunity win-probability model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

TARGET = "deal_outcome"
NUMERIC_FEATURES = [
    "engage_year",
    "engage_month",
    "account_revenue",
    "account_employees",
    "product_price",
]
CATEGORICAL_FEATURES = [
    "sales_agent",
    "product",
    "account",
    "sector",
    "manager",
    "regional_office",
]


def build_pipeline() -> Pipeline:
    """Build a leakage-resistant preprocessing and classification pipeline."""
    transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    classifier = XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=2,
    )
    return Pipeline([("features", transformer), ("model", classifier)])


def train_and_evaluate(data: pd.DataFrame) -> tuple[Pipeline, dict[str, float]]:
    """Train on closed opportunities and evaluate on the latest 20 percent."""
    ordered = data.sort_values("engage_date").reset_index(drop=True)
    split_index = int(len(ordered) * 0.8)
    train_data = ordered.iloc[:split_index]
    test_data = ordered.iloc[split_index:]
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    target = lambda frame: (frame[TARGET] == "Won").astype(int)

    pipeline = build_pipeline()
    pipeline.fit(train_data[features], target(train_data))
    probabilities = pipeline.predict_proba(test_data[features])[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(target(test_data), probabilities)),
        "brier_score": float(brier_score_loss(target(test_data), probabilities)),
        "train_rows": float(len(train_data)),
        "test_rows": float(len(test_data)),
    }
    return pipeline, metrics


def save_model(pipeline: Pipeline, output_path: Path) -> Path:
    """Persist the trained pipeline for the future API service."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    return output_path