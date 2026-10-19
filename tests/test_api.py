from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException

from sales_forecasting import api
from sales_forecasting.data.database import initialize_database


def test_health_reports_ok() -> None:
    assert api.health() == {"status": "ok"}


def test_pipeline_summary_counts_csv_rows(tmp_path, monkeypatch) -> None:
    pipeline_path = tmp_path / "sales_pipeline.csv"
    pd.DataFrame(
        [
            {"deal_stage": "Won", "close_value": "1000"},
            {"deal_stage": "Lost", "close_value": "250"},
            {"deal_stage": "Prospecting", "close_value": None},
            {"deal_stage": "Engaging", "close_value": "not available"},
        ]
    ).to_csv(pipeline_path, index=False)
    monkeypatch.setattr(api, "DATA_PATH", pipeline_path)

    assert api.pipeline_summary() == {
        "total_opportunities": 4,
        "won_opportunities": 1,
        "lost_opportunities": 1,
        "open_opportunities": 2,
        "closed_revenue": 1000.0,
    }


def test_pipeline_summary_reports_missing_dataset(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api, "DATA_PATH", Path(tmp_path / "missing.csv"))

    with pytest.raises(HTTPException) as error:
        api.pipeline_summary()

    assert error.value.status_code == 503


def test_pipeline_opportunities_filters_and_paginates(tmp_path, monkeypatch) -> None:
    data = pd.DataFrame(
        [
            {
                "opportunity_id": "OP-1",
                "account": "Acme",
                "product": "GTXPro",
                "sales_agent": "Alex",
                "deal_stage": "Prospecting",
                "close_value": 1000,
                "engage_date": pd.Timestamp("2026-01-01"),
            },
            {
                "opportunity_id": "OP-2",
                "account": "Beta",
                "product": "GTX Pro",
                "sales_agent": "Blair",
                "deal_stage": "Won",
                "close_value": 2000,
                "engage_date": pd.Timestamp("2026-01-02"),
            },
        ]
    )
    data_path = tmp_path / "available.csv"
    data_path.touch()
    monkeypatch.setattr(api, "DATA_PATH", data_path)
    monkeypatch.setattr(api, "load_enriched_pipeline", lambda: data)
    monkeypatch.setattr(api, "MODEL_PATH", Path("missing-model.joblib"))

    result = api.pipeline_opportunities(stage="Won", search="beta", limit=1, offset=0)

    assert result["total"] == 1
    assert result["items"][0]["opportunity_id"] == "OP-2"
    assert result["items"][0]["win_probability"] is None


def test_prediction_request_requires_all_model_fields() -> None:
    with pytest.raises(ValueError):
        api.WinProbabilityRequest(
            engage_year=2026,
            engage_month=10,
            account_revenue=1000,
            account_employees=10,
            product_price=100,
            sales_agent="Alex",
            product="GTX Pro",
            account="Acme",
            sector="Technology",
            manager="Morgan",
        )


def test_intelligence_endpoint_returns_structured_answer(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "sales.sqlite3"
    connection = __import__("sqlite3").connect(database_path)
    initialize_database(connection)
    connection.close()
    monkeypatch.setattr(api, "INTELLIGENCE_DATABASE_PATH", database_path)

    response = api.ask_intelligence(api.IntelligenceQuestionRequest(question="What is the weather?"))

    assert response.status == "unsupported_question"
    assert response.sources == []


def test_load_enriched_pipeline_joins_account_product_and_team_data(tmp_path, monkeypatch) -> None:
    pipeline_path = tmp_path / "sales_pipeline.csv"
    pd.DataFrame(
        [
            {
                "opportunity_id": "OP-1",
                "account": "Acme",
                "product": "GTX Pro",
                "sales_agent": "Alex",
                "deal_stage": "Won",
                "close_value": 1000,
                "engage_date": "2026-01-01",
                "close_date": "2026-02-01",
            }
        ]
    ).to_csv(pipeline_path, index=False)
    pd.DataFrame([{"account": "Acme", "revenue": 50000, "employees": 25}]).to_csv(
        tmp_path / "accounts.csv", index=False
    )
    pd.DataFrame([{"product": "GTX Pro", "sales_price": 1250}]).to_csv(
        tmp_path / "products.csv", index=False
    )
    pd.DataFrame(
        [{"sales_agent": "Alex", "manager": "Morgan", "regional_office": "East"}]
    ).to_csv(tmp_path / "sales_teams.csv", index=False)
    monkeypatch.setattr(api, "DATA_PATH", pipeline_path)

    enriched = api.load_enriched_pipeline()

    assert enriched.loc[0, "account_revenue"] == 50000
    assert enriched.loc[0, "account_employees"] == 25
    assert enriched.loc[0, "product_price"] == 1250
    assert enriched.loc[0, "manager"] == "Morgan"
