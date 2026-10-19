import sqlite3

import pandas as pd

from sales_forecasting.data.database import import_crm_data, initialize_database
from sales_forecasting.data.repository import SQLiteSalesRepository


def test_initialize_database_creates_collaboration_tables() -> None:
    connection = sqlite3.connect(":memory:")

    initialize_database(connection)

    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert {"accounts", "products", "users", "opportunities", "activities", "comments", "tasks"} <= tables


def test_import_crm_data_loads_normalized_snapshot(tmp_path) -> None:
    raw_path = tmp_path / "raw"
    raw_path.mkdir()
    pd.DataFrame(
        [
            {
                "opportunity_id": "OP-1",
                "account": "Acme",
                "product": "GTXPro",
                "sales_agent": "Alex",
                "deal_stage": "Prospecting",
                "close_value": None,
                "engage_date": "2026-01-01",
                "close_date": None,
            }
        ]
    ).to_csv(raw_path / "sales_pipeline.csv", index=False)
    pd.DataFrame([{"account": "Acme", "sector": "Technology", "revenue": 50000, "employees": 25}]).to_csv(
        raw_path / "accounts.csv", index=False
    )
    pd.DataFrame([{"product": "GTX Pro", "sales_price": 1250}]).to_csv(
        raw_path / "products.csv", index=False
    )
    pd.DataFrame([{"sales_agent": "Alex", "manager": "Morgan", "regional_office": "East"}]).to_csv(
        raw_path / "sales_teams.csv", index=False
    )

    database_path = tmp_path / "sales.sqlite3"
    assert import_crm_data(database_path, raw_path) == 1

    with sqlite3.connect(database_path) as connection:
        opportunity = connection.execute(
            "SELECT product_name, deal_stage FROM opportunities WHERE opportunity_id = 'OP-1'"
        ).fetchone()
        product = connection.execute(
            "SELECT price FROM products WHERE product_name = 'GTX Pro'"
        ).fetchone()

    assert opportunity == ("GTX Pro", "Prospecting")
    assert product == (1250.0,)


def test_sqlite_repository_returns_pipeline_summary(tmp_path) -> None:
    database_path = tmp_path / "sales.sqlite3"
    connection = sqlite3.connect(database_path)
    initialize_database(connection)
    connection.execute("INSERT INTO accounts(account_name) VALUES ('Acme')")
    connection.execute("INSERT INTO accounts(account_name) VALUES ('Beta')")
    connection.execute("INSERT INTO products(product_name, price) VALUES ('GTX Pro', 1250)")
    connection.execute(
        "INSERT INTO users(sales_agent, manager, regional_office) VALUES ('Alex', 'Morgan', 'East')"
    )
    connection.executemany(
        """
        INSERT INTO opportunities(
            opportunity_id, account_name, product_name, sales_agent,
            deal_stage, close_value
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            ("OP-1", "Acme", "GTX Pro", "Alex", "Won", 1000),
            ("OP-2", None, "GTX Pro", "Alex", "Prospecting", None),
            ("OP-3", "Beta", "GTX Pro", "Alex", "Lost", 250),
        ],
    )
    connection.commit()
    connection.close()

    repository = SQLiteSalesRepository(database_path)

    assert repository.pipeline_summary() == {
        "total_opportunities": 3,
        "won_opportunities": 1,
        "lost_opportunities": 1,
        "open_opportunities": 1,
        "closed_revenue": 1000.0,
    }
    assert repository.opportunity_count() == 3