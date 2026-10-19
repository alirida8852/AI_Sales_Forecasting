"""SQLite persistence for collaborative sales data."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from sales_forecasting.data.cleaning import normalize_product_names


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATABASE_PATH = ROOT / "data" / "sales_forecasting.sqlite3"
DEFAULT_RAW_PATH = Path(__file__).resolve().parent / "raw" / "CRM+Sales+Opportunities"

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS accounts (
    account_name TEXT PRIMARY KEY,
    sector TEXT,
    revenue REAL,
    employees INTEGER
);

CREATE TABLE IF NOT EXISTS products (
    product_name TEXT PRIMARY KEY,
    price REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    sales_agent TEXT PRIMARY KEY,
    manager TEXT,
    regional_office TEXT
);

CREATE TABLE IF NOT EXISTS opportunities (
    opportunity_id TEXT PRIMARY KEY,
    account_name TEXT,
    product_name TEXT NOT NULL,
    sales_agent TEXT NOT NULL,
    deal_stage TEXT NOT NULL,
    close_value REAL,
    engage_date TEXT,
    close_date TEXT,
    FOREIGN KEY (account_name) REFERENCES accounts(account_name),
    FOREIGN KEY (product_name) REFERENCES products(product_name),
    FOREIGN KEY (sales_agent) REFERENCES users(sales_agent)
);

CREATE TABLE IF NOT EXISTS activities (
    activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    opportunity_id TEXT NOT NULL,
    activity_type TEXT NOT NULL,
    notes TEXT,
    occurred_at TEXT NOT NULL,
    FOREIGN KEY (opportunity_id) REFERENCES opportunities(opportunity_id)
);

CREATE TABLE IF NOT EXISTS comments (
    comment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    opportunity_id TEXT NOT NULL,
    author TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (opportunity_id) REFERENCES opportunities(opportunity_id)
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    opportunity_id TEXT NOT NULL,
    assignee TEXT NOT NULL,
    title TEXT NOT NULL,
    due_date TEXT,
    status TEXT NOT NULL DEFAULT 'Open',
    FOREIGN KEY (opportunity_id) REFERENCES opportunities(opportunity_id)
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    document_type TEXT NOT NULL,
    source TEXT NOT NULL,
    access_scope TEXT NOT NULL DEFAULT 'company',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE,
    UNIQUE (document_id, chunk_index)
);

CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
    content,
    document_id UNINDEXED,
    chunk_id UNINDEXED
);

CREATE INDEX IF NOT EXISTS idx_opportunities_stage ON opportunities(deal_stage);
CREATE INDEX IF NOT EXISTS idx_opportunities_agent ON opportunities(sales_agent);
CREATE INDEX IF NOT EXISTS idx_activities_opportunity ON activities(opportunity_id);
CREATE INDEX IF NOT EXISTS idx_comments_opportunity ON comments(opportunity_id);
CREATE INDEX IF NOT EXISTS idx_tasks_opportunity ON tasks(opportunity_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_document ON document_chunks(document_id);
"""


def connect(database_path: Path = DEFAULT_DATABASE_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with foreign-key checks enabled."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    """Create the persistence schema if it does not already exist."""
    connection.executescript(SCHEMA)
    connection.commit()


def import_crm_data(
    database_path: Path = DEFAULT_DATABASE_PATH,
    raw_path: Path = DEFAULT_RAW_PATH,
) -> int:
    """Import the current CRM CSV snapshot into SQLite and return opportunity count."""
    pipeline = normalize_product_names(pd.read_csv(raw_path / "sales_pipeline.csv"))
    accounts = pd.read_csv(raw_path / "accounts.csv")
    products = pd.read_csv(raw_path / "products.csv").rename(columns={"sales_price": "price"})
    users = pd.read_csv(raw_path / "sales_teams.csv")

    with connect(database_path) as connection:
        initialize_database(connection)
        connection.executescript(
            """
            DELETE FROM tasks;
            DELETE FROM comments;
            DELETE FROM activities;
            DELETE FROM opportunities;
            DELETE FROM users;
            DELETE FROM products;
            DELETE FROM accounts;
            """
        )
        accounts.rename(columns={"revenue": "account_revenue", "employees": "account_employees"})[
            ["account", "sector", "account_revenue", "account_employees"]
        ].to_sql("accounts_staging", connection, if_exists="replace", index=False)
        connection.execute(
            """
            INSERT INTO accounts(account_name, sector, revenue, employees)
            SELECT account, sector, account_revenue, account_employees FROM accounts_staging
            """
        )
        products[["product", "price"]].to_sql("products_staging", connection, if_exists="replace", index=False)
        connection.execute(
            """
            INSERT INTO products(product_name, price)
            SELECT product, price FROM products_staging
            """
        )
        users[["sales_agent", "manager", "regional_office"]].to_sql(
            "users_staging", connection, if_exists="replace", index=False
        )
        connection.execute(
            """
            INSERT INTO users(sales_agent, manager, regional_office)
            SELECT sales_agent, manager, regional_office FROM users_staging
            """
        )
        pipeline[
            [
                "opportunity_id",
                "account",
                "product",
                "sales_agent",
                "deal_stage",
                "close_value",
                "engage_date",
                "close_date",
            ]
        ].to_sql("opportunities_staging", connection, if_exists="replace", index=False)
        connection.execute(
            """
            INSERT INTO opportunities(
                opportunity_id, account_name, product_name, sales_agent,
                deal_stage, close_value, engage_date, close_date
            )
            SELECT opportunity_id, account, product, sales_agent,
                   deal_stage, close_value, engage_date, close_date
            FROM opportunities_staging
            """
        )
        connection.executescript(
            """
            DROP TABLE accounts_staging;
            DROP TABLE products_staging;
            DROP TABLE users_staging;
            DROP TABLE opportunities_staging;
            """
        )
        connection.commit()

    return len(pipeline)