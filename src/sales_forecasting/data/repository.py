"""Read-only queries over the collaborative sales database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sales_forecasting.data.database import DEFAULT_DATABASE_PATH, connect


class SQLiteSalesRepository:
    """Expose stable read queries without coupling callers to SQL."""

    def __init__(self, database_path: Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = database_path

    def pipeline_summary(self) -> dict[str, float | int]:
        """Return the same summary fields as the existing CSV API."""
        with connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(1) AS total_opportunities,
                    SUM(CASE WHEN deal_stage = 'Won' THEN 1 ELSE 0 END) AS won_opportunities,
                    SUM(CASE WHEN deal_stage = 'Lost' THEN 1 ELSE 0 END) AS lost_opportunities,
                    SUM(CASE WHEN deal_stage IN ('Engaging', 'Prospecting') THEN 1 ELSE 0 END)
                        AS open_opportunities,
                    COALESCE(
                        SUM(CASE WHEN deal_stage = 'Won' THEN COALESCE(close_value, 0) ELSE 0 END),
                        0
                    ) AS closed_revenue
                FROM opportunities
                """
            ).fetchone()
        if row is None:
            raise RuntimeError("SQLite summary query returned no row")
        return {
            "total_opportunities": int(row[0]),
            "won_opportunities": int(row[1]),
            "lost_opportunities": int(row[2]),
            "open_opportunities": int(row[3]),
            "closed_revenue": float(row[4]),
        }

    def opportunity_count(self) -> int:
        """Return the number of imported opportunities."""
        with connect(self.database_path) as connection:
            row = connection.execute("SELECT COUNT(1) FROM opportunities").fetchone()
        if row is None:
            raise RuntimeError("SQLite opportunity count query returned no row")
        return int(row[0])

    def search_opportunities(
        self,
        search: str | None = None,
        stage: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, object]]:
        """Find opportunities for grounded conversational lookup."""
        clauses: list[str] = []
        parameters: list[object] = []
        if search:
            pattern = f"%{search.lower()}%"
            clauses.append(
                "lower(opportunity_id) LIKE ? OR lower(COALESCE(account_name, '')) LIKE ? "
                "OR lower(product_name) LIKE ? OR lower(sales_agent) LIKE ?"
            )
            parameters.extend([pattern, pattern, pattern, pattern])
        if stage:
            clauses.append("lower(deal_stage) = ?")
            parameters.append(stage.lower())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)
        with connect(self.database_path) as connection:
            rows = connection.execute(
                f"""
                SELECT opportunity_id, account_name, product_name, sales_agent,
                       deal_stage, close_value, engage_date
                FROM opportunities
                {where}
                ORDER BY COALESCE(close_value, 0) DESC, opportunity_id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [
            {
                "opportunity_id": row[0],
                "account": row[1],
                "product": row[2],
                "sales_agent": row[3],
                "deal_stage": row[4],
                "close_value": float(row[5] or 0),
                "engage_date": row[6],
            }
            for row in rows
        ]

    def stage_counts(self) -> dict[str, int]:
        """Return opportunity counts grouped by current stage."""
        with connect(self.database_path) as connection:
            rows = connection.execute(
                "SELECT deal_stage, COUNT(1) FROM opportunities GROUP BY deal_stage ORDER BY deal_stage"
            ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}