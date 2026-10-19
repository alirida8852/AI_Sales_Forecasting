import sqlite3

from sales_forecasting.data.database import initialize_database
from sales_forecasting.data.intelligence import route_question
from sales_forecasting.data.knowledge import ingest_document


def test_router_answers_open_pipeline_question(tmp_path) -> None:
    database_path = tmp_path / "sales.sqlite3"
    connection = sqlite3.connect(database_path)
    initialize_database(connection)
    connection.execute("INSERT INTO accounts(account_name) VALUES ('Acme')")
    connection.execute("INSERT INTO products(product_name, price) VALUES ('GTX Pro', 1250)")
    connection.execute(
        "INSERT INTO users(sales_agent, manager, regional_office) VALUES ('Alex', 'Morgan', 'East')"
    )
    connection.execute(
        """
        INSERT INTO opportunities(
            opportunity_id, account_name, product_name, sales_agent, deal_stage
        ) VALUES ('OP-1', 'Acme', 'GTX Pro', 'Alex', 'Prospecting')
        """
    )
    connection.commit()
    connection.close()

    result = route_question("How many open opportunities do we have?", database_path)

    assert result.intent == "analytics"
    assert result.status == "answered"
    assert result.data == {"open_opportunities": 1}
    assert result.sources == [{"type": "database", "source": "opportunities"}]
    assert result.next_action is not None

    lookup = route_question("Show opportunities for Acme", database_path)
    assert lookup.data is not None
    assert lookup.data["results"][0]["opportunity_id"] == "OP-1"

    stages = route_question("Show deals by stage", database_path)
    assert stages.data == {"stages": {"Prospecting": 1}}


def test_router_retrieves_approved_company_document(tmp_path) -> None:
    database_path = tmp_path / "sales.sqlite3"
    ingest_document(
        document_id="playbook-1",
        title="Enterprise Playbook",
        document_type="playbook",
        source="enterprise.md",
        content="Enterprise opportunities require an executive sponsor before proposal.",
        database_path=database_path,
    )

    result = route_question("What does the playbook say about executive sponsor?", database_path)

    assert result.intent == "knowledge"
    assert result.status == "retrieved"
    assert result.sources[0]["document_id"] == "playbook-1"
    assert "executive sponsor" in result.sources[0]["excerpt"]
    assert result.next_action is not None


def test_router_rejects_unsupported_question(tmp_path) -> None:
    result = route_question("What is the weather today?", tmp_path / "sales.sqlite3")

    assert result.status == "unsupported_question"
    assert result.sources == []