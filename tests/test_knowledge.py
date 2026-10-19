import pytest

from sales_forecasting.data.knowledge import ingest_document, ingest_file, search_documents


def test_ingest_and_search_document(tmp_path) -> None:
    database_path = tmp_path / "sales.sqlite3"

    chunk_count = ingest_document(
        document_id="playbook-1",
        title="Enterprise Sales Playbook",
        document_type="playbook",
        source="enterprise-sales.md",
        content="Enterprise opportunities require an executive sponsor and a discovery call before proposal.",
        database_path=database_path,
    )

    results = search_documents("executive sponsor", database_path=database_path)

    assert chunk_count == 1
    assert len(results) == 1
    assert results[0]["document_id"] == "playbook-1"
    assert results[0]["source"] == "enterprise-sales.md"


def test_ingest_replaces_existing_document_version(tmp_path) -> None:
    database_path = tmp_path / "sales.sqlite3"
    common = {
        "document_id": "policy-1",
        "title": "Discount Policy",
        "document_type": "policy",
        "source": "discounts.md",
        "database_path": database_path,
    }

    ingest_document(content="Discount approvals require manager review.", **common)
    ingest_document(content="Discount approvals require finance review.", **common)

    assert search_documents("manager review", database_path=database_path) == []
    assert len(search_documents("finance review", database_path=database_path)) == 1


def test_ingest_file_imports_approved_text_document(tmp_path) -> None:
    document_path = tmp_path / "enterprise_playbook.md"
    document_path.write_text("Enterprise deals require an executive sponsor.", encoding="utf-8")
    database_path = tmp_path / "sales.sqlite3"

    assert ingest_file(document_path, "playbook", database_path=database_path) == 1
    results = search_documents("executive sponsor", database_path=database_path)

    assert results[0]["source"] == "enterprise_playbook.md"


def test_ingest_file_rejects_unsupported_format(tmp_path) -> None:
    document_path = tmp_path / "pricing.pdf"
    document_path.write_bytes(b"not imported")

    with pytest.raises(ValueError, match="Only"):
        ingest_file(document_path, "policy", database_path=tmp_path / "sales.sqlite3")


def test_search_documents_respects_access_scope(tmp_path) -> None:
    database_path = tmp_path / "sales.sqlite3"
    ingest_document(
        document_id="finance-policy",
        title="Finance Policy",
        document_type="policy",
        source="finance.md",
        content="Finance-only discount approvals require director review.",
        access_scope="finance",
        database_path=database_path,
    )

    assert search_documents("discount approvals", database_path=database_path) == []
    assert len(search_documents("discount approvals", database_path=database_path, access_scope="finance")) == 1