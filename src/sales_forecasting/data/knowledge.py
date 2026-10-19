"""Company document storage and lightweight full-text retrieval."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from sales_forecasting.data.database import DEFAULT_DATABASE_PATH, connect, initialize_database

SUPPORTED_DOCUMENT_SUFFIXES = {".md", ".markdown", ".txt"}
MAX_DOCUMENT_SIZE = 2 * 1024 * 1024


def _chunks(content: str, chunk_size: int = 1200) -> list[str]:
    """Split text into bounded chunks while retaining all source content."""
    normalized = " ".join(content.split())
    return [normalized[index : index + chunk_size] for index in range(0, len(normalized), chunk_size)]


def ingest_document(
    document_id: str,
    title: str,
    document_type: str,
    source: str,
    content: str,
    access_scope: str = "company",
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> int:
    """Store a document and its searchable chunks, replacing an existing version."""
    chunks = _chunks(content)
    if not chunks:
        raise ValueError("Document content cannot be empty")

    created_at = datetime.now(UTC).isoformat()
    with connect(database_path) as connection:
        initialize_database(connection)
        old_chunks = connection.execute(
            "SELECT chunk_id FROM document_chunks WHERE document_id = ?", (document_id,)
        ).fetchall()
        connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        connection.execute(
            """
            INSERT INTO documents(document_id, title, document_type, source, access_scope, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (document_id, title, document_type, source, access_scope, created_at),
        )
        for (chunk_id,) in old_chunks:
            connection.execute("DELETE FROM document_chunks_fts WHERE chunk_id = ?", (chunk_id,))
        for index, chunk in enumerate(chunks):
            cursor = connection.execute(
                """
                INSERT INTO document_chunks(document_id, chunk_index, content)
                VALUES (?, ?, ?)
                """,
                (document_id, index, chunk),
            )
            chunk_id = cursor.lastrowid
            connection.execute(
                "INSERT INTO document_chunks_fts(content, document_id, chunk_id) VALUES (?, ?, ?)",
                (chunk, document_id, chunk_id),
            )
        connection.commit()
    return len(chunks)


def ingest_file(
    file_path: Path,
    document_type: str,
    access_scope: str = "company",
    document_id: str | None = None,
    database_path: Path = DEFAULT_DATABASE_PATH,
) -> int:
    """Validate and import one approved Markdown or text file."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Document does not exist: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_DOCUMENT_SUFFIXES:
        raise ValueError("Only .md, .markdown, and .txt documents are supported")
    if file_path.stat().st_size > MAX_DOCUMENT_SIZE:
        raise ValueError("Document exceeds the 2 MB ingestion limit")
    content = file_path.read_text(encoding="utf-8")
    return ingest_document(
        document_id=document_id or file_path.stem.lower().replace(" ", "-"),
        title=file_path.stem.replace("-", " ").replace("_", " ").title(),
        document_type=document_type,
        source=file_path.name,
        content=content,
        access_scope=access_scope,
        database_path=database_path,
    )


def search_documents(
    query: str,
    database_path: Path = DEFAULT_DATABASE_PATH,
    limit: int = 5,
    access_scope: str = "internal",
) -> list[dict[str, str | int]]:
    """Return matching chunks visible to the requested access scope."""
    if not query.strip():
        return []
    terms = [term for term in query.split() if term.strip()]
    safe_query = " ".join('"' + term.replace('"', '""') + '"' for term in terms)
    with connect(database_path) as connection:
        initialize_database(connection)
        rows = connection.execute(
            """
            SELECT documents.document_id, documents.title, documents.source,
                   document_chunks.chunk_index, document_chunks.content
            FROM document_chunks_fts
            JOIN document_chunks ON document_chunks.chunk_id = document_chunks_fts.chunk_id
            JOIN documents ON documents.document_id = document_chunks.document_id
                        WHERE document_chunks_fts MATCH ?
                            AND documents.access_scope IN ('company', ?)
            ORDER BY rank
            LIMIT ?
            """,
                        (safe_query, access_scope, limit),
        ).fetchall()
    return [
        {
            "document_id": row[0],
            "title": row[1],
            "source": row[2],
            "chunk_index": row[3],
            "content": row[4],
        }
        for row in rows
    ]