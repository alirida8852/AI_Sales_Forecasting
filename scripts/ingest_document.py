"""Import one approved company knowledge document into SQLite."""

from __future__ import annotations

import argparse
from pathlib import Path

from sales_forecasting.data.knowledge import ingest_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to a Markdown or text document")
    parser.add_argument("--type", default="company-knowledge", dest="document_type")
    parser.add_argument("--scope", default="company", dest="access_scope")
    parser.add_argument("--id", default=None, dest="document_id")
    args = parser.parse_args()
    chunk_count = ingest_file(
        args.path,
        document_type=args.document_type,
        access_scope=args.access_scope,
        document_id=args.document_id,
    )
    print(f"Imported {args.path.name} as {args.document_id or args.path.stem} ({chunk_count} chunks).")


if __name__ == "__main__":
    main()