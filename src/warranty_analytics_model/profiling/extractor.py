"""Read-only business-table extraction for Phase 3 diagnostics."""

from __future__ import annotations

import re
from collections.abc import Iterator
from time import monotonic
from typing import Any

import pandas as pd
from sqlalchemy import text

from ..database.config import DatabaseSettings
from ..database.connection import DatabaseConnection
from ..database.models import SchemaContract, TableSpec

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_identifier(identifier: str) -> str:
    """Quote a contract identifier after strict validation."""

    if not _IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
    return f"[{identifier}]"


def table_select_sql(table: TableSpec) -> str:
    """Build a reviewed, explicit-column SELECT for one approved table."""

    columns = ", ".join(quote_identifier(column.name) for column in table.columns)
    return f"SELECT {columns} FROM {quote_identifier(table.schema)}.{quote_identifier(table.table)}"


def table_count_sql(table: TableSpec) -> str:
    """Build the exact-count query for one approved table."""

    return (
        f"SELECT COUNT_BIG(*) AS [exact_row_count] FROM "
        f"{quote_identifier(table.schema)}.{quote_identifier(table.table)}"
    )


class LiveProfileExtractor:
    """Reuse one Phase 2 read-only connection for all approved table reads."""

    def __init__(
        self, settings: DatabaseSettings, contract: SchemaContract, chunk_size: int = 10_000
    ) -> None:
        self.settings = settings
        self.contract = contract
        self.chunk_size = chunk_size
        self.connection = DatabaseConnection(settings)

    def count_rows(self, db_connection: Any, table: TableSpec) -> int:
        """Read one exact row count; no excluded object can enter this method."""

        result = db_connection.execute(text(table_count_sql(table))).scalar_one()
        return int(result)

    def read_table(self, db_connection: Any, table: TableSpec) -> pd.DataFrame:
        """Read only contract columns in chunks and concatenate in memory for one run."""

        started = monotonic()
        result = db_connection.execution_options(stream_results=True).execute(
            text(table_select_sql(table))
        )
        column_names = [column.name for column in table.columns]
        batches: list[pd.DataFrame] = []
        while True:
            rows = result.fetchmany(self.chunk_size)
            if not rows:
                break
            batches.append(pd.DataFrame([tuple(row) for row in rows], columns=column_names))
        frame = (
            pd.concat(batches, ignore_index=True) if batches else pd.DataFrame(columns=column_names)
        )
        # The duration is intentionally retained as non-sensitive metadata for callers/logging.
        frame.attrs["profile_read_duration_seconds"] = round(monotonic() - started, 6)
        return frame

    def extract_all(self) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
        """Extract all 16 included tables and exact counts using one connection."""

        self.settings.validate_for_connection()
        frames: dict[str, pd.DataFrame] = {}
        exact_counts: dict[str, int] = {}
        try:
            with self.connection.connect() as db_connection:
                for table in self.contract.tables:
                    exact_counts[table.name] = self.count_rows(db_connection, table)
                    frames[table.name] = self.read_table(db_connection, table)
        finally:
            self.connection.dispose()
        return frames, exact_counts


def iter_table_chunks(
    extractor: LiveProfileExtractor,
    db_connection: Any,
    table: TableSpec,
) -> Iterator[pd.DataFrame]:
    """Yield explicit-column batches for callers that do not need concatenation."""

    result = db_connection.execution_options(stream_results=True).execute(
        text(table_select_sql(table))
    )
    column_names = [column.name for column in table.columns]
    while True:
        rows = result.fetchmany(extractor.chunk_size)
        if not rows:
            return
        yield pd.DataFrame([tuple(row) for row in rows], columns=column_names)
