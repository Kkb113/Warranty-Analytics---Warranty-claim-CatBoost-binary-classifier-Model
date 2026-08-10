"""Read-only SQL Server extraction for the Phase 5 mart bundle."""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd
from sqlalchemy import text

from ..database.config import DatabaseSettings
from ..database.connection import DatabaseConnection
from .extraction_plan import ExtractionPlan, explicit_count_sql, explicit_select_sql, plan_columns
from .models import FeatureMartError


class ExtractionResult(TypedDict):
    frames: dict[str, pd.DataFrame]
    source_row_counts: dict[str, int]


class LiveFeatureMartExtractor:
    """Extract only columns declared by the Phase 5 plan over one connection."""

    def __init__(self, settings: DatabaseSettings, plan: ExtractionPlan, chunk_size: int = 10_000):
        self.settings = settings
        self.plan = plan
        self.chunk_size = chunk_size
        self.connection = DatabaseConnection(settings)

    def _read_table(self, db_connection: Any, table_name: str) -> pd.DataFrame:
        columns = list(plan_columns(self.plan, table_name))
        result = db_connection.execution_options(stream_results=True).execute(
            text(explicit_select_sql(table_name, columns))
        )
        batches: list[pd.DataFrame] = []
        while True:
            rows = result.fetchmany(self.chunk_size)
            if not rows:
                break
            batches.append(pd.DataFrame([tuple(row) for row in rows], columns=columns))
        return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame(columns=columns)

    def extract(self) -> ExtractionResult:
        """Read all planned source rows and exact source table counts."""

        self.settings.validate_for_connection()
        frames: dict[str, pd.DataFrame] = {}
        counts: dict[str, int] = {}
        try:
            with self.connection.connect() as db_connection:
                for table_name in self.plan.table_names:
                    count = db_connection.execute(text(explicit_count_sql(table_name))).scalar_one()
                    counts[table_name] = int(count)
                    frames[table_name] = self._read_table(db_connection, table_name)
        except Exception:
            raise
        finally:
            self.connection.dispose()
        if set(frames) != set(self.plan.table_names):
            raise FeatureMartError("Phase 5 extraction did not return every planned table.")
        return {"frames": frames, "source_row_counts": counts}
