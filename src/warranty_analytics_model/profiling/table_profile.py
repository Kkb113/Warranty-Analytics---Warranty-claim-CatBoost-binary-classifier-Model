"""Table-level and all-column profiling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .column_profile import looks_like_date_column, profile_series


def _date_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        date_like = (
            pd.api.types.is_datetime64_any_dtype(frame[column])
            or looks_like_date_column(str(column))
            or str(column).casefold().endswith("_at")
        )
        values = (
            pd.to_datetime(frame[column], errors="coerce")
            if date_like
            else pd.Series(pd.NaT, index=frame.index)
        )
        if date_like and values.notna().any():
            columns.append(str(column))
    return columns


def profile_table(
    table_name: str,
    frame: pd.DataFrame,
    *,
    table_spec: Any | None = None,
    row_count_estimate: int | None = None,
    top_categories: int = 20,
    percentiles: tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
    rare_category_thresholds: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, object]:
    """Return a secret-safe profile for one DataFrame."""

    row_count = int(len(frame))
    pk_columns: list[str] = []
    nullable_column_count: int | None = None
    if table_spec is not None:
        pk = getattr(table_spec, "primary_key", None)
        pk_columns = list(getattr(pk, "columns", []) or [])
        contract_columns = list(getattr(table_spec, "columns", []) or [])
        nullable_column_count = sum(
            1 for column in contract_columns if bool(getattr(column, "nullable", False))
        )
    if not pk_columns:
        candidates = [str(column) for column in frame.columns if str(column).endswith("_key")]
        if len(candidates) == 1:
            pk_columns = candidates

    duplicate_pk_values = 0
    duplicate_pk_records = 0
    pk_unique = True
    if pk_columns and all(column in frame.columns for column in pk_columns):
        keys = frame[pk_columns].astype("string").fillna("<null>").astype(str)
        key_index = (
            keys.iloc[:, 0] if len(pk_columns) == 1 else keys.astype(str).agg("|".join, axis=1)
        )
        repeated = key_index.value_counts()
        duplicate_pk_values = int((repeated > 1).sum())
        duplicate_pk_records = int(repeated[repeated > 1].sum()) if duplicate_pk_values else 0
        pk_unique = duplicate_pk_values == 0 and int(key_index.isna().sum()) == 0

    date_ranges: dict[str, dict[str, object]] = {}
    for column in _date_columns(frame):
        values = pd.to_datetime(frame[column], errors="coerce").dropna()
        date_ranges[column] = {
            "min": values.min().isoformat() if not values.empty else None,
            "max": values.max().isoformat() if not values.empty else None,
        }

    column_profiles = {
        str(column): profile_series(
            frame[column],
            column_name=str(column),
            top_categories=top_categories,
            percentiles=percentiles,
            rare_category_thresholds=rare_category_thresholds,
        )
        for column in frame.columns
    }
    return {
        "table": table_name,
        "row_count": row_count,
        "column_count": int(len(frame.columns)),
        "primary_key_columns": pk_columns,
        "primary_key_unique": pk_unique,
        "duplicate_primary_key_value_count": duplicate_pk_values,
        "duplicate_primary_key_record_count": duplicate_pk_records,
        "full_duplicate_row_count": int(frame.duplicated(keep=False).sum()),
        "date_ranges": date_ranges,
        "approximate_memory_bytes": int(frame.memory_usage(index=True, deep=True).sum()),
        "contract_row_count_estimate": row_count_estimate,
        "row_count_difference_from_estimate": (
            row_count - row_count_estimate if row_count_estimate is not None else None
        ),
        "nullable_column_count": nullable_column_count,
        "columns_with_nulls": int(frame.isna().any(axis=0).sum()),
        "column_profiles": column_profiles,
    }


def profile_tables(
    frames: Mapping[str, pd.DataFrame],
    *,
    contract: Any | None = None,
    top_categories: int = 20,
    percentiles: tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
    rare_category_thresholds: tuple[int, ...] = (1, 5, 10, 20),
) -> list[dict[str, object]]:
    """Profile each supplied table in deterministic name order."""

    profiles: list[dict[str, object]] = []
    contract_map = getattr(contract, "table_map", {}) if contract is not None else {}
    for name in sorted(frames):
        spec = contract_map.get(name)
        profiles.append(
            profile_table(
                name,
                frames[name],
                table_spec=spec,
                row_count_estimate=getattr(spec, "estimated_rows", None),
                top_categories=top_categories,
                percentiles=percentiles,
                rare_category_thresholds=rare_category_thresholds,
            )
        )
    return profiles
