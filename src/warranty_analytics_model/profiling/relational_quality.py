"""Foreign-key, missingness, and arithmetic quality checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def foreign_key_orphans(
    frames: Mapping[str, pd.DataFrame], contract: Any
) -> list[dict[str, object]]:
    """Count actual child rows whose declared parent key is absent."""

    table_map = getattr(contract, "table_map", {})
    output: list[dict[str, object]] = []
    for child_name in sorted(table_map):
        child_spec = table_map[child_name]
        child = frames.get(child_name)
        if child is None:
            continue
        for foreign_key in getattr(child_spec, "foreign_keys", []):
            parent_name = str(foreign_key.referenced_table)
            parent = frames.get(parent_name)
            parent_columns = list(foreign_key.referenced_columns)
            child_columns = list(foreign_key.parent_columns)
            if parent is None or not all(column in child for column in child_columns):
                continue
            if parent is None or not all(column in parent for column in parent_columns):
                continue
            parent_keys = parent[parent_columns].dropna().drop_duplicates()
            if len(child_columns) == 1:
                parent_key = set(parent_keys.iloc[:, 0].tolist())
            else:
                parent_key = set(parent_keys.astype(str).agg("|".join, axis=1).tolist())
            # Count child rows, not only distinct keys.
            valid_child_rows = child[child_columns].notna().all(axis=1)
            if len(child_columns) == 1:
                child_values = child[child_columns[0]]
                orphan_count = int((valid_child_rows & ~child_values.isin(parent_key)).sum())
            else:
                child_values = child[child_columns].astype(str).agg("|".join, axis=1)
                orphan_count = int((valid_child_rows & ~child_values.isin(parent_key)).sum())
            rows_checked = int(child[child_columns].notna().all(axis=1).sum())
            output.append(
                {
                    "parent_table": child_name,
                    "foreign_key": foreign_key.name,
                    "referenced_table": parent_name,
                    "rows_checked": rows_checked,
                    "orphan_count": orphan_count,
                    "orphan_percentage": round(orphan_count / rows_checked * 100, 6)
                    if rows_checked
                    else 0.0,
                }
            )
    return output


def cost_arithmetic_audit(repair_lines: pd.DataFrame) -> dict[str, object]:
    """Compare repair line cost with a transparent possible-cost expression."""

    required = {"line_cost", "part_quantity", "part_unit_cost", "labor_hours", "labor_rate"}
    if not required.issubset(repair_lines.columns):
        return {"available": False, "reason": "required repair columns are not present"}
    frame = repair_lines[list(required)].apply(pd.to_numeric, errors="coerce").dropna()
    if frame.empty:
        return {"available": True, "records": 0, "exact_match_count": 0, "near_match_count": 0}
    calculated = (
        frame["part_quantity"] * frame["part_unit_cost"]
        + frame["labor_hours"] * frame["labor_rate"]
    )
    difference = (frame["line_cost"] - calculated).abs()
    return {
        "available": True,
        "records": int(len(frame)),
        "exact_match_count": int((difference <= 0.01).sum()),
        "near_match_count": int((difference <= 1.0).sum()),
        "exact_match_percentage": round(float((difference <= 0.01).mean() * 100), 6),
        "near_match_percentage": round(float((difference <= 1.0).mean() * 100), 6),
        "interpretation": "Synthetic-generation evidence only; additional charges may exist.",
    }


def missingness_summary(frames: Mapping[str, pd.DataFrame]) -> list[dict[str, object]]:
    """Summarize missingness by table and column."""

    rows: list[dict[str, object]] = []
    for table, frame in sorted(frames.items()):
        for column in frame.columns:
            count = int(frame[column].isna().sum())
            rows.append(
                {
                    "table": table,
                    "field": str(column),
                    "row_count": int(len(frame)),
                    "null_count": count,
                    "null_percentage": round(count / len(frame) * 100, 6) if len(frame) else 0.0,
                }
            )
    return rows
