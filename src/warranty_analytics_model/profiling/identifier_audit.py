"""Identifier pattern and memorization diagnostics."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

import pandas as pd

IDENTIFIER_COLUMNS = (
    "warranty_claim_key",
    "claim_id",
    "service_event_key",
    "truck_key",
    "vin",
    "production_batch_id",
    "component_serial_no",
    "component_lot_no",
    "engine_serial_no",
    "transmission_serial_no",
    "technician_id",
    "inspector_id",
)


def _hash(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def audit_identifiers(
    frame: pd.DataFrame,
    target_column: str = "high_cost_claim_flag",
    columns: Iterable[str] = IDENTIFIER_COLUMNS,
    *,
    minimum_group_support: int = 5,
) -> dict[str, object]:
    """Report aggregate identifier patterns, never raw identifier values."""

    target = (
        pd.to_numeric(frame[target_column], errors="coerce") if target_column in frame else None
    )
    audits: list[dict[str, Any]] = []
    flags: list[str] = []
    for column in columns:
        if column not in frame:
            continue
        values = frame[column].astype("string")
        non_null = values.dropna()
        item: dict[str, Any] = {
            "field": column,
            "records": int(len(frame)),
            "non_null": int(non_null.size),
            "distinct": int(non_null.nunique()),
            "uniqueness_percentage": round(non_null.nunique() / len(non_null) * 100, 6)
            if len(non_null)
            else 0.0,
            "prefix_groups": [],
            "suffix_groups": [],
            "numeric_sequence": None,
            "date_like_fragment": False,
            "max_supported_pure_group": 0,
            "synthetic_identifier_leakage": False,
        }
        structured = non_null.str.len().fillna(0).ge(2)
        if structured.any():
            prefix = values.str.slice(0, 3)
            suffix = values.str.slice(-3)
            for label, grouped in (("prefix_groups", prefix), ("suffix_groups", suffix)):
                if target is not None:
                    rates = pd.DataFrame({"group": grouped, "target": target}).dropna()
                    rates = (
                        rates[rates.target.isin([0, 1])]
                        .groupby("group")["target"]
                        .agg(records="size", rate="mean")
                    )
                    entries = []
                    for group, row in (
                        rates[rates.records >= minimum_group_support].head(20).iterrows()
                    ):
                        entries.append(
                            {
                                "group_hash": _hash(group),
                                "records": int(row.records),
                                "positive_rate_percentage": round(float(row.rate) * 100, 6),
                            }
                        )
                        if row.rate in (0.0, 1.0):
                            item["synthetic_identifier_leakage"] = True
                            item["max_supported_pure_group"] = max(
                                int(item["max_supported_pure_group"]), int(row.records)
                            )
                    item[label] = entries
        numeric = pd.to_numeric(values.str.extract(r"(\d+)$", expand=False), errors="coerce")
        if numeric.notna().sum() >= minimum_group_support:
            item["numeric_sequence"] = {
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "positive_rate_by_quartile": _numeric_quartile_rates(numeric, target),
            }
        item["date_like_fragment"] = bool(
            values.str.contains(r"20\d{2}[01]\d[0-3]\d", regex=True, na=False).any()
        )
        if item["synthetic_identifier_leakage"]:
            flags.append("SYNTHETIC_IDENTIFIER_LEAKAGE")
        audits.append(item)
    return {"fields": audits, "flags": sorted(set(flags))}


def _numeric_quartile_rates(values: pd.Series, target: pd.Series | None) -> list[dict[str, object]]:
    if target is None:
        return []
    valid = pd.DataFrame({"value": values, "target": target}).dropna()
    valid = valid[valid.target.isin([0, 1])]
    if valid.empty:
        return []
    try:
        valid["quartile"] = pd.qcut(valid.value, q=4, duplicates="drop")
    except ValueError:
        return []
    return [
        {"records": int(row.size), "positive_rate_percentage": round(float(row.mean()) * 100, 6)}
        for _, row in valid.groupby("quartile", observed=True)["target"]
        .agg(size="size", mean="mean")
        .iterrows()
    ]
