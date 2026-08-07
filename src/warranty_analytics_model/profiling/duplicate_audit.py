"""Deterministic duplicate and repeated-scenario fingerprints."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import pandas as pd

_SURROGATE_PARTS = ("_key", "_id", "serial", "target", "flag")


def _fingerprint(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    selected = [column for column in columns if column in frame]
    if not selected:
        return pd.Series(["empty"] * len(frame), index=frame.index)
    values = frame[selected].astype("string").fillna("<null>").agg("|".join, axis=1)
    return values.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:16])


def _fingerprint_summary(
    label: str, frame: pd.DataFrame, columns: list[str], target_column: str
) -> dict[str, object]:
    fingerprint = _fingerprint(frame, columns)
    counts = fingerprint.value_counts()
    duplicated = counts[counts > 1]
    spans_target = 0
    if target_column in frame:
        grouped = pd.DataFrame({"fingerprint": fingerprint, "target": frame[target_column]})
        spans_target = int(
            (grouped.groupby("fingerprint", observed=True)["target"].nunique(dropna=True) > 1).sum()
        )
    return {
        "scenario": label,
        "fingerprint_columns": columns,
        "unique_fingerprints": int(counts.size),
        "duplicated_fingerprints": int(duplicated.size),
        "records_in_duplicated_fingerprints": int(duplicated.sum()) if not duplicated.empty else 0,
        "maximum_repetitions": int(counts.max()) if not counts.empty else 0,
        "fingerprints_spanning_target_values": spans_target,
        "possible_split_contamination": bool(not duplicated.empty),
        "recommendation": (
            "Use group-aware or fingerprint-aware Phase 6 splits and report family-level support."
            if not duplicated.empty
            else "No repeated fingerprint family detected in this diagnostic input."
        ),
    }


def audit_duplicates(
    frames: dict[str, pd.DataFrame], target_column: str = "high_cost_claim_flag"
) -> dict[str, object]:
    """Audit exact rows and scenario fingerprints without deleting anything."""

    summaries: list[dict[str, object]] = []
    for table, frame in sorted(frames.items()):
        columns = [str(column) for column in frame.columns]
        exact_count = int(frame.duplicated(keep=False).sum())
        summaries.append(
            {
                "scenario": f"exact_rows:{table}",
                "fingerprint_columns": [],
                "unique_fingerprints": int(len(frame.drop_duplicates())),
                "duplicated_fingerprints": int(frame.duplicated(keep=False).any()),
                "records_in_duplicated_fingerprints": exact_count,
                "maximum_repetitions": int(frame.value_counts(dropna=False).max())
                if len(frame)
                else 0,
                "fingerprints_spanning_target_values": 0,
                "possible_split_contamination": exact_count > 0,
                "recommendation": "Retain records; use a duplicate-aware Phase 6 split if repeated families remain.",
            }
        )
        fingerprint_columns = [
            column
            for column in columns
            if not any(part in column.casefold() for part in _SURROGATE_PARTS)
        ]
        if target_column in fingerprint_columns:
            fingerprint_columns.remove(target_column)
        if fingerprint_columns:
            summaries.append(_fingerprint_summary(table, frame, fingerprint_columns, target_column))

    special_columns = {
        "claims_without_surrogates": [
            column
            for column in frames.get("dbo.fact_warranty_claim", pd.DataFrame()).columns
            if column not in {"warranty_claim_key", "claim_id", target_column}
        ],
        "service_scenario": [
            column
            for column in frames.get("dbo.fact_service_event", pd.DataFrame()).columns
            if column not in {"service_event_key", "service_event_id"}
        ],
        "repair_pattern": [
            column
            for column in frames.get("dbo.fact_repair_line", pd.DataFrame()).columns
            if column
            not in {"repair_line_key", "warranty_claim_key", "service_event_key", "technician_id"}
        ],
        "telemetry_sequence": [
            column
            for column in frames.get("dbo.fact_telemetry_monthly", pd.DataFrame()).columns
            if column not in {"telemetry_month_key"}
        ],
    }
    for label, columns in special_columns.items():
        table = (
            "dbo.fact_warranty_claim"
            if label == "claims_without_surrogates"
            else "dbo.fact_service_event"
            if label == "service_scenario"
            else "dbo.fact_repair_line"
            if label == "repair_pattern"
            else "dbo.fact_telemetry_monthly"
        )
        if columns and table in frames:
            summaries.append(_fingerprint_summary(label, frames[table], columns, target_column))
    return {
        "summaries": summaries,
        "duplicates_found": any(item["possible_split_contamination"] for item in summaries),
    }
