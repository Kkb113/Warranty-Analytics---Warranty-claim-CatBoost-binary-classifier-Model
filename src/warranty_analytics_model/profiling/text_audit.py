"""Aggregate text-template diagnostics without publishing text."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

import pandas as pd

from .column_profile import normalize_text

TEXT_COLUMNS = ("complaint_description", "diagnostic_summary", "technician_notes", "repair_notes")
OUTCOME_WORDS = ("high cost", "approved", "rejected", "recall")


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def audit_text(
    frames: dict[str, pd.DataFrame],
    target_column: str = "high_cost_claim_flag",
    columns: Iterable[str] = TEXT_COLUMNS,
) -> dict[str, object]:
    """Report repeated normalized text using hashes and counts only."""

    fields: list[dict[str, object]] = []
    flags: list[str] = []
    for table, frame in sorted(frames.items()):
        for column in columns:
            if column not in frame:
                continue
            normalized = frame[column].map(normalize_text)
            non_empty = normalized[normalized != ""]
            counts = non_empty.value_counts()
            target = (
                pd.to_numeric(frame[target_column], errors="coerce")
                if target_column in frame
                else None
            )
            duplicated = counts[counts > 1]
            outcome_rates: dict[str, dict[str, float | None]] = {}
            for word in OUTCOME_WORDS:
                present = normalized.str.contains(re.escape(word), regex=True, na=False)
                rates: dict[str, float | None] = {}
                if target is not None:
                    for label in (0, 1):
                        group = target == label
                        rates[str(label)] = (
                            round(float(present[group].mean() * 100), 6) if group.any() else None
                        )
                outcome_rates[word] = rates
            max_purity = 0.0
            if target is not None and not duplicated.empty:
                grouped = pd.DataFrame({"text": normalized, "target": target}).query("text != ''")
                purity = grouped.groupby("text", observed=True)["target"].agg(["size", "nunique"])
                supported = purity[purity["size"] >= 5]
                max_purity = (
                    float((supported["nunique"] == 1).mean()) if not supported.empty else 0.0
                )
            leakage = bool(not duplicated.empty and max_purity >= 0.8)
            if leakage:
                flags.append("SYNTHETIC_TEXT_TEMPLATE_LEAKAGE")
            fields.append(
                {
                    "table": table,
                    "field": column,
                    "records": int(len(frame)),
                    "non_empty_records": int(len(non_empty)),
                    "normalized_distinct_texts": int(non_empty.nunique()),
                    "duplicated_text_values": int(len(duplicated)),
                    "records_in_duplicated_text_values": int(duplicated.sum())
                    if not duplicated.empty
                    else 0,
                    "maximum_repetitions": int(counts.max()) if not counts.empty else 0,
                    "duplicate_text_by_target": _duplicate_target_rates(normalized, target),
                    "outcome_word_rates_by_target": outcome_rates,
                    "repeated_template_purity_percentage": round(max_purity * 100, 6),
                    "synthetic_text_template_leakage": leakage,
                    "repeated_template_hashes": [
                        _hash(str(value)) for value in duplicated.head(20).index
                    ],
                }
            )
    return {"fields": fields, "flags": sorted(set(flags))}


def _duplicate_target_rates(text: pd.Series, target: pd.Series | None) -> dict[str, object]:
    if target is None:
        return {}
    frame = pd.DataFrame({"text": text, "target": target})
    frame = frame[(frame.text != "") & frame.target.isin([0, 1])]
    if frame.empty:
        return {}
    counts = frame.groupby(["text", "target"], observed=True).size().unstack(fill_value=0)
    return {
        "duplicate_text_groups": int((counts.sum(axis=1) > 1).sum()),
        "target_pure_duplicate_groups": int((counts.astype(bool).sum(axis=1) == 1).sum()),
        "target_mixed_duplicate_groups": int((counts.astype(bool).sum(axis=1) > 1).sum()),
    }
