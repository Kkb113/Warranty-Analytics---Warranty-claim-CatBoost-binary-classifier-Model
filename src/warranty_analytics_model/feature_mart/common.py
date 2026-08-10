"""Shared pure DataFrame helpers for claim snapshots and history bridges."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from .models import FeatureMartError


def as_datetime(values: Any) -> pd.Series:
    """Convert a date-like series without filling or shifting missing values."""

    return pd.to_datetime(values, errors="coerce")


def assert_unique_key(frame: pd.DataFrame, key: str, label: str) -> None:
    """Fail closed when a dimension or source primary key is not unique."""

    if key not in frame.columns:
        raise FeatureMartError(f"{label} is missing required key column: {key}")
    if frame[key].isna().any():
        raise FeatureMartError(f"{label} contains null key values: {key}")
    if frame[key].duplicated(keep=False).any():
        raise FeatureMartError(f"{label} contains duplicate key values: {key}")


def merge_many_to_one(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str | list[str],
    label: str,
    suffixes: tuple[str, str] = ("", "_dimension"),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Perform a left many-to-one join and prove it did not multiply rows."""

    keys = [on] if isinstance(on, str) else list(on)
    missing_left = sorted(set(keys) - set(left.columns))
    missing_right = sorted(set(keys) - set(right.columns))
    if missing_left or missing_right:
        raise FeatureMartError(
            f"{label} join keys missing; left={missing_left}, right={missing_right}"
        )
    if right.duplicated(keys, keep=False).any():
        raise FeatureMartError(f"{label} dimension keys are not unique: {keys}")
    before = len(left)
    merged = left.merge(
        right,
        how="left",
        on=keys,
        validate="many_to_one",
        indicator="_phase5_join_status",
        suffixes=suffixes,
    )
    if len(merged) != before:
        raise FeatureMartError(
            f"{label} multiplied claim rows: before={before}, after={len(merged)}"
        )
    unmatched = int((merged["_phase5_join_status"] == "left_only").sum())
    merged = merged.drop(columns=["_phase5_join_status"])
    return merged, {
        "input_rows": before,
        "output_rows": len(merged),
        "unmatched_rows": unmatched,
        "multiplication_count": len(merged) - before,
    }


def assert_pair_unique(frame: pd.DataFrame, pair_key: Iterable[str], label: str) -> None:
    """Fail when a one-claim-to-one-source-record pair is duplicated."""

    keys = list(pair_key)
    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise FeatureMartError(f"{label} is missing bridge key columns: {', '.join(missing)}")
    if frame.duplicated(keys, keep=False).any():
        raise FeatureMartError(f"{label} contains duplicate bridge pairs: {', '.join(keys)}")


def empty_with_columns(columns: Iterable[str]) -> pd.DataFrame:
    """Return a typed-empty frame with deterministic columns."""

    return pd.DataFrame(columns=list(dict.fromkeys(columns)))


def history_diagnostics(
    eligible_claims: pd.DataFrame,
    bridge: pd.DataFrame,
    *,
    claim_key: str = "warranty_claim_key",
    bridge_claim_key: str = "current_warranty_claim_key",
) -> dict[str, int | float]:
    """Calculate aggregate history coverage diagnostics only."""

    eligible = int(len(eligible_claims))
    counts = (
        bridge.groupby(bridge_claim_key, dropna=False).size()
        if not bridge.empty
        else pd.Series(dtype="int64")
    )
    counts = counts.reindex(eligible_claims[claim_key].drop_duplicates(), fill_value=0)
    return {
        "eligible_claims": eligible,
        "claims_with_history": int((counts > 0).sum()),
        "claims_without_history": int((counts == 0).sum()),
        "total_bridge_rows": int(len(bridge)),
        "minimum_rows_per_claim": int(counts.min()) if len(counts) else 0,
        "median_rows_per_claim": float(counts.median()) if len(counts) else 0.0,
        "p95_rows_per_claim": float(counts.quantile(0.95)) if len(counts) else 0.0,
        "maximum_rows_per_claim": int(counts.max()) if len(counts) else 0,
    }


def deterministic_sort(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Sort stablely on available keys without changing values or imputing nulls."""

    sort_columns = [column for column in columns if column in frame.columns]
    if frame.empty or not sort_columns:
        return frame.reset_index(drop=True)
    return frame.sort_values(sort_columns, kind="mergesort", na_position="last").reset_index(
        drop=True
    )
