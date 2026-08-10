"""Deterministic target-independent chronological boundary selection."""

from __future__ import annotations

from datetime import date

import pandas as pd

from .models import BoundaryResult, SplitError, SplitSettings


def _normalized_dates(snapshot: pd.DataFrame) -> pd.Series:
    """Return normalized claim dates and fail on missing/invalid values."""

    if "warranty_claim_key" not in snapshot or "claim__claim_date" not in snapshot:
        raise SplitError("Snapshot must contain warranty_claim_key and claim__claim_date.")
    if snapshot["warranty_claim_key"].isna().any():
        raise SplitError("Snapshot contains null warranty_claim_key values.")
    if snapshot["warranty_claim_key"].duplicated().any():
        raise SplitError("Snapshot warranty_claim_key values must be unique for splitting.")
    dates = pd.to_datetime(snapshot["claim__claim_date"], errors="coerce")
    if dates.isna().any():
        raise SplitError("Snapshot claim__claim_date contains null or invalid values.")
    return dates.dt.normalize()


def _closest_date_index(cumulative: pd.Series, target: float, candidates: list[int]) -> int:
    """Choose the closest cumulative count, using the earlier date on ties."""

    if not candidates:
        raise SplitError("No eligible date remains for a chronological split boundary.")
    return min(candidates, key=lambda index: (abs(float(cumulative.iloc[index]) - target), index))


def determine_boundaries(snapshot: pd.DataFrame, settings: SplitSettings) -> BoundaryResult:
    """Choose date boundaries using only dates, row counts, and configured fractions."""

    dates = _normalized_dates(snapshot)
    total_claims = int(len(snapshot))
    if total_claims < 3:
        raise SplitError("At least three claims are required to create three chronological splits.")
    counts = (
        pd.DataFrame({"claim_date": dates})
        .groupby("claim_date", as_index=False)
        .size()
        .rename(columns={"size": "row_count"})
        .sort_values("claim_date", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(counts) < 3:
        raise SplitError("At least three unique claim dates are required for three splits.")
    counts["cumulative_count"] = counts["row_count"].cumsum()
    train_target = total_claims * settings.train_fraction
    validation_end_target = total_claims * (settings.train_fraction + settings.validation_fraction)
    train_index = _closest_date_index(
        counts["cumulative_count"], train_target, list(range(len(counts)))
    )
    validation_candidates = list(range(train_index + 1, len(counts)))
    validation_index = _closest_date_index(
        counts["cumulative_count"], validation_end_target, validation_candidates
    )
    train_end = counts.loc[train_index, "claim_date"]
    validation_end = counts.loc[validation_index, "claim_date"]
    train_count = int(counts.loc[train_index, "cumulative_count"])
    validation_count = int(
        counts.loc[validation_index, "cumulative_count"]
        - counts.loc[train_index, "cumulative_count"]
    )
    test_count = total_claims - train_count - validation_count
    if train_count <= 0 or validation_count <= 0 or test_count <= 0:
        raise SplitError(
            "Chronological boundaries must produce non-empty TRAIN, VALIDATION, and TEST."
        )
    return BoundaryResult(
        total_claims=total_claims,
        unique_dates=int(len(counts)),
        train_target_count=float(train_target),
        validation_end_target_count=float(validation_end_target),
        train_end_date=date.fromisoformat(pd.Timestamp(train_end).date().isoformat()),
        validation_end_date=date.fromisoformat(pd.Timestamp(validation_end).date().isoformat()),
        train_date_count=int(train_index + 1),
        validation_date_count=int(validation_index - train_index),
        test_date_count=int(len(counts) - validation_index - 1),
        train_count=train_count,
        validation_count=validation_count,
        test_count=test_count,
    )
