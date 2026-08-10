"""Claim-level split assignments and membership integrity helpers."""

from __future__ import annotations

import pandas as pd

from ..feature_mart.common import deterministic_sort
from .boundary import _normalized_dates
from .models import BoundaryResult, SplitError

ASSIGNMENT_COLUMNS = ["warranty_claim_key", "claim_date", "split"]
SPLIT_VALUES = frozenset({"TRAIN", "VALIDATION", "TEST"})


def build_split_assignments(
    snapshot: pd.DataFrame,
    boundaries: BoundaryResult,
) -> pd.DataFrame:
    """Assign every claim to exactly one date-ordered split."""

    dates = _normalized_dates(snapshot)
    train_end = pd.Timestamp(boundaries.train_end_date)
    validation_end = pd.Timestamp(boundaries.validation_end_date)
    split = pd.Series("TEST", index=snapshot.index, dtype="object")
    split.loc[dates <= train_end] = "TRAIN"
    split.loc[(dates > train_end) & (dates <= validation_end)] = "VALIDATION"
    assignments = pd.DataFrame(
        {
            "warranty_claim_key": snapshot["warranty_claim_key"].to_numpy(copy=True),
            "claim_date": dates.to_numpy(copy=True),
            "split": split.to_numpy(copy=True),
        }
    )
    assignments = deterministic_sort(assignments, ["claim_date", "warranty_claim_key"])
    validate_assignment_frame(assignments, expected_claim_count=len(snapshot))
    return assignments


def validate_assignment_frame(
    assignments: pd.DataFrame,
    *,
    expected_claim_count: int | None = None,
) -> None:
    """Fail closed on assignment grain, coverage, and allowed split values."""

    missing = sorted(set(ASSIGNMENT_COLUMNS) - set(assignments.columns))
    if missing:
        raise SplitError(f"Split assignments are missing required columns: {', '.join(missing)}")
    if "target__high_cost_claim_flag" in assignments.columns:
        raise SplitError("Split assignments must not contain the target column.")
    if assignments["warranty_claim_key"].isna().any():
        raise SplitError("Split assignments contain null claim keys.")
    if assignments["warranty_claim_key"].duplicated().any():
        raise SplitError("Split assignments contain duplicate claim keys.")
    dates = pd.to_datetime(assignments["claim_date"], errors="coerce")
    if dates.isna().any():
        raise SplitError("Split assignments contain null or invalid claim dates.")
    if assignments["split"].isna().any():
        raise SplitError("Split assignments contain null split labels.")
    invalid = sorted(set(assignments["split"].astype(str)) - SPLIT_VALUES)
    if invalid:
        raise SplitError(f"Split assignments contain invalid split labels: {', '.join(invalid)}")
    if expected_claim_count is not None and len(assignments) != expected_claim_count:
        raise SplitError(
            f"Split assignments contain {len(assignments)} rows; expected {expected_claim_count}."
        )


def split_date_ranges(assignments: pd.DataFrame) -> dict[str, dict[str, str | int]]:
    """Return aggregate date ranges and counts without exposing claim keys."""

    validate_assignment_frame(assignments)
    result: dict[str, dict[str, str | int]] = {}
    for split in ("TRAIN", "VALIDATION", "TEST"):
        subset = assignments.loc[assignments["split"] == split]
        dates = pd.to_datetime(subset["claim_date"], errors="coerce")
        result[split] = {
            "row_count": int(len(subset)),
            "earliest_claim_date": dates.min().date().isoformat() if len(subset) else "",
            "latest_claim_date": dates.max().date().isoformat() if len(subset) else "",
        }
    return result


def assignment_date_order_errors(assignments: pd.DataFrame) -> list[str]:
    """Return chronological and same-day integrity errors."""

    validate_assignment_frame(assignments)
    errors: list[str] = []
    normalized = pd.to_datetime(assignments["claim_date"], errors="coerce").dt.normalize()
    date_split_counts = (
        pd.DataFrame({"claim_date": normalized, "split": assignments["split"].astype(str)})
        .groupby("claim_date")["split"]
        .nunique()
    )
    if (date_split_counts > 1).any():
        errors.append("A claim date appears in more than one split.")
    ranges = split_date_ranges(assignments)
    train_latest = str(ranges["TRAIN"]["latest_claim_date"])
    validation_earliest = str(ranges["VALIDATION"]["earliest_claim_date"])
    validation_latest = str(ranges["VALIDATION"]["latest_claim_date"])
    test_earliest = str(ranges["TEST"]["earliest_claim_date"])
    if train_latest and validation_earliest and train_latest >= validation_earliest:
        errors.append("TRAIN dates do not precede VALIDATION dates.")
    if validation_latest and test_earliest and validation_latest >= test_earliest:
        errors.append("VALIDATION dates do not precede TEST dates.")
    return errors
