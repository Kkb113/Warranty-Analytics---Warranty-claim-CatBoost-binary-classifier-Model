"""Date-grouped chronological expanding inner-fold construction."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from .models import InnerFold, InnerFoldPlan, OptimizationError
from .provenance import fold_content_sha256, inner_membership_sha256

KEY = "warranty_claim_key"
TARGET = "target__high_cost_claim_flag"
DATE = "claim_date"


def _date_strings(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce").dt.date
    if parsed.isna().any():
        raise OptimizationError("Inner folds require non-null claim_date values.")
    return parsed.map(date.isoformat)


def _choose_boundary(
    sorted_dates: list[str],
    counts: list[int],
    fraction: float,
    *,
    minimum_index: int,
) -> int:
    total = sum(counts)
    target = fraction * total
    cumulative = 0
    candidates: list[tuple[float, int, int]] = []
    for index, count in enumerate(counts):
        cumulative += count
        if index < minimum_index:
            continue
        if fraction < 1.0 and index >= len(sorted_dates) - 1:
            continue
        candidates.append((abs(cumulative - target), index, cumulative))
    if not candidates:
        raise OptimizationError("Chronological inner fold boundaries cannot be separated.")
    # sorted by distance then index makes an exact tie choose the earlier date.
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def build_inner_fold_plan(
    train_metadata: pd.DataFrame,
    train_targets: pd.DataFrame,
    *,
    fractions: tuple[float, ...] = (0.55, 0.70, 0.85, 1.0),
    minimum_train_positive: int = 40,
    minimum_validation_positive: int = 10,
) -> InnerFoldPlan:
    """Build three expanding folds from TRAIN rows using date groups only for boundaries."""

    required = {KEY, DATE}
    if not required.issubset(train_metadata.columns):
        raise OptimizationError("Inner fold metadata requires warranty_claim_key and claim_date.")
    if train_metadata[KEY].duplicated().any():
        raise OptimizationError("Inner fold metadata contains duplicate claim keys.")
    if set(train_metadata[KEY].astype(int)) != set(train_targets[KEY].astype(int)):
        raise OptimizationError("Inner fold target membership differs from TRAIN metadata.")
    metadata = train_metadata[[KEY, DATE]].copy()
    metadata[KEY] = metadata[KEY].astype(int)
    metadata[DATE] = _date_strings(metadata[DATE])
    target = train_targets[[KEY, TARGET]].copy()
    target[KEY] = target[KEY].astype(int)
    target[TARGET] = pd.to_numeric(target[TARGET], errors="coerce").astype("int8")
    joined = metadata.merge(target, on=KEY, how="left", validate="one_to_one")
    if joined[TARGET].isna().any():
        raise OptimizationError("Inner fold target contains missing TRAIN labels.")
    date_counts = joined.groupby(DATE, sort=True)[KEY].size()
    sorted_dates = [str(value) for value in date_counts.index.tolist()]
    counts = [int(date_counts.loc[value]) for value in date_counts.index]
    boundaries: list[int] = []
    minimum = 0
    for fraction in fractions:
        index = _choose_boundary(sorted_dates, counts, float(fraction), minimum_index=minimum)
        boundaries.append(index)
        minimum = index + 1
    if len(boundaries) != 4 or boundaries[-1] != len(sorted_dates) - 1:
        raise OptimizationError("Inner fold boundary plan must end at the final TRAIN date.")
    memberships: list[dict[str, Any]] = []
    folds: list[InnerFold] = []
    for fold_id, (train_index, validation_index) in enumerate(
        zip(boundaries[:3], boundaries[1:], strict=True), start=1
    ):
        train_dates = set(sorted_dates[: train_index + 1])
        validation_dates = set(sorted_dates[train_index + 1 : validation_index + 1])
        train_part = joined.loc[joined[DATE].isin(train_dates)].sort_values(KEY)
        validation_part = joined.loc[joined[DATE].isin(validation_dates)].sort_values(KEY)
        if train_part.empty or validation_part.empty:
            raise OptimizationError(f"Inner fold {fold_id} has an empty role.")
        train_max = str(train_part[DATE].max())
        validation_min = str(validation_part[DATE].min())
        validation_max = str(validation_part[DATE].max())
        if not train_max < validation_min:
            raise OptimizationError(f"Inner fold {fold_id} violates strict date chronology.")
        train_positive = int(train_part[TARGET].sum())
        validation_positive = int(validation_part[TARGET].sum())
        if (
            train_positive < minimum_train_positive
            or validation_positive < minimum_validation_positive
        ):
            raise OptimizationError(
                f"Inner fold {fold_id} lacks class sufficiency: "
                f"train_positive={train_positive}, validation_positive={validation_positive}."
            )
        if train_part[TARGET].nunique() != 2 or validation_part[TARGET].nunique() != 2:
            raise OptimizationError(f"Inner fold {fold_id} must contain both target classes.")
        train_keys = tuple(int(value) for value in train_part[KEY])
        validation_keys = tuple(int(value) for value in validation_part[KEY])
        folds.append(
            InnerFold(
                fold_id=fold_id,
                train_keys=train_keys,
                validation_keys=validation_keys,
                train_max_date=train_max,
                validation_min_date=validation_min,
                validation_max_date=validation_max,
                train_rows=len(train_part),
                validation_rows=len(validation_part),
                train_positive_count=train_positive,
                validation_positive_count=validation_positive,
                train_membership_sha256=inner_membership_sha256(train_keys),
                validation_membership_sha256=inner_membership_sha256(validation_keys),
            )
        )
        for key, claim_date in train_part[[KEY, DATE]].itertuples(index=False, name=None):
            memberships.append(
                {KEY: int(key), DATE: str(claim_date), "fold_id": fold_id, "role": "TRAIN"}
            )
        for key, claim_date in validation_part[[KEY, DATE]].itertuples(index=False, name=None):
            memberships.append(
                {KEY: int(key), DATE: str(claim_date), "fold_id": fold_id, "role": "VALIDATION"}
            )
    assignments = pd.DataFrame(memberships, columns=[KEY, DATE, "fold_id", "role"])
    assignments = assignments.sort_values(["fold_id", "role", KEY], kind="mergesort").reset_index(
        drop=True
    )
    manifest = {
        "source_outer_split": "TRAIN",
        "outer_train_count": int(len(joined)),
        "fold_count": len(folds),
        "boundary_algorithm": "date-grouped cumulative row fractions; closest date endpoint, ties earlier",
        "fraction_targets": list(fractions),
        "folds": [
            {
                "fold_id": fold.fold_id,
                "train_rows": fold.train_rows,
                "validation_rows": fold.validation_rows,
                "train_max_date": fold.train_max_date,
                "validation_min_date": fold.validation_min_date,
                "validation_max_date": fold.validation_max_date,
                "train_positive_count": fold.train_positive_count,
                "validation_positive_count": fold.validation_positive_count,
                "train_membership_sha256": fold.train_membership_sha256,
                "validation_membership_sha256": fold.validation_membership_sha256,
            }
            for fold in folds
        ],
    }
    manifest["fold_content_sha256"] = fold_content_sha256(assignments)
    content_hash = str(manifest["fold_content_sha256"])
    return InnerFoldPlan(
        assignments=assignments,
        folds=tuple(folds),
        manifest=manifest,
        content_sha256=content_hash,
    )


def fold_indices(
    matrix: pd.DataFrame,
    fold: InnerFold,
) -> tuple[pd.Index, pd.Index]:
    """Return deterministic positional indexes for one fold from a keyed matrix."""

    if KEY not in matrix.columns:
        raise OptimizationError("Fold matrix lacks warranty_claim_key.")
    by_key = pd.Series(matrix.index, index=matrix[KEY].astype(int))
    try:
        train_index = pd.Index([by_key.loc[key] for key in fold.train_keys])
        validation_index = pd.Index([by_key.loc[key] for key in fold.validation_keys])
    except KeyError as exc:
        raise OptimizationError("Fold membership key is missing from the model matrix.") from exc
    return train_index, validation_index
