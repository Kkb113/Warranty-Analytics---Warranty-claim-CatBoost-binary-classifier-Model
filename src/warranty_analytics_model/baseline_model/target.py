"""Fail-closed TRAIN/VALIDATION-only target access for Phase 9."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds

from .models import BaselineModelError, DevelopmentTargets

KEY = "warranty_claim_key"
TARGET = "target__high_cost_claim_flag"


def target_content_sha256(frame: pd.DataFrame) -> str:
    """Hash canonical claim-key/target pairs without including TEST."""

    ordered = frame[[KEY, TARGET]].sort_values(KEY, kind="mergesort")
    records = [[int(key), int(target)] for key, target in ordered.itertuples(index=False)]
    payload = json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_development_targets(
    claim_snapshot_path: Path,
    assignments: pd.DataFrame,
) -> DevelopmentTargets:
    """Load only TRAIN and VALIDATION labels through a filtered Arrow scan."""

    required = {KEY, "split"}
    if not required.issubset(assignments.columns):
        raise BaselineModelError("Phase 6 assignments lack target-access control columns.")
    if assignments[KEY].duplicated().any():
        raise BaselineModelError("Phase 6 assignments contain duplicate claim keys.")
    train_keys = set(assignments.loc[assignments["split"] == "TRAIN", KEY].astype(int))
    validation_keys = set(assignments.loc[assignments["split"] == "VALIDATION", KEY].astype(int))
    test_keys = set(assignments.loc[assignments["split"] == "TEST", KEY].astype(int))
    development_keys = train_keys | validation_keys
    if not train_keys or not validation_keys or not test_keys:
        raise BaselineModelError("TRAIN, VALIDATION, and TEST membership must all be nonempty.")
    if (train_keys & validation_keys) or (development_keys & test_keys):
        raise BaselineModelError("Phase 6 split memberships overlap.")
    dataset = ds.dataset(str(claim_snapshot_path), format="parquet")
    schema_names = set(dataset.schema.names)
    if not {KEY, TARGET}.issubset(schema_names):
        raise BaselineModelError("Phase 5 claim snapshot lacks the authoritative target columns.")
    table = dataset.to_table(
        columns=[KEY, TARGET],
        filter=ds.field(KEY).isin(sorted(development_keys)),
    )
    loaded = table.to_pandas()
    if loaded[KEY].duplicated().any():
        raise BaselineModelError("Filtered development target contains duplicate claim keys.")
    loaded_keys = set(loaded[KEY].astype(int))
    if loaded_keys != development_keys:
        missing = len(development_keys - loaded_keys)
        extra = len(loaded_keys - development_keys)
        raise BaselineModelError(
            f"Development target membership mismatch: missing={missing}, extra={extra}."
        )
    if loaded_keys & test_keys:
        raise BaselineModelError("TEST target rows were materialized; Phase 9 is blocked.")
    if loaded[TARGET].isna().any():
        raise BaselineModelError("Development target contains NULL values.")
    numeric_target = pd.to_numeric(loaded[TARGET], errors="coerce")
    if numeric_target.isna().any():
        raise BaselineModelError("Development target contains a non-numeric value.")
    values = set(numeric_target.astype(int))
    if values != {0, 1}:
        raise BaselineModelError(f"Development target must contain exactly {{0, 1}}; got {values}.")
    loaded[TARGET] = numeric_target.astype("int8")
    joined = assignments.loc[
        assignments["split"].isin(["TRAIN", "VALIDATION"]), [KEY, "split"]
    ].merge(
        loaded,
        on=KEY,
        how="left",
        validate="one_to_one",
    )
    train = (
        joined.loc[joined["split"] == "TRAIN", [KEY, TARGET]]
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    validation = (
        joined.loc[joined["split"] == "VALIDATION", [KEY, TARGET]]
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    if len(train) != len(train_keys) or len(validation) != len(validation_keys):
        raise BaselineModelError("TRAIN or VALIDATION target row count changed after the join.")
    if train[TARGET].nunique() != 2 or validation[TARGET].nunique() != 2:
        raise BaselineModelError("TRAIN and VALIDATION must each contain both target classes.")
    audit: dict[str, Any] = {
        "train_target_rows_loaded": len(train),
        "validation_target_rows_loaded": len(validation),
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
        "development_target_source": "Phase 5 claim snapshot",
    }
    return DevelopmentTargets(
        train=train,
        validation=validation,
        train_target_content_sha256=target_content_sha256(train),
        validation_target_content_sha256=target_content_sha256(validation),
        audit=audit,
    )


def target_summary(targets: DevelopmentTargets) -> dict[str, Any]:
    """Return authorized aggregate TRAIN/VALIDATION target statistics only."""

    result: dict[str, Any] = {}
    for name, frame, digest in (
        ("train", targets.train, targets.train_target_content_sha256),
        ("validation", targets.validation, targets.validation_target_content_sha256),
    ):
        positives = int(frame[TARGET].sum())
        total = len(frame)
        result[name] = {
            "rows": total,
            "positive_count": positives,
            "negative_count": total - positives,
            "prevalence": positives / total,
            "target_content_sha256": digest,
        }
    return result
