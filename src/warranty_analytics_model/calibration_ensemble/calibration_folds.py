"""Locked two-level temporal calibration folds."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..catboost_optimization.provenance import canonical_json_sha256

CALIBRATION_FOLDS: tuple[dict[str, Any], ...] = (
    {"calibration_fold_id": "C1", "train_source_folds": (1,), "validation_source_fold": 2},
    {
        "calibration_fold_id": "C2",
        "train_source_folds": (1, 2),
        "validation_source_fold": 3,
    },
)


def calibration_fold_assignments(source_oof: pd.DataFrame) -> pd.DataFrame:
    """Assign every source OOF row to its calibration train/validation role."""

    required = [
        "warranty_claim_key",
        "track",
        "strategy_id",
        "fold_id",
        "high_cost_probability",
        "claim_date",
    ]
    if list(source_oof.columns) != required:
        raise ValueError("Source OOF schema does not match the Phase 13 fold contract.")
    rows: list[dict[str, Any]] = []
    for definition in CALIBRATION_FOLDS:
        fold_id = str(definition["calibration_fold_id"])
        train_folds = tuple(int(value) for value in definition["train_source_folds"])
        validation_fold = int(definition["validation_source_fold"])
        for row in source_oof.to_dict("records"):
            source_fold = int(row["fold_id"])
            if source_fold in train_folds:
                role = "CALIBRATION_TRAIN"
            elif source_fold == validation_fold:
                role = "CALIBRATION_VALIDATION"
            else:
                continue
            rows.append(
                {
                    "warranty_claim_key": int(row["warranty_claim_key"]),
                    "track": str(row["track"]),
                    "strategy_id": str(row["strategy_id"]),
                    "source_fold_id": source_fold,
                    "calibration_fold_id": fold_id,
                    "role": role,
                    "claim_date": str(row["claim_date"]),
                }
            )
    result = pd.DataFrame(
        rows,
        columns=[
            "warranty_claim_key",
            "track",
            "strategy_id",
            "source_fold_id",
            "calibration_fold_id",
            "role",
            "claim_date",
        ],
    )
    if result.duplicated(
        ["warranty_claim_key", "track", "strategy_id", "calibration_fold_id"]
    ).any():
        raise ValueError("Calibration fold assignments contain duplicate rows.")
    return result.sort_values(
        ["calibration_fold_id", "role", "source_fold_id", "track", "warranty_claim_key"],
        kind="mergesort",
    ).reset_index(drop=True)


def calibration_fold_manifest(assignments: pd.DataFrame) -> tuple[dict[str, Any], str]:
    required = [
        "warranty_claim_key",
        "track",
        "strategy_id",
        "source_fold_id",
        "calibration_fold_id",
        "role",
        "claim_date",
    ]
    if list(assignments.columns) != required:
        raise ValueError("Calibration assignment schema changed.")
    records = [
        [
            int(row.warranty_claim_key),
            str(row.track),
            str(row.strategy_id),
            int(row.source_fold_id),
            str(row.calibration_fold_id),
            str(row.role),
            str(row.claim_date),
        ]
        for row in assignments.itertuples(index=False)
    ]
    digest = canonical_json_sha256({"columns": required, "rows": records})
    folds: list[dict[str, Any]] = []
    for definition in CALIBRATION_FOLDS:
        fold_id = str(definition["calibration_fold_id"])
        part = assignments[assignments["calibration_fold_id"] == fold_id]
        train = part[part["role"] == "CALIBRATION_TRAIN"]
        validation = part[part["role"] == "CALIBRATION_VALIDATION"]
        if train.empty or validation.empty:
            raise ValueError(f"Calibration fold {fold_id} is empty.")
        max_train = max(str(value) for value in train["claim_date"])
        min_validation = min(str(value) for value in validation["claim_date"])
        if not max_train < min_validation:
            raise ValueError(f"Calibration fold {fold_id} chronology is not strict.")
        folds.append(
            {
                "calibration_fold_id": fold_id,
                "train_source_folds": list(definition["train_source_folds"]),
                "validation_source_fold": int(definition["validation_source_fold"]),
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "train_max_claim_date": max_train,
                "validation_min_claim_date": min_validation,
            }
        )
    return {
        "phase": 13,
        "folds": folds,
        "calibration_fold_content_sha256": digest,
        "source_folds_used_for_evaluation": [2, 3],
    }, digest


__all__ = ["CALIBRATION_FOLDS", "calibration_fold_assignments", "calibration_fold_manifest"]
