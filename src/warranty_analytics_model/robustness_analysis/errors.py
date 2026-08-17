"""Aggregate and local-only error cohort diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def error_cohorts(keys: Any, targets: Any, probabilities: Any, threshold: float) -> pd.DataFrame:
    y = np.asarray(targets, dtype="int8")
    p = np.asarray(probabilities, dtype="float64")
    predicted = p >= float(threshold)
    error = np.where(
        (predicted == 1) & (y == 1),
        "TRUE_POSITIVE",
        np.where(
            (predicted == 1) & (y == 0),
            "FALSE_POSITIVE",
            np.where((predicted == 0) & (y == 1), "FALSE_NEGATIVE", "TRUE_NEGATIVE"),
        ),
    )
    margin = np.where(predicted, p - float(threshold), float(threshold) - p)
    return (
        pd.DataFrame(
            {
                "warranty_claim_key": np.asarray(keys, dtype="int64"),
                "target": y,
                "probability": p,
                "frozen_threshold": float(threshold),
                "predicted_class": predicted.astype("int8"),
                "error_type": error,
                "threshold_margin": margin,
            }
        )
        .sort_values("warranty_claim_key", kind="mergesort")
        .reset_index(drop=True)
    )


def high_confidence_errors(cohorts: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    selected = cohorts.loc[cohorts["error_type"].isin(["FALSE_POSITIVE", "FALSE_NEGATIVE"])].copy()
    return (
        selected.sort_values(
            ["error_type", "threshold_margin", "warranty_claim_key"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("error_type", sort=True, group_keys=False)
        .head(int(limit))
        .reset_index(drop=True)
    )


def error_profile(cohorts: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    frame = cohorts.merge(context, on="warranty_claim_key", how="left", validate="one_to_one")
    rows = []
    for error_type, group in frame.groupby("error_type", sort=True):
        row: dict[str, Any] = {
            "error_type": str(error_type),
            "row_count": int(len(group)),
            "mean_probability": float(group["probability"].mean()) if len(group) else 0.0,
            "mean_threshold_margin": float(group["threshold_margin"].mean()) if len(group) else 0.0,
        }
        for column in ("claim__claim_date", "feature_missingness_count", "risk_decile"):
            if column in group:
                row[f"{column}_missing_count"] = int(group[column].isna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["error_cohorts", "error_profile", "high_confidence_errors"]
