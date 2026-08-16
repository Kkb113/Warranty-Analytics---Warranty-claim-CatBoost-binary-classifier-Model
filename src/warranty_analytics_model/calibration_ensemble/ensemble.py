"""Controlled convex T1/T3 ensemble evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .calibration_metrics import probability_metrics
from .config import ENSEMBLE_WEIGHTS
from .selection import select_ensemble


def align_selected_tracks(t1: pd.DataFrame, t3: pd.DataFrame) -> pd.DataFrame:
    required = {
        "warranty_claim_key",
        "source_fold_id",
        "calibration_fold_id",
        "track",
        "calibrated_probability",
        "target",
    }
    if not required.issubset(t1.columns) or not required.issubset(t3.columns):
        raise ValueError("Selected calibrated track schema is incomplete.")
    key_columns = ["warranty_claim_key", "source_fold_id", "calibration_fold_id"]
    left_keys = set(map(tuple, t1[key_columns].itertuples(index=False, name=None)))
    right_keys = set(map(tuple, t3[key_columns].itertuples(index=False, name=None)))
    if left_keys != right_keys:
        raise ValueError("T1 and T3 calibration populations do not match exactly.")
    left = t1[key_columns + ["calibrated_probability", "target"]].rename(
        columns={"calibrated_probability": "p_t1", "target": "target_t1"}
    )
    right = t3[key_columns + ["calibrated_probability", "target"]].rename(
        columns={"calibrated_probability": "p_t3", "target": "target_t3"}
    )
    merged = left.merge(right, on=key_columns, how="outer", validate="one_to_one", indicator=True)
    if (merged["_merge"] != "both").any() or (merged["target_t1"] != merged["target_t3"]).any():
        raise ValueError("T1 and T3 target/alignment membership differs.")
    result = merged.drop(columns=["_merge", "target_t3"]).rename(columns={"target_t1": "target"})
    result = result[key_columns + ["p_t1", "p_t3", "target"]]
    return result.sort_values(key_columns, kind="mergesort").reset_index(drop=True)


def blend_probability(t1_probability: Any, t3_probability: Any, t1_weight: float) -> np.ndarray:
    weight = float(t1_weight)
    if weight < 0 or weight > 1:
        raise ValueError("Ensemble weights must be convex values in [0, 1].")
    p1 = np.asarray(t1_probability, dtype="float64")
    p3 = np.asarray(t3_probability, dtype="float64")
    if p1.shape != p3.shape:
        raise ValueError("Ensemble probabilities must have equal shapes.")
    result = weight * p1 + (1.0 - weight) * p3
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("Ensemble probabilities are not finite and bounded.")
    return result


def evaluate_ensemble_weights(
    aligned: pd.DataFrame,
    *,
    weights: tuple[float, ...] = ENSEMBLE_WEIGHTS,
    bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if list(aligned.columns) != [
        "warranty_claim_key",
        "source_fold_id",
        "calibration_fold_id",
        "p_t1",
        "p_t3",
        "target",
    ]:
        raise ValueError("Aligned ensemble schema changed.")
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    y = aligned["target"].to_numpy(dtype="int8")
    keys = aligned["warranty_claim_key"].to_numpy()
    for weight in weights:
        probability = blend_probability(aligned["p_t1"], aligned["p_t3"], weight)
        metric = probability_metrics(y, probability, bins=bins, keys=keys)
        fold_metrics: list[dict[str, Any]] = []
        for fold_id, part in aligned.groupby("calibration_fold_id", sort=True):
            fold_probability = blend_probability(part["p_t1"], part["p_t3"], weight)
            fold = probability_metrics(
                part["target"],
                fold_probability,
                bins=bins,
                keys=part["warranty_claim_key"],
            )
            fold_metrics.append(
                {
                    "t1_weight": float(weight),
                    "calibration_fold_id": str(fold_id),
                    **fold,
                }
            )
        summary_rows.append(
            {
                "t1_weight": float(weight),
                "t3_weight": round(1.0 - float(weight), 1),
                **metric,
                "min_fold_average_precision": min(
                    float(item["average_precision"]) for item in fold_metrics
                ),
                "mean_fold_average_precision": float(
                    np.mean([float(item["average_precision"]) for item in fold_metrics])
                ),
            }
        )
        prediction_rows.extend(
            {
                "warranty_claim_key": int(key),
                "source_fold_id": int(source_fold),
                "calibration_fold_id": str(calibration_fold),
                "t1_weight": float(weight),
                "t3_weight": round(1.0 - float(weight), 1),
                "p_t1": float(p1),
                "p_t3": float(p3),
                "ensemble_probability": float(probability_value),
                "target": int(target),
            }
            for key, source_fold, calibration_fold, p1, p3, probability_value, target in zip(
                aligned["warranty_claim_key"],
                aligned["source_fold_id"],
                aligned["calibration_fold_id"],
                aligned["p_t1"],
                aligned["p_t3"],
                probability,
                aligned["target"],
                strict=True,
            )
        )
    return pd.DataFrame(prediction_rows), pd.DataFrame(summary_rows)


__all__ = [
    "align_selected_tracks",
    "blend_probability",
    "evaluate_ensemble_weights",
    "select_ensemble",
]
