"""Phase 10 inner and outer metric helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..baseline_model.metrics import calculate_metrics, validate_probabilities
from .models import OptimizationError


def aggregate_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate exactly the metrics used by the deterministic trial ranker."""

    if not fold_metrics:
        raise OptimizationError("Cannot aggregate an empty inner-fold metric set.")
    ap = np.asarray([float(item["average_precision"]) for item in fold_metrics], dtype="float64")
    roc = np.asarray([float(item["roc_auc"]) for item in fold_metrics], dtype="float64")
    losses = np.asarray([float(item["log_loss"]) for item in fold_metrics], dtype="float64")
    brier = np.asarray([float(item["brier_score"]) for item in fold_metrics], dtype="float64")
    return {
        "mean_average_precision": float(ap.mean()),
        "min_average_precision": float(ap.min()),
        "max_average_precision": float(ap.max()),
        "std_average_precision": float(ap.std(ddof=0)),
        "mean_roc_auc": float(roc.mean()),
        "min_roc_auc": float(roc.min()),
        "mean_log_loss": float(losses.mean()),
        "mean_brier_score": float(brier.mean()),
        "fold_count": len(fold_metrics),
    }


def metrics_for_predictions(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    validate_probabilities(probabilities)
    if threshold != 0.5:
        raise OptimizationError("Phase 10 threshold must remain fixed at 0.5.")
    return calculate_metrics(y_true, probabilities, threshold=threshold)


def compare_metrics(
    baseline: dict[str, Any], optimized: dict[str, Any], *, tolerance: float = 1.0e-6
) -> dict[str, Any]:
    """Compare one optimized finalist with its corresponding Phase 9 baseline."""

    baseline_ap = float(baseline["average_precision"])
    optimized_ap = float(optimized["average_precision"])
    return {
        "baseline_average_precision": baseline_ap,
        "optimized_average_precision": optimized_ap,
        "average_precision_delta": optimized_ap - baseline_ap,
        "average_precision_relative_lift": (
            (optimized_ap - baseline_ap) / baseline_ap if baseline_ap else None
        ),
        "roc_auc_delta": float(optimized["roc_auc"]) - float(baseline["roc_auc"]),
        "log_loss_delta": float(optimized["log_loss"]) - float(baseline["log_loss"]),
        "brier_score_delta": float(optimized["brier_score"]) - float(baseline["brier_score"]),
        "optimized_beats_baseline": optimized_ap > baseline_ap + tolerance,
        "fallback_to_baseline": optimized_ap <= baseline_ap + tolerance,
    }


def metrics_payload(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {str(key): value for key, value in metrics.items()}


def validate_prediction_frame(frame: pd.DataFrame, expected_candidates: set[str]) -> None:
    required = ["warranty_claim_key", "candidate_id", "high_cost_probability"]
    if list(frame.columns) != required:
        raise OptimizationError("Phase 10 validation predictions have an unexpected schema.")
    if set(frame["candidate_id"].astype(str)) != expected_candidates:
        raise OptimizationError("Phase 10 validation predictions contain unexpected candidates.")
    if frame.duplicated(["warranty_claim_key", "candidate_id"]).any():
        raise OptimizationError(
            "Phase 10 validation predictions contain duplicate claim/candidate rows."
        )
    probabilities = pd.to_numeric(frame["high_cost_probability"], errors="coerce").to_numpy(
        dtype="float64"
    )
    validate_probabilities(probabilities)
