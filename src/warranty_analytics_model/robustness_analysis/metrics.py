"""Canonical Phase 14 metric and support-policy helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..calibration_ensemble.calibration_metrics import probability_metrics
from ..imbalance_threshold.metrics import threshold_metrics


def overall_metrics(
    y_true: Any,
    probabilities: Any,
    threshold: float,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    """Return ranking, probability, prevalence, and frozen-threshold metrics."""

    y = np.asarray(y_true, dtype="int8").reshape(-1)
    p = np.asarray(probabilities, dtype="float64").reshape(-1)
    result: dict[str, Any] = dict(probability_metrics(y, p, bins=bins))
    result.update(threshold_metrics(y, p, float(threshold)))
    prevalence = float(np.mean(y)) if len(y) else 0.0
    result["prevalence"] = prevalence
    result["ap_lift_over_prevalence"] = (
        float(result["average_precision"] / prevalence) if prevalence > 0 else 0.0
    )
    result["primary_signal_pass"] = bool(
        float(result["average_precision"]) > prevalence + 1.0e-6 and float(result["roc_auc"]) > 0.50
    )
    return result


def support_status(
    row_count: int,
    positive_count: int,
    negative_count: int,
    *,
    min_rows: int = 75,
    min_positive_ranking: int = 5,
    min_negative_ranking: int = 20,
) -> str:
    if (
        int(row_count) < min_rows
        or int(positive_count) < min_positive_ranking
        or int(negative_count) < min_negative_ranking
    ):
        return "LOW_SUPPORT"
    return "SUPPORTED"


def safe_metric_dict(
    y_true: Any, probabilities: Any, threshold: float, *, bins: int = 10
) -> dict[str, Any]:
    """Compute descriptive metrics even when a slice has one class."""

    y = np.asarray(y_true, dtype="int8").reshape(-1)
    p = np.asarray(probabilities, dtype="float64").reshape(-1)
    if len(y) == 0:
        return {
            "row_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "prevalence": 0.0,
            "status": "LOW_SUPPORT",
        }
    try:
        result = overall_metrics(y, p, threshold, bins=bins)
    except (ValueError, FloatingPointError):
        # Ranking metrics are undefined without both classes; confusion metrics
        # remain useful and are reconstructed with a safe probability metric set.
        predicted = p >= float(threshold)
        positive = y == 1
        negative = ~positive
        tp = int(np.count_nonzero(predicted & positive))
        fp = int(np.count_nonzero(predicted & negative))
        tn = int(np.count_nonzero(~predicted & negative))
        fn = int(np.count_nonzero(~predicted & positive))
        result = {
            "row_count": int(len(y)),
            "positive_count": int(positive.sum()),
            "negative_count": int(negative.sum()),
            "prevalence": float(positive.mean()),
            "average_precision": None,
            "roc_auc": None,
            "log_loss": None,
            "brier_score": float(np.mean((p - y) ** 2)),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
    result["status"] = support_status(
        int(result.get("row_count", len(y))),
        int(result.get("positive_count", int((y == 1).sum()))),
        int(result.get("negative_count", int((y == 0).sum()))),
    )
    return result


__all__ = ["overall_metrics", "safe_metric_dict", "support_status"]
