"""Probability and calibration metrics for Phase 13."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from .reliability import ece_mce


def validate_probability_inputs(y_true: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype="int8").reshape(-1)
    p = np.asarray(probabilities, dtype="float64").reshape(-1)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Targets and probabilities must be non-empty and equally sized.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Targets must be binary.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    return y, p


def probability_metrics(
    y_true: Any,
    probabilities: Any,
    *,
    bins: int = 10,
    keys: Any | None = None,
) -> dict[str, float | int]:
    y, p = validate_probability_inputs(y_true, probabilities)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ap = float(average_precision_score(y, p))
        roc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else 0.0
        loss = float(log_loss(y, p, labels=[0, 1]))
    ece, mce = ece_mce(y, p, bins=bins, keys=keys)
    return {
        "average_precision": ap if np.isfinite(ap) else 0.0,
        "roc_auc": roc if np.isfinite(roc) else 0.0,
        "log_loss": loss if np.isfinite(loss) else float("inf"),
        "brier_score": float(np.mean((p - y) ** 2)),
        "ece_10": ece,
        "mce_10": mce,
        "row_count": int(len(y)),
        "positive_count": int(np.count_nonzero(y == 1)),
        "negative_count": int(np.count_nonzero(y == 0)),
        "mean_predicted_probability": float(np.mean(p)),
        "observed_prevalence": float(np.mean(y)),
    }


def add_probability_metrics(
    frame: dict[str, Any],
    y_true: Any,
    probabilities: Any,
    *,
    bins: int = 10,
    keys: Any | None = None,
) -> dict[str, Any]:
    """Return a metadata row with the canonical metric keys."""

    return {**frame, **probability_metrics(y_true, probabilities, bins=bins, keys=keys)}


__all__ = ["add_probability_metrics", "probability_metrics", "validate_probability_inputs"]
