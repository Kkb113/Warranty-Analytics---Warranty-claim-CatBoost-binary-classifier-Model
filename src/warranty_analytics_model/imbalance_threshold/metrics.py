"""Finite imbalance-aware ranking and operating-point metrics."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def validate_binary_inputs(y_true: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype="int8").reshape(-1)
    p = np.asarray(probabilities, dtype="float64").reshape(-1)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("Targets and probabilities must be non-empty and equally sized.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Targets must be binary.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    return y, p


def ranking_metrics(y_true: Any, probabilities: Any) -> dict[str, float]:
    y, p = validate_binary_inputs(y_true, probabilities)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ap = float(average_precision_score(y, p))
        roc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else 0.0
        loss = float(log_loss(y, p, labels=[0, 1]))
    brier = float(np.mean((p - y) ** 2))
    return {
        "average_precision": ap if np.isfinite(ap) else 0.0,
        "roc_auc": roc if np.isfinite(roc) else 0.0,
        "log_loss": loss if np.isfinite(loss) else float("inf"),
        "brier_score": brier if np.isfinite(brier) else float("inf"),
    }


def threshold_metrics(y_true: Any, probabilities: Any, threshold: float) -> dict[str, float | int]:
    y, p = validate_binary_inputs(y_true, probabilities)
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value) or not 0 < threshold_value < 1:
        raise ValueError("Threshold must be finite and strictly between zero and one.")
    predicted = p >= threshold_value
    positive = y == 1
    negative = ~positive
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & negative))
    tn = int(np.count_nonzero(~predicted & negative))
    fn = int(np.count_nonzero(~predicted & positive))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    npv = _safe_divide(tn, tn + fn)
    fpr = _safe_divide(fp, fp + tn)
    fnr = _safe_divide(fn, fn + tp)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    f2 = _safe_divide(5 * precision * recall, 4 * precision + recall)
    balanced = (recall + specificity) / 2.0
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = _safe_divide(tp * tn - fp * fn, denominator)
    return {
        "threshold": threshold_value,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "negative_predictive_value": npv,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "f1": f1,
        "f2": f2,
        "balanced_accuracy": balanced,
        "mcc": mcc,
        "predicted_positive_count": int(predicted.sum()),
        "predicted_positive_rate": float(predicted.mean()),
    }


def fold_metric_row(
    y_true: Any,
    probabilities: Any,
    *,
    track: str,
    strategy_id: str,
    fold_id: int,
    train_positive_count: int,
    train_negative_count: int,
    training_seconds: float,
    weighting_parameters: dict[str, Any],
) -> dict[str, Any]:
    y, p = validate_binary_inputs(y_true, probabilities)
    metrics: dict[str, Any] = ranking_metrics(y, p)
    metrics.update(
        {
            "track": track,
            "strategy_id": strategy_id,
            "fold_id": int(fold_id),
            "train_positive_count": int(train_positive_count),
            "train_negative_count": int(train_negative_count),
            "validation_positive_count": int(y.sum()),
            "validation_negative_count": int((y == 0).sum()),
            "weighting_parameters": weighting_parameters,
            "prediction_minimum": float(p.min()),
            "prediction_maximum": float(p.max()),
            "prediction_mean": float(p.mean()),
            "prediction_median": float(np.median(p)),
            "training_seconds": float(training_seconds),
        }
    )
    return metrics


def aggregate_strategy_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 3:
        raise ValueError("Each Phase 12 strategy must have exactly three fold rows.")
    ap = np.asarray([float(row["average_precision"]) for row in rows], dtype="float64")
    roc = np.asarray([float(row["roc_auc"]) for row in rows], dtype="float64")
    losses = np.asarray([float(row["log_loss"]) for row in rows], dtype="float64")
    brier = np.asarray([float(row["brier_score"]) for row in rows], dtype="float64")
    return {
        "track": str(rows[0]["track"]),
        "strategy_id": str(rows[0]["strategy_id"]),
        "mean_average_precision": float(ap.mean()),
        "min_average_precision": float(ap.min()),
        "max_average_precision": float(ap.max()),
        "std_average_precision": float(ap.std(ddof=0)),
        "mean_roc_auc": float(roc.mean()),
        "min_roc_auc": float(roc.min()),
        "mean_log_loss": float(losses.mean()),
        "mean_brier_score": float(brier.mean()),
        "fold_count": 3,
        "total_training_seconds": float(sum(float(row["training_seconds"]) for row in rows)),
    }


def strategy_fold_metrics_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "track",
        "strategy_id",
        "fold_id",
        "average_precision",
        "roc_auc",
        "log_loss",
        "brier_score",
        "train_positive_count",
        "train_negative_count",
        "validation_positive_count",
        "validation_negative_count",
        "weighting_parameters",
        "prediction_minimum",
        "prediction_maximum",
        "prediction_mean",
        "prediction_median",
        "training_seconds",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame:
            frame[column] = None
    return (
        frame.loc[:, columns]
        .sort_values(["track", "strategy_id", "fold_id"], kind="mergesort")
        .reset_index(drop=True)
    )


__all__ = [
    "aggregate_strategy_metrics",
    "fold_metric_row",
    "ranking_metrics",
    "strategy_fold_metrics_frame",
    "threshold_metrics",
    "validate_binary_inputs",
]
