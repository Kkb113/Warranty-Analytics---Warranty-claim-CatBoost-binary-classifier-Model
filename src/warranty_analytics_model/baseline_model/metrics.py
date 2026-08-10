"""Phase 9 validation-only metric definitions and champion selection."""

from __future__ import annotations

import functools
import hashlib
import json
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .models import BaselineModelError, ExperimentResult


def validate_probabilities(probabilities: np.ndarray) -> None:
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
        raise BaselineModelError("Phase 9 probabilities must be one-dimensional and finite.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise BaselineModelError("Phase 9 probabilities must be within [0, 1].")


def probability_sha256(probabilities: np.ndarray) -> str:
    validate_probabilities(probabilities)
    payload = json.dumps(
        [format(float(value), ".17g") for value in probabilities],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate fixed validation metrics with Average Precision as primary."""

    validate_probabilities(probabilities)
    values = set(np.asarray(y_true, dtype=int).tolist())
    if values != {0, 1}:
        raise BaselineModelError("Metric target must contain both binary classes.")
    if threshold != 0.5:
        raise BaselineModelError("Phase 9 threshold-dependent metrics must use 0.5.")
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, probabilities)
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "pr_auc_trapezoidal": float(auc(recall_curve, precision_curve)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "specificity": specificity,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def prevalence_probabilities(y_train: np.ndarray, validation_rows: int) -> np.ndarray:
    prevalence = float(np.mean(y_train))
    return np.full(validation_rows, prevalence, dtype="float64")


def apply_ap_lift(metrics: dict[str, dict[str, Any]]) -> None:
    baseline = float(metrics["E0"]["average_precision"])
    for item in metrics.values():
        item["ap_lift_over_prevalence_baseline"] = (
            float(item["average_precision"]) / baseline if baseline > 0 else None
        )


def select_champion(
    results: list[ExperimentResult],
    *,
    tolerance: float = 1.0e-6,
) -> ExperimentResult:
    """Select the validation-only development champion with fixed tie-breaks."""

    eligible = [item for item in results if item.experiment_id != "E0" and item.status == "SUCCESS"]
    if not eligible:
        raise BaselineModelError("No trained Phase 9 experiment is champion-eligible.")
    order = {"E1": 0, "E2": 1, "E3": 2, "E4": 3}

    def compare(left: ExperimentResult, right: ExperimentResult) -> int:
        for key, direction in (("average_precision", 1), ("roc_auc", 1), ("log_loss", -1)):
            delta = float(left.metrics[key]) - float(right.metrics[key])
            if abs(delta) > tolerance:
                return -1 if delta * direction > 0 else 1
        return -1 if order[left.experiment_id] < order[right.experiment_id] else 1

    return sorted(eligible, key=functools.cmp_to_key(compare))[0]


def performance_warnings(champion: ExperimentResult, baseline_metrics: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if champion.metrics["average_precision"] <= baseline_metrics["average_precision"]:
        warnings.append("WEAK_BASELINE_SIGNAL")
    if champion.metrics["roc_auc"] <= 0.5:
        warnings.append("NO_RANKING_SIGNAL")
    if champion.metrics["roc_auc"] >= 0.98 or champion.metrics["average_precision"] >= 0.90:
        warnings.append("SUSPICIOUSLY_HIGH_BASELINE_PERFORMANCE")
    return warnings
