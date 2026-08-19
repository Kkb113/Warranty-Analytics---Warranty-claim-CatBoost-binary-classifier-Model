"""Phase 15 metric helpers delegating to the canonical Phase 13/14 metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..robustness_analysis.metrics import overall_metrics


def test_metrics(y_true: Any, probabilities: Any, threshold: float) -> dict[str, Any]:
    result = overall_metrics(y_true, probabilities, threshold)
    result["phase"] = 15
    result["ap_lift_over_prevalence"] = (
        float(result["average_precision"]) / float(result["prevalence"])
        if float(result.get("prevalence", 0.0)) > 0
        else 0.0
    )
    return result


def signal_status(metrics: dict[str, Any]) -> dict[str, Any]:
    ap = float(metrics.get("average_precision", 0.0))
    prevalence = float(metrics.get("prevalence", metrics.get("observed_prevalence", 0.0)))
    roc = float(metrics.get("roc_auc", 0.0))
    status = "SIGNAL_CONFIRMED" if ap > prevalence and roc > 0.50 else "SIGNAL_COLLAPSE"
    return {
        "phase": 15,
        "status": status,
        "average_precision": ap,
        "prevalence": prevalence,
        "roc_auc": roc,
        "ap_above_prevalence": bool(ap > prevalence),
        "roc_above_random": bool(roc > 0.50),
    }


def reliability_table(
    y_true: Any, probabilities: Any, bins: int = 10
) -> tuple[pd.DataFrame, dict[str, float]]:
    y = np.asarray(y_true, dtype="int8")
    p = np.asarray(probabilities, dtype="float64")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    bucket = np.minimum(np.digitize(p, edges[1:-1], right=False), int(bins) - 1)
    rows: list[dict[str, Any]] = []
    for index in range(int(bins)):
        mask = bucket == index
        rows.append(
            {
                "bin": index + 1,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "row_count": int(mask.sum()),
                "mean_predicted_probability": float(p[mask].mean()) if mask.any() else 0.0,
                "observed_prevalence": float(y[mask].mean()) if mask.any() else 0.0,
            }
        )
    table = pd.DataFrame(rows)
    weights = table["row_count"].to_numpy(dtype="float64") / max(len(y), 1)
    gap = (table["mean_predicted_probability"] - table["observed_prevalence"]).abs().to_numpy()
    return table, {"ece_10": float(np.sum(weights * gap)), "mce_10": float(gap.max(initial=0.0))}


def validation_test_comparison(
    validation: dict[str, Any],
    test: dict[str, Any],
    *,
    moderate_ap_ratio: float,
    moderate_roc_drop: float,
) -> dict[str, Any]:
    validation_ap = float(validation.get("average_precision", 0.0))
    test_ap = float(test.get("average_precision", 0.0))
    validation_roc = float(validation.get("roc_auc", 0.0))
    test_roc = float(test.get("roc_auc", 0.0))
    ratio = test_ap / validation_ap if validation_ap else 0.0
    if test_ap <= float(test.get("prevalence", 0.0)) or test_roc <= 0.50:
        status = "SEVERE_DEGRADATION"
    elif ratio < float(moderate_ap_ratio) or test_roc < validation_roc - float(moderate_roc_drop):
        status = "MODERATE_DEGRADATION"
    else:
        status = "STABLE_GENERALIZATION"
    return {
        "validation_average_precision": validation_ap,
        "test_average_precision": test_ap,
        "ap_ratio": ratio,
        "ap_difference": test_ap - validation_ap,
        "validation_roc_auc": validation_roc,
        "test_roc_auc": test_roc,
        "roc_difference": test_roc - validation_roc,
        "log_loss_difference": float(test.get("log_loss", 0.0))
        - float(validation.get("log_loss", 0.0)),
        "brier_difference": float(test.get("brier_score", 0.0))
        - float(validation.get("brier_score", 0.0)),
        "prevalence_difference": float(test.get("prevalence", 0.0))
        - float(validation.get("prevalence", 0.0)),
        "generalization_status": status,
        "moderate_ap_ratio": float(moderate_ap_ratio),
        "moderate_roc_drop": float(moderate_roc_drop),
    }


__all__ = ["reliability_table", "signal_status", "test_metrics", "validation_test_comparison"]
