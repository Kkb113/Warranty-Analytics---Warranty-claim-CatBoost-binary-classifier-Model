"""Deterministic raw-score threshold curves and technical policies."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from .metrics import threshold_metrics, validate_binary_inputs

THRESHOLD_COLUMNS = [
    "track",
    "strategy_id",
    "threshold",
    "tp",
    "fp",
    "tn",
    "fn",
    "precision",
    "recall",
    "specificity",
    "negative_predictive_value",
    "f1",
    "f2",
    "balanced_accuracy",
    "mcc",
    "false_positive_rate",
    "false_negative_rate",
    "predicted_positive_count",
    "predicted_positive_rate",
]


def threshold_grid(
    start: float = 0.001, stop: float = 0.999, step: float = 0.001
) -> tuple[float, ...]:
    count = int(round((stop - start) / step)) + 1
    values = tuple(float(f"{start + index * step:.3f}") for index in range(count))
    if values[0] != 0.001 or values[-1] != 0.999 or len(values) != 999:
        raise ValueError("Phase 12 threshold grid must contain 0.001 through 0.999 inclusive.")
    return values


def build_threshold_curve(
    y_true: Any,
    probabilities: Any,
    *,
    track: str,
    strategy_id: str,
    grid: Iterable[float] | None = None,
) -> pd.DataFrame:
    y, p = validate_binary_inputs(y_true, probabilities)
    rows: list[dict[str, Any]] = []
    for threshold in grid or threshold_grid():
        row: dict[str, Any] = dict(threshold_metrics(y, p, float(threshold)))
        row.update({"track": str(track), "strategy_id": str(strategy_id)})
        rows.append(row)
    return pd.DataFrame(rows).loc[:, THRESHOLD_COLUMNS]


def _better_threshold(
    candidate: dict[str, Any], incumbent: dict[str, Any], tolerance: float
) -> bool:
    for metric, sign in (("mcc", 1), ("f2", 1), ("recall", 1), ("precision", 1)):
        left = float(candidate[metric])
        right = float(incumbent[metric])
        if abs(left - right) > tolerance:
            return sign * left > sign * right
    return float(candidate["threshold"]) < float(incumbent["threshold"])


def select_mcc_threshold(curve: pd.DataFrame, tie_tolerance: float = 1.0e-12) -> dict[str, Any]:
    if curve.empty or not set(THRESHOLD_COLUMNS).issubset(curve.columns):
        raise ValueError("Threshold curve is empty or has an unexpected schema.")
    rows = curve.sort_values("threshold", kind="mergesort").to_dict("records")
    best_mcc = max(float(row["mcc"]) for row in rows)
    eligible = [row for row in rows if float(row["mcc"]) >= best_mcc - tie_tolerance]
    selected = eligible[0]
    for row in eligible[1:]:
        if _better_threshold(row, selected, tie_tolerance):
            selected = row
    return {
        "method": "MCC_MAX",
        "threshold": float(selected["threshold"]),
        "metrics": {
            key: selected[key]
            for key in THRESHOLD_COLUMNS
            if key not in {"track", "strategy_id", "threshold"}
        },
        "decision_trace": {
            "maximum_mcc": best_mcc,
            "tie_tolerance": tie_tolerance,
            "tie_break": ["higher_f2", "higher_recall", "higher_precision", "lower_threshold"],
            "tied_thresholds": [float(row["threshold"]) for row in eligible],
        },
    }


def _select_max(curve: pd.DataFrame, metric: str, tie_tolerance: float = 1.0e-12) -> dict[str, Any]:
    rows = curve.to_dict("records")
    best = max(float(row[metric]) for row in rows)
    tied = [row for row in rows if float(row[metric]) >= best - tie_tolerance]
    selected = min(tied, key=lambda row: float(row["threshold"]))
    return {
        "method": metric.upper() + "_MAX",
        "threshold": float(selected["threshold"]),
        "metrics": {
            key: selected[key]
            for key in THRESHOLD_COLUMNS
            if key not in {"track", "strategy_id", "threshold"}
        },
    }


def pareto_frontier(curve: pd.DataFrame) -> list[dict[str, Any]]:
    ordered = curve.sort_values(
        ["recall", "precision", "threshold"], ascending=[True, False, True], kind="mergesort"
    )
    result: list[dict[str, Any]] = []
    best_precision = -1.0
    for row in ordered.to_dict("records"):
        precision = float(row["precision"])
        if precision > best_precision + 1.0e-15:
            result.append(
                {
                    "threshold": float(row["threshold"]),
                    "precision": precision,
                    "recall": float(row["recall"]),
                }
            )
            best_precision = precision
    return sorted(result, key=lambda item: item["threshold"])


def threshold_summary(curve: pd.DataFrame) -> dict[str, Any]:
    technical = select_mcc_threshold(curve)
    alternatives = {
        "F1_MAX": _select_max(curve, "f1"),
        "F2_MAX": _select_max(curve, "f2"),
        "BALANCED_ACCURACY_MAX": _select_max(curve, "balanced_accuracy"),
    }
    precision_at_recall: dict[str, dict[str, Any] | None] = {}
    for required in (0.50, 0.70, 0.80, 0.90):
        eligible = curve.loc[curve["recall"] >= required]
        if eligible.empty:
            precision_at_recall[f"recall_at_least_{required:.2f}"] = None
        else:
            row = eligible.sort_values(
                ["precision", "recall", "threshold"],
                ascending=[False, False, True],
                kind="mergesort",
            ).iloc[0]
            precision_at_recall[f"recall_at_least_{required:.2f}"] = {
                "threshold": float(row["threshold"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
            }
    return {
        "technical_default": technical,
        "alternatives": alternatives,
        "precision_at_recall": precision_at_recall,
        "pareto_frontier": pareto_frontier(curve),
    }


__all__ = [
    "THRESHOLD_COLUMNS",
    "build_threshold_curve",
    "pareto_frontier",
    "select_mcc_threshold",
    "threshold_grid",
    "threshold_summary",
]
