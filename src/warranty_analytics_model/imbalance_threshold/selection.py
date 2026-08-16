"""TRAIN-only strategy selection and conservative outer replacement rules."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .config import STRATEGY_IDS


def select_strategy(
    summary: pd.DataFrame,
    threshold_summaries: dict[str, dict[str, Any]],
    *,
    max_ap_tolerance: float = 0.0025,
    max_min_ap_drop: float = 0.005,
    max_roc_auc_drop: float = 0.010,
    prefer_none_mcc_tolerance: float = 0.005,
) -> dict[str, Any]:
    if summary.empty:
        raise ValueError("Cannot select a strategy from an empty summary.")
    table = summary.copy()
    required = {
        "strategy_id",
        "mean_average_precision",
        "min_average_precision",
        "std_average_precision",
        "mean_roc_auc",
    }
    if not required.issubset(table.columns):
        raise ValueError("Strategy summary lacks selection columns.")
    best_row = table.sort_values(
        ["mean_average_precision", "strategy_id"], ascending=[False, True], kind="mergesort"
    ).iloc[0]
    best_ap = float(best_row["mean_average_precision"])
    best_std = float(best_row["std_average_precision"])
    ap_tolerance = min(best_std / math.sqrt(3.0), max_ap_tolerance)
    best_min = float(best_row["min_average_precision"])
    best_roc = float(best_row["mean_roc_auc"])
    eligible: list[dict[str, Any]] = []
    for row in table.to_dict("records"):
        strategy_id = str(row["strategy_id"])
        threshold = threshold_summaries.get(strategy_id, {}).get("technical_default", {})
        metrics = threshold.get("metrics", {})
        item = {
            **row,
            "mcc": float(metrics.get("mcc", 0.0)),
            "technical_threshold": float(threshold.get("threshold", 0.5)),
            "eligible": (
                float(row["mean_average_precision"]) >= best_ap - ap_tolerance
                and float(row["min_average_precision"]) >= best_min - max_min_ap_drop
                and float(row["mean_roc_auc"]) >= best_roc - max_roc_auc_drop
            ),
        }
        if item["eligible"]:
            eligible.append(item)
    if not eligible:
        raise ValueError("No Phase 12 strategy satisfies AP/ROC guardrails.")
    ranked = sorted(
        eligible,
        key=lambda row: (
            -float(row["mcc"]),
            -float(row["mean_average_precision"]),
            -float(row["min_average_precision"]),
            float(row["std_average_precision"]),
            -float(row["mean_roc_auc"]),
            STRATEGY_IDS.index(str(row["strategy_id"])),
        ),
    )
    selected = ranked[0]
    none = next((row for row in eligible if row["strategy_id"] == "S0_NONE"), None)
    none_preferred = False
    if none is not None:
        weighted = [row for row in eligible if row["strategy_id"] != "S0_NONE"]
        if weighted:
            best_weighted_mcc = max(float(row["mcc"]) for row in weighted)
            if float(none["mcc"]) >= best_weighted_mcc - prefer_none_mcc_tolerance:
                selected = none
                none_preferred = True
    return {
        "selected_strategy_id": str(selected["strategy_id"]),
        "selected_mcc": float(selected["mcc"]),
        "selected_threshold": float(selected["technical_threshold"]),
        "best_strategy_id": str(best_row["strategy_id"]),
        "best_mean_average_precision": best_ap,
        "best_std_average_precision": best_std,
        "effective_ap_tolerance": ap_tolerance,
        "best_min_average_precision": best_min,
        "best_mean_roc_auc": best_roc,
        "eligible_strategy_ids": [str(row["strategy_id"]) for row in ranked],
        "none_preferred": none_preferred,
        "decision_rule": "MCC, mean AP, min AP, lower AP std, ROC, lower weighting complexity; prefer S0_NONE within MCC tolerance",
    }


def replacement_decision(
    parent_metrics: dict[str, Any],
    weighted_metrics: dict[str, Any],
    *,
    ap_improvement_tolerance: float = 0.000001,
    max_ap_regression_for_mcc_gain: float = 0.0005,
    max_roc_auc_regression: float = 0.005,
    required_mcc_gain: float = 0.005,
) -> dict[str, Any]:
    parent_ap = float(parent_metrics["average_precision"])
    weighted_ap = float(weighted_metrics["average_precision"])
    parent_roc = float(parent_metrics["roc_auc"])
    weighted_roc = float(weighted_metrics["roc_auc"])
    parent_mcc = float(parent_metrics.get("mcc", 0.0))
    weighted_mcc = float(weighted_metrics.get("mcc", 0.0))
    route_a = (
        weighted_ap > parent_ap + ap_improvement_tolerance
        and weighted_roc >= parent_roc - max_roc_auc_regression
    )
    route_b = (
        weighted_ap >= parent_ap - max_ap_regression_for_mcc_gain
        and weighted_roc >= parent_roc - max_roc_auc_regression
        and weighted_mcc >= parent_mcc + required_mcc_gain
    )
    return {
        "replace_parent": bool(route_a or route_b),
        "route_a_ranking_improvement": bool(route_a),
        "route_b_operating_point_improvement": bool(route_b),
        "reason": "ROUTE_A_RANKING"
        if route_a
        else ("ROUTE_B_OPERATING_POINT" if route_b else "FALLBACK_PARENT"),
        "parent_average_precision": parent_ap,
        "weighted_average_precision": weighted_ap,
        "average_precision_delta": weighted_ap - parent_ap,
        "parent_roc_auc": parent_roc,
        "weighted_roc_auc": weighted_roc,
        "parent_mcc": parent_mcc,
        "weighted_mcc": weighted_mcc,
        "mcc_delta": weighted_mcc - parent_mcc,
    }


def select_phase12_champion(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        raise ValueError("No Phase 12 effective candidates available.")
    ordered = sorted(
        candidates,
        key=lambda row: (
            -float(row["validation_metrics"]["average_precision"]),
            -float(row["validation_metrics"].get("mcc", 0.0)),
            -float(row["validation_metrics"]["roc_auc"]),
            float(row["validation_metrics"]["log_loss"]),
            int(row.get("feature_count", 0)),
            int(row.get("complexity_order", 0)),
            str(row["candidate_id"]),
        ),
    )
    return str(ordered[0]["candidate_id"])


__all__ = ["replacement_decision", "select_phase12_champion", "select_strategy"]
