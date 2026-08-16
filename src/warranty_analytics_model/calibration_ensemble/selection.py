"""Tolerance-aware calibration, ensemble, and champion selection."""

from __future__ import annotations

from functools import cmp_to_key
from typing import Any

import pandas as pd

from .config import (
    CALIBRATION_COMPLEXITY,
    CalibrationEnsembleSettings,
    load_calibration_ensemble_settings,
)

TOLERANCE = 1.0e-6
_CALIBRATION_METHOD_ALIASES = {
    "NONE": "NONE",
    "C0_NONE": "NONE",
    "SIGMOID": "SIGMOID",
    "C1_SIGMOID": "SIGMOID",
    "ISOTONIC": "ISOTONIC",
    "C2_ISOTONIC": "ISOTONIC",
}


def _compare(left: float, right: float, *, higher: bool, tolerance: float = TOLERANCE) -> int:
    if abs(float(left) - float(right)) <= tolerance:
        return 0
    if higher:
        return -1 if left > right else 1
    return -1 if left < right else 1


def compare_champion_candidates(
    left: dict[str, Any], right: dict[str, Any], *, tolerance: float = TOLERANCE
) -> int:
    left_metrics = left.get("validation_metrics", left.get("metrics", {}))
    right_metrics = right.get("validation_metrics", right.get("metrics", {}))
    for metric, higher in (
        ("average_precision", True),
        ("log_loss", False),
        ("brier_score", False),
        ("roc_auc", True),
        ("mcc", True),
    ):
        result = _compare(
            float(left_metrics.get(metric, 0.0)),
            float(right_metrics.get(metric, 0.0)),
            higher=higher,
            tolerance=tolerance,
        )
        if result:
            return result
    for key in ("complexity_order", "feature_count"):
        left_value = int(left.get(key, 0))
        right_value = int(right.get(key, 0))
        if left_value != right_value:
            return -1 if left_value < right_value else 1
    return (
        -1
        if str(left["candidate_id"]) < str(right["candidate_id"])
        else (1 if str(left["candidate_id"]) > str(right["candidate_id"]) else 0)
    )


def select_phase13_champion(
    candidates: list[dict[str, Any]], settings: CalibrationEnsembleSettings | None = None
) -> str:
    if not candidates:
        raise ValueError("No Phase 13 candidates are available.")
    locked = settings or load_calibration_ensemble_settings()

    def comparator(left: dict[str, Any], right: dict[str, Any]) -> int:
        return compare_champion_candidates(left, right, tolerance=locked.selection_tie_tolerance)

    return str(sorted(candidates, key=cmp_to_key(comparator))[0]["candidate_id"])


def select_best_single_candidate(
    candidates: list[dict[str, Any]], settings: CalibrationEnsembleSettings | None = None
) -> dict[str, Any]:
    """Select the best effective single using frozen Phase 13 semantics."""

    if not candidates:
        raise ValueError("No effective single candidates are available.")
    locked = settings or load_calibration_ensemble_settings()

    def comparator(left: dict[str, Any], right: dict[str, Any]) -> int:
        return compare_champion_candidates(left, right, tolerance=locked.selection_tie_tolerance)

    return sorted(candidates, key=cmp_to_key(comparator))[0]


def select_calibration_method(
    summary: pd.DataFrame, settings: CalibrationEnsembleSettings | None = None
) -> dict[str, Any]:
    if summary.empty:
        raise ValueError("Calibration summary is empty.")
    locked = settings or load_calibration_ensemble_settings()
    none = summary.loc[summary["calibration_method"] == "C0_NONE"]
    if len(none) != 1:
        raise ValueError("Calibration summary must contain exactly one NONE baseline.")
    baseline = none.iloc[0].to_dict()
    candidates: list[dict[str, Any]] = []
    for row in summary.to_dict("records"):
        method = str(row["calibration_method"])
        eligible = bool(row.get("eligible", True))
        if method != "C0_NONE":
            eligible = eligible and (
                float(row["pooled_average_precision"])
                >= float(baseline["pooled_average_precision"]) - locked.selection_max_ap_drop
                and float(row["min_fold_average_precision"])
                >= float(baseline["min_fold_average_precision"])
                - locked.selection_max_min_fold_ap_drop
                and float(row["pooled_roc_auc"])
                >= float(baseline["pooled_roc_auc"]) - locked.selection_max_roc_auc_drop
            )
        item = {**row, "eligible": eligible}
        if eligible:
            candidates.append(item)
    if not candidates:
        raise ValueError("No eligible calibration method remains.")
    ranked = sorted(
        candidates,
        key=lambda row: (
            float(row["pooled_log_loss"]),
            float(row["pooled_brier_score"]),
            float(row["pooled_ece"]),
            -float(row["pooled_average_precision"]),
            -float(row["min_fold_average_precision"]),
            CALIBRATION_COMPLEXITY.get(str(row["calibration_method"]), 99),
        ),
    )
    winner = ranked[0]
    none_preferred = False
    if winner["calibration_method"] != "C0_NONE":
        log_loss_gain = float(baseline["pooled_log_loss"]) - float(winner["pooled_log_loss"])
        brier_gain = float(baseline["pooled_brier_score"]) - float(winner["pooled_brier_score"])
        if (
            log_loss_gain < locked.none_log_loss_tolerance
            and brier_gain < locked.none_brier_tolerance
        ):
            winner = baseline
            none_preferred = True
    return {
        "selected_calibration_method": str(winner["calibration_method"]),
        "eligible_methods": [str(row["calibration_method"]) for row in ranked],
        "none_preferred_for_trivial_gain": none_preferred,
        "baseline_none": baseline,
        "ranked_candidates": ranked,
        "decision_rule": "log-loss, Brier, ECE, AP, minimum-fold AP, calibration complexity",
    }


def _ensemble_cmp(
    left: dict[str, Any], right: dict[str, Any], *, tolerance: float = TOLERANCE
) -> int:
    for metric, higher in (
        ("pooled_average_precision", True),
        ("min_fold_average_precision", True),
        ("pooled_roc_auc", True),
        ("pooled_log_loss", False),
        ("pooled_brier_score", False),
    ):
        result = _compare(
            float(left[metric]), float(right[metric]), higher=higher, tolerance=tolerance
        )
        if result:
            return result
    left_weight = float(left["t1_weight"])
    right_weight = float(right["t1_weight"])
    left_endpoint = left_weight in (0.0, 1.0)
    right_endpoint = right_weight in (0.0, 1.0)
    if left_endpoint != right_endpoint:
        return -1 if left_endpoint else 1
    if not left_endpoint and abs(left_weight - 0.5) != abs(right_weight - 0.5):
        return -1 if abs(left_weight - 0.5) < abs(right_weight - 0.5) else 1
    return -1 if left_weight < right_weight else (1 if left_weight > right_weight else 0)


def select_ensemble(
    summary: pd.DataFrame, settings: CalibrationEnsembleSettings | None = None
) -> dict[str, Any]:
    if summary.empty or set(float(value) for value in summary["t1_weight"]) != {
        index / 10.0 for index in range(11)
    }:
        raise ValueError("Ensemble summary must contain exactly 11 controlled weights.")
    locked = settings or load_calibration_ensemble_settings()
    rows = summary.to_dict("records")
    endpoints = [row for row in rows if float(row["t1_weight"]) in (0.0, 1.0)]

    def comparator(left: dict[str, Any], right: dict[str, Any]) -> int:
        return _ensemble_cmp(left, right, tolerance=locked.selection_tie_tolerance)

    best_single = sorted(endpoints, key=cmp_to_key(comparator))[0]
    blends = [row for row in rows if 0.0 < float(row["t1_weight"]) < 1.0]
    accepted: list[dict[str, Any]] = []
    for row in blends:
        route_a = (
            float(row["pooled_average_precision"])
            > float(best_single["pooled_average_precision"]) + locked.ranking_minimum_ap_improvement
            and float(row["min_fold_average_precision"])
            >= float(best_single["min_fold_average_precision"])
            - locked.ranking_max_min_fold_ap_drop
            and float(row["pooled_roc_auc"])
            >= float(best_single["pooled_roc_auc"]) - locked.ranking_max_roc_auc_drop
        )
        route_b = (
            float(row["pooled_average_precision"])
            >= float(best_single["pooled_average_precision"]) - locked.calibration_route_max_ap_drop
            and float(row["min_fold_average_precision"])
            >= float(best_single["min_fold_average_precision"])
            - locked.calibration_route_max_min_fold_ap_drop
            and float(row["pooled_roc_auc"])
            >= float(best_single["pooled_roc_auc"]) - locked.calibration_route_max_roc_auc_drop
            and float(row["pooled_log_loss"])
            <= float(best_single["pooled_log_loss"])
            - locked.calibration_route_min_log_loss_improvement
            and float(row["pooled_brier_score"])
            <= float(best_single["pooled_brier_score"])
            - locked.calibration_route_min_brier_improvement
        )
        if route_a or route_b:
            accepted.append({**row, "route_a": route_a, "route_b": route_b})
    if not accepted:
        return {
            "selected_policy": "BEST_SINGLE",
            "selected_weight": float(best_single["t1_weight"]),
            "best_single": best_single,
            "accepted_blends": [],
            "selection_route": "NONE",
            "decision_trace": {"best_single": best_single, "accepted_blends": []},
        }
    winner = sorted(accepted, key=cmp_to_key(comparator))[0]
    return {
        "selected_policy": "TRUE_BLEND",
        "selected_weight": float(winner["t1_weight"]),
        "best_single": best_single,
        "accepted_blends": accepted,
        "selection_route": "ROUTE_A_RANKING" if winner["route_a"] else "ROUTE_B_CALIBRATION",
        "decision_trace": {"best_single": best_single, "accepted_blends": accepted},
    }


def accept_track_calibration(
    raw: dict[str, Any],
    calibrated: dict[str, Any],
    settings: Any,
    *,
    calibration_method: str,
) -> dict[str, Any]:
    try:
        method = _CALIBRATION_METHOD_ALIASES[str(calibration_method).upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported Phase 13 calibration method: {calibration_method}") from exc
    if method == "NONE":
        return {"accepted": False, "reason": "NONE", "effective": "RAW_PHASE12"}
    guardrails = (
        float(calibrated["average_precision"])
        >= float(raw["average_precision"]) - settings.validation_calibration_max_ap_drop
        and float(calibrated["roc_auc"])
        >= float(raw["roc_auc"]) - settings.validation_calibration_max_roc_auc_drop
        and float(calibrated["log_loss"])
        <= float(raw["log_loss"]) + settings.validation_calibration_max_log_loss_regression
        and float(calibrated["brier_score"])
        <= float(raw["brier_score"]) + settings.validation_calibration_max_brier_regression
    )
    material = (
        float(calibrated["log_loss"])
        <= float(raw["log_loss"]) - settings.validation_calibration_min_log_loss_improvement
        or float(calibrated["brier_score"])
        <= float(raw["brier_score"]) - settings.validation_calibration_min_brier_improvement
    )
    accepted = bool(guardrails and material)
    return {
        "accepted": accepted,
        "reason": "ACCEPTED" if accepted else "CALIBRATION_REJECTED_ON_VALIDATION",
        "guardrails": guardrails,
        "material_probability_gain": material,
        "effective": "CALIBRATED" if accepted else "RAW_PHASE12",
    }


def accept_ensemble(
    ensemble: dict[str, Any], best_single: dict[str, Any], settings: Any
) -> dict[str, Any]:
    route_a = (
        float(ensemble["average_precision"])
        > float(best_single["average_precision"])
        + settings.validation_ensemble_ap_improvement_tolerance
        and float(ensemble["roc_auc"])
        >= float(best_single["roc_auc"]) - settings.validation_ensemble_max_roc_auc_drop
        and float(ensemble["log_loss"])
        <= float(best_single["log_loss"]) + settings.validation_ensemble_max_log_loss_regression
    )
    route_b = (
        float(ensemble["average_precision"])
        >= float(best_single["average_precision"])
        - settings.validation_ensemble_max_ap_drop_for_calibration_route
        and float(ensemble["roc_auc"])
        >= float(best_single["roc_auc"]) - settings.validation_ensemble_max_roc_auc_drop
        and float(ensemble["log_loss"])
        <= float(best_single["log_loss"]) - settings.validation_ensemble_min_log_loss_improvement
        and float(ensemble["brier_score"])
        <= float(best_single["brier_score"]) - settings.validation_ensemble_min_brier_improvement
    )
    accepted = bool(route_a or route_b)
    return {
        "accepted": accepted,
        "route_a": route_a,
        "route_b": route_b,
        "reason": "ACCEPTED" if accepted else "ENSEMBLE_REJECTED_ON_VALIDATION",
    }


__all__ = [
    "TOLERANCE",
    "accept_ensemble",
    "accept_track_calibration",
    "compare_champion_candidates",
    "select_calibration_method",
    "select_best_single_candidate",
    "select_ensemble",
    "select_phase13_champion",
]
