"""Deterministic Phase 10 trial, replacement, and champion selection."""

from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import Any

import pandas as pd

from .models import OptimizationError

TOLERANCE = 1.0e-6


def _numeric_delta(left: Any, right: Any) -> float:
    return float(left) - float(right)


def compare_trial_rows(
    left: dict[str, Any], right: dict[str, Any], tolerance: float = TOLERANCE
) -> int:
    """Return -1 when left ranks ahead of right."""

    for key, direction in (
        ("mean_average_precision", 1),
        ("min_average_precision", 1),
        ("mean_roc_auc", 1),
        ("std_average_precision", -1),
        ("mean_log_loss", -1),
    ):
        delta = _numeric_delta(left[key], right[key])
        if abs(delta) > tolerance:
            return -1 if delta * direction > 0 else 1
    left_depth = int(left.get("depth", 99))
    right_depth = int(right.get("depth", 99))
    if left_depth != right_depth:
        return -1 if left_depth < right_depth else 1
    left_iterations = int(left.get("iterations", 10**9))
    right_iterations = int(right.get("iterations", 10**9))
    if left_iterations != right_iterations:
        return -1 if left_iterations < right_iterations else 1
    return -1 if int(left.get("trial_number", 10**9)) < int(right.get("trial_number", 10**9)) else 1


def select_best_trial(history: pd.DataFrame) -> dict[str, Any]:
    completed = history.loc[history["state"].astype(str) == "COMPLETE"].copy()
    if completed.empty:
        raise OptimizationError("No completed Phase 10 trials are available for selection.")
    rows = completed.to_dict(orient="records")
    selected = sorted(rows, key=functools.cmp_to_key(compare_trial_rows))[0]
    return {str(key): value for key, value in selected.items()}


def replacement_decision(baseline: dict[str, Any], optimized: dict[str, Any]) -> dict[str, Any]:
    baseline_ap = float(baseline["average_precision"])
    optimized_ap = float(optimized["average_precision"])
    beats = optimized_ap > baseline_ap + TOLERANCE
    return {
        "optimized_beats_baseline": beats,
        "fallback_to_baseline": not beats,
        "average_precision_delta": optimized_ap - baseline_ap,
        "tolerance": TOLERANCE,
    }


def _compare_candidate(left: dict[str, Any], right: dict[str, Any]) -> int:
    for key, direction in (("average_precision", 1), ("roc_auc", 1), ("log_loss", -1)):
        delta = float(left["metrics"][key]) - float(right["metrics"][key])
        if abs(delta) > TOLERANCE:
            return -1 if delta * direction > 0 else 1
    left_count = int(left["feature_count"])
    right_count = int(right["feature_count"])
    if left_count != right_count:
        return -1 if left_count < right_count else 1
    left_baseline = str(left["candidate_id"]).endswith("BASELINE")
    right_baseline = str(right["candidate_id"]).endswith("BASELINE")
    if left_baseline != right_baseline:
        return -1 if left_baseline else 1
    return -1 if str(left["candidate_id"]) < str(right["candidate_id"]) else 1


def select_development_champion(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    eligible = list(candidates)
    if not eligible:
        raise OptimizationError("No Phase 10 champion candidates are available.")
    return sorted(eligible, key=functools.cmp_to_key(_compare_candidate))[0]
