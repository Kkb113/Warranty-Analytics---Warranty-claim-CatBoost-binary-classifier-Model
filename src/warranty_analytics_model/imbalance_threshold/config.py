"""Fail-closed Phase 12 experimental configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..paths import discover_repository_root

PHASE12_VERSION = "phase12_imbalance_threshold_v1"
TRACKS = ("T1", "T3")
STRATEGY_IDS = (
    "S0_NONE",
    "S1_SCALE_POS_WEIGHT_2",
    "S2_SCALE_POS_WEIGHT_4",
    "S3_SCALE_POS_WEIGHT_8",
    "S4_SCALE_POS_WEIGHT_16",
    "S5_SCALE_POS_WEIGHT_32",
    "S6_AUTO_SQRT_BALANCED",
    "S7_AUTO_BALANCED",
)
STRATEGY_TYPES = (
    "none",
    "scale_pos_weight",
    "scale_pos_weight",
    "scale_pos_weight",
    "scale_pos_weight",
    "scale_pos_weight",
    "auto_class_weights",
    "auto_class_weights",
)
STRATEGY_VALUES: tuple[float | str | None, ...] = (
    None,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    "SqrtBalanced",
    "Balanced",
)


class ImbalanceThresholdError(ValueError):
    """Raised when Phase 12 policy or configuration validation fails."""


@dataclass(frozen=True, slots=True)
class ImbalanceThresholdSettings:
    tracks: tuple[str, ...]
    primary_ranking_metric: str
    strategy_ids: tuple[str, ...]
    strategy_types: tuple[str, ...]
    strategy_values: tuple[float | str | None, ...]
    max_ap_tolerance: float
    max_min_ap_drop: float
    max_roc_auc_drop: float
    prefer_none_mcc_tolerance: float
    threshold_start: float
    threshold_stop: float
    threshold_step: float
    threshold_tie_tolerance: float
    threshold_tie_break: tuple[str, ...]
    ap_improvement_tolerance: float
    max_ap_regression_for_mcc_gain: float
    max_roc_auc_regression: float
    required_mcc_gain: float
    preferred_search_workers: int
    preferred_threads_per_search_fit: int
    preferred_single_fit_threads: int
    reserve_logical_threads: int
    high_performance_local: bool
    checkpoint_each_fold: bool
    resume_supported: bool
    output_directory: str
    report_directory: str

    @property
    def strategy_count(self) -> int:
        return len(self.strategy_ids)


def _number(raw: Any, label: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ImbalanceThresholdError(f"Phase 12 {label} must be numeric.")
    value = float(raw)
    if not math.isfinite(value):
        raise ImbalanceThresholdError(f"Phase 12 {label} must be finite.")
    return value


def _positive_int(raw: Any, label: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ImbalanceThresholdError(f"Phase 12 {label} must be a positive integer.")
    return int(raw)


def load_imbalance_threshold_settings(
    project_root: Path | None = None,
) -> ImbalanceThresholdSettings:
    root = discover_repository_root(project_root)
    path = root / "configs" / "imbalance_threshold.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ImbalanceThresholdError(f"Could not read Phase 12 configuration: {path}") from exc
    raw = payload.get("phase12_imbalance_threshold") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        raise ImbalanceThresholdError(
            "Phase 12 configuration must contain phase12_imbalance_threshold."
        )
    tracks = tuple(str(item) for item in raw.get("tracks", []))
    if tracks != TRACKS:
        raise ImbalanceThresholdError(f"Phase 12 tracks must be exactly {TRACKS}; got {tracks}.")
    if raw.get("primary_ranking_metric") != "mean_average_precision":
        raise ImbalanceThresholdError("Phase 12 primary ranking metric drifted.")
    strategies = raw.get("strategies")
    if not isinstance(strategies, list) or len(strategies) != len(STRATEGY_IDS):
        raise ImbalanceThresholdError("Phase 12 must declare exactly eight strategies.")
    ids: list[str] = []
    types: list[str] = []
    values: list[float | str | None] = []
    for index, item in enumerate(strategies):
        if not isinstance(item, dict):
            raise ImbalanceThresholdError("Each Phase 12 strategy must be a mapping.")
        expected_id = STRATEGY_IDS[index]
        expected_type = STRATEGY_TYPES[index]
        expected_value = STRATEGY_VALUES[index]
        if item.get("id") != expected_id or item.get("type") != expected_type:
            raise ImbalanceThresholdError("Phase 12 strategy order or type drifted.")
        value = item.get("value")
        if expected_value is None:
            if "value" in item and value is not None:
                raise ImbalanceThresholdError("S0_NONE must not define a weighting value.")
            value = None
        elif isinstance(expected_value, float):
            if _number(value, f"{expected_id}.value") != expected_value:
                raise ImbalanceThresholdError(f"{expected_id} value drifted.")
            value = expected_value
        elif value != expected_value:
            raise ImbalanceThresholdError(f"{expected_id} value drifted.")
        ids.append(expected_id)
        types.append(expected_type)
        values.append(value)
    selection = raw.get("strategy_selection")
    threshold = raw.get("threshold")
    replacement = raw.get("outer_replacement")
    compute = raw.get("compute")
    if not all(isinstance(item, dict) for item in (selection, threshold, replacement, compute)):
        raise ImbalanceThresholdError(
            "Phase 12 selection, threshold, replacement, or compute policy is missing."
        )
    assert isinstance(selection, dict)
    assert isinstance(threshold, dict)
    assert isinstance(replacement, dict)
    assert isinstance(compute, dict)
    if selection != {
        "max_ap_tolerance": 0.0025,
        "max_min_ap_drop": 0.005,
        "max_roc_auc_drop": 0.010,
        "prefer_none_mcc_tolerance": 0.005,
    }:
        raise ImbalanceThresholdError("Phase 12 strategy-selection policy drifted.")
    expected_threshold = {
        "primary_metric": "matthews_correlation_coefficient",
        "grid": {"start": 0.001, "stop": 0.999, "step": 0.001},
        "tie_tolerance": 1.0e-12,
        "tie_break": ["higher_f2", "higher_recall", "higher_precision", "lower_threshold"],
    }
    if threshold != expected_threshold:
        raise ImbalanceThresholdError("Phase 12 threshold policy drifted.")
    expected_replacement = {
        "ap_improvement_tolerance": 0.000001,
        "max_ap_regression_for_mcc_gain": 0.0005,
        "max_roc_auc_regression": 0.005,
        "required_mcc_gain": 0.005,
    }
    if replacement != expected_replacement:
        raise ImbalanceThresholdError("Phase 12 replacement policy drifted.")
    expected_compute = {
        "high_performance_local": True,
        "preferred_search_workers": 4,
        "preferred_threads_per_search_fit": 5,
        "preferred_single_fit_threads": 16,
        "reserve_logical_threads": 2,
    }
    if compute != expected_compute:
        raise ImbalanceThresholdError("Phase 12 compute policy drifted.")
    if raw.get("checkpoint_each_fold") is not True or raw.get("resume_supported") is not True:
        raise ImbalanceThresholdError("Phase 12 checkpoint/resume policy must remain enabled.")
    if raw.get("output_directory") != "artifacts/imbalance_threshold":
        raise ImbalanceThresholdError("Phase 12 output directory drifted.")
    if raw.get("report_directory") != "reports/phase12_imbalance_threshold":
        raise ImbalanceThresholdError("Phase 12 report directory drifted.")
    return ImbalanceThresholdSettings(
        tracks=tracks,
        primary_ranking_metric="mean_average_precision",
        strategy_ids=tuple(ids),
        strategy_types=tuple(types),
        strategy_values=tuple(values),
        max_ap_tolerance=0.0025,
        max_min_ap_drop=0.005,
        max_roc_auc_drop=0.010,
        prefer_none_mcc_tolerance=0.005,
        threshold_start=0.001,
        threshold_stop=0.999,
        threshold_step=0.001,
        threshold_tie_tolerance=1.0e-12,
        threshold_tie_break=("higher_f2", "higher_recall", "higher_precision", "lower_threshold"),
        ap_improvement_tolerance=0.000001,
        max_ap_regression_for_mcc_gain=0.0005,
        max_roc_auc_regression=0.005,
        required_mcc_gain=0.005,
        preferred_search_workers=4,
        preferred_threads_per_search_fit=5,
        preferred_single_fit_threads=16,
        reserve_logical_threads=2,
        high_performance_local=True,
        checkpoint_each_fold=True,
        resume_supported=True,
        output_directory="artifacts/imbalance_threshold",
        report_directory="reports/phase12_imbalance_threshold",
    )


def settings_payload(settings: ImbalanceThresholdSettings) -> dict[str, Any]:
    strategies = []
    for index, strategy_id in enumerate(settings.strategy_ids):
        item: dict[str, Any] = {
            "id": strategy_id,
            "type": settings.strategy_types[index],
        }
        if settings.strategy_values[index] is not None:
            item["value"] = settings.strategy_values[index]
        strategies.append(item)
    return {
        "tracks": list(settings.tracks),
        "primary_ranking_metric": settings.primary_ranking_metric,
        "strategies": strategies,
        "strategy_selection": {
            "max_ap_tolerance": settings.max_ap_tolerance,
            "max_min_ap_drop": settings.max_min_ap_drop,
            "max_roc_auc_drop": settings.max_roc_auc_drop,
            "prefer_none_mcc_tolerance": settings.prefer_none_mcc_tolerance,
        },
        "threshold": {
            "primary_metric": "matthews_correlation_coefficient",
            "grid": {
                "start": settings.threshold_start,
                "stop": settings.threshold_stop,
                "step": settings.threshold_step,
            },
            "tie_tolerance": settings.threshold_tie_tolerance,
            "tie_break": list(settings.threshold_tie_break),
        },
        "outer_replacement": {
            "ap_improvement_tolerance": settings.ap_improvement_tolerance,
            "max_ap_regression_for_mcc_gain": settings.max_ap_regression_for_mcc_gain,
            "max_roc_auc_regression": settings.max_roc_auc_regression,
            "required_mcc_gain": settings.required_mcc_gain,
        },
        "compute": {
            "high_performance_local": settings.high_performance_local,
            "preferred_search_workers": settings.preferred_search_workers,
            "preferred_threads_per_search_fit": settings.preferred_threads_per_search_fit,
            "preferred_single_fit_threads": settings.preferred_single_fit_threads,
            "reserve_logical_threads": settings.reserve_logical_threads,
        },
        "checkpoint_each_fold": settings.checkpoint_each_fold,
        "resume_supported": settings.resume_supported,
        "output_directory": settings.output_directory,
        "report_directory": settings.report_directory,
    }


__all__ = [
    "ImbalanceThresholdError",
    "ImbalanceThresholdSettings",
    "PHASE12_VERSION",
    "STRATEGY_IDS",
    "STRATEGY_TYPES",
    "STRATEGY_VALUES",
    "TRACKS",
    "load_imbalance_threshold_settings",
    "settings_payload",
]
