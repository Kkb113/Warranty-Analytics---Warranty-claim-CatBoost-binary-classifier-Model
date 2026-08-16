"""Strict Phase 11 configuration loading and compute planning settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..paths import discover_repository_root


class FeatureSelectionError(ValueError):
    """Raised when a Phase 11 safety or configuration gate fails."""


TRACKS = ("T1", "T3")
TRACK_TO_EXPERIMENT = {"T1": "E1", "T3": "E3"}
PHASE11_VERSION = "phase11_feature_selection_ablation_v1"


@dataclass(frozen=True, slots=True)
class FeatureSelectionSettings:
    tracks: tuple[str, ...]
    primary_metric: str
    candidate_fractions: tuple[float, ...]
    minimum_feature_count: int
    maximum_candidate_subsets_per_track: int
    simpler_enabled: bool
    maximum_ap_tolerance: float
    maximum_min_ap_drop: float
    maximum_roc_auc_drop: float
    maximum_log_loss_increase: float
    ap_improvement_tolerance: float
    complexity_allowed: bool
    complexity_maximum_ap_drop: float
    complexity_minimum_reduction_fraction: float
    complexity_maximum_roc_auc_drop: float
    complexity_maximum_log_loss_increase: float
    preferred_max_workers: int
    preferred_threads_per_worker: int
    preferred_single_fit_threads: int
    reserve_logical_threads: int
    high_performance_local: bool
    checkpoint_each_fold: bool
    resume_supported: bool
    output_directory: str
    report_directory: str


def _number(raw: Any, label: str) -> float:
    if isinstance(raw, bool):
        raise FeatureSelectionError(f"Phase 11 {label} must be numeric.")
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise FeatureSelectionError(f"Phase 11 {label} must be numeric.") from exc


def _positive_int(raw: Any, label: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise FeatureSelectionError(f"Phase 11 {label} must be a positive integer.")
    return int(raw)


def load_feature_selection_settings(project_root: Path | None = None) -> FeatureSelectionSettings:
    root = discover_repository_root(project_root)
    path = root / "configs" / "feature_selection_ablation.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FeatureSelectionError(f"Could not read Phase 11 configuration: {path}") from exc
    raw = payload.get("phase11_feature_selection") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        raise FeatureSelectionError(
            "Phase 11 configuration must contain phase11_feature_selection."
        )
    tracks = tuple(str(item) for item in raw.get("tracks", []))
    if tracks != TRACKS:
        raise FeatureSelectionError(f"Phase 11 tracks must be exactly {TRACKS}; got {tracks}.")
    if raw.get("primary_metric") != "mean_average_precision":
        raise FeatureSelectionError("Phase 11 primary_metric must be mean_average_precision.")
    fractions_raw = raw.get("candidate_fractions")
    if not isinstance(fractions_raw, list) or not fractions_raw:
        raise FeatureSelectionError("Phase 11 candidate_fractions must be a non-empty list.")
    fractions = tuple(_number(item, "candidate_fractions") for item in fractions_raw)
    if fractions != (1.0, 0.85, 0.70, 0.55, 0.40, 0.25):
        raise FeatureSelectionError("Phase 11 candidate_fractions drifted from the locked values.")
    if (
        any(value <= 0 or value > 1 for value in fractions)
        or tuple(sorted(fractions, reverse=True)) != fractions
    ):
        raise FeatureSelectionError("Phase 11 candidate_fractions must descend from 1.00 to 0.25.")
    importance = raw.get("feature_importance")
    if importance != {
        "loss_function_change": True,
        "shap_mean_absolute": True,
        "fold_stability_required": True,
    }:
        raise FeatureSelectionError("Phase 11 feature importance policy is incomplete or changed.")
    simpler = raw.get("simpler_candidate_selection")
    outer = raw.get("outer_validation_replacement")
    complexity = outer.get("complexity_tradeoff") if isinstance(outer, dict) else None
    compute = raw.get("compute")
    if (
        not isinstance(simpler, dict)
        or not isinstance(outer, dict)
        or not isinstance(complexity, dict)
    ):
        raise FeatureSelectionError("Phase 11 selection/replacement policy is incomplete.")
    if not isinstance(compute, dict):
        raise FeatureSelectionError("Phase 11 compute policy is missing.")
    minimum = _positive_int(raw.get("minimum_feature_count"), "minimum_feature_count")
    maximum = _positive_int(
        raw.get("maximum_candidate_subsets_per_track"), "maximum_candidate_subsets_per_track"
    )
    if maximum > 8:
        raise FeatureSelectionError("Phase 11 candidate subset cap cannot exceed 8.")
    return FeatureSelectionSettings(
        tracks=tracks,
        primary_metric="mean_average_precision",
        candidate_fractions=fractions,
        minimum_feature_count=minimum,
        maximum_candidate_subsets_per_track=maximum,
        simpler_enabled=bool(simpler.get("enabled")) and simpler.get("enabled") is True,
        maximum_ap_tolerance=_number(simpler.get("maximum_ap_tolerance"), "maximum_ap_tolerance"),
        maximum_min_ap_drop=_number(simpler.get("maximum_min_ap_drop"), "maximum_min_ap_drop"),
        maximum_roc_auc_drop=_number(simpler.get("maximum_roc_auc_drop"), "maximum_roc_auc_drop"),
        maximum_log_loss_increase=_number(
            simpler.get("maximum_log_loss_increase"), "maximum_log_loss_increase"
        ),
        ap_improvement_tolerance=_number(
            outer.get("ap_improvement_tolerance"), "ap_improvement_tolerance"
        ),
        complexity_allowed=complexity.get("allowed") is True,
        complexity_maximum_ap_drop=_number(
            complexity.get("maximum_ap_drop"), "complexity.maximum_ap_drop"
        ),
        complexity_minimum_reduction_fraction=_number(
            complexity.get("minimum_feature_reduction_fraction"),
            "complexity.minimum_feature_reduction_fraction",
        ),
        complexity_maximum_roc_auc_drop=_number(
            complexity.get("maximum_roc_auc_drop"), "complexity.maximum_roc_auc_drop"
        ),
        complexity_maximum_log_loss_increase=_number(
            complexity.get("maximum_log_loss_increase"), "complexity.maximum_log_loss_increase"
        ),
        preferred_max_workers=_positive_int(
            compute.get("preferred_max_workers"), "preferred_max_workers"
        ),
        preferred_threads_per_worker=_positive_int(
            compute.get("preferred_threads_per_worker"), "preferred_threads_per_worker"
        ),
        preferred_single_fit_threads=_positive_int(
            compute.get("preferred_single_fit_threads"), "preferred_single_fit_threads"
        ),
        reserve_logical_threads=int(compute.get("reserve_logical_threads", 0)),
        high_performance_local=compute.get("high_performance_local") is True,
        checkpoint_each_fold=raw.get("checkpoint_each_fold") is True,
        resume_supported=raw.get("resume_supported") is True,
        output_directory=str(raw.get("output_directory")),
        report_directory=str(raw.get("report_directory")),
    )


def settings_payload(settings: FeatureSelectionSettings) -> dict[str, Any]:
    return {
        "tracks": list(settings.tracks),
        "primary_metric": settings.primary_metric,
        "candidate_fractions": list(settings.candidate_fractions),
        "minimum_feature_count": settings.minimum_feature_count,
        "maximum_candidate_subsets_per_track": settings.maximum_candidate_subsets_per_track,
        "simpler_candidate_selection": {
            "enabled": settings.simpler_enabled,
            "maximum_ap_tolerance": settings.maximum_ap_tolerance,
            "maximum_min_ap_drop": settings.maximum_min_ap_drop,
            "maximum_roc_auc_drop": settings.maximum_roc_auc_drop,
            "maximum_log_loss_increase": settings.maximum_log_loss_increase,
        },
        "outer_validation_replacement": {
            "ap_improvement_tolerance": settings.ap_improvement_tolerance,
            "complexity_tradeoff": {
                "allowed": settings.complexity_allowed,
                "maximum_ap_drop": settings.complexity_maximum_ap_drop,
                "minimum_feature_reduction_fraction": settings.complexity_minimum_reduction_fraction,
                "maximum_roc_auc_drop": settings.complexity_maximum_roc_auc_drop,
                "maximum_log_loss_increase": settings.complexity_maximum_log_loss_increase,
            },
        },
        "compute": {
            "high_performance_local": settings.high_performance_local,
            "preferred_max_workers": settings.preferred_max_workers,
            "preferred_threads_per_worker": settings.preferred_threads_per_worker,
            "preferred_single_fit_threads": settings.preferred_single_fit_threads,
            "reserve_logical_threads": settings.reserve_logical_threads,
        },
        "checkpoint_each_fold": settings.checkpoint_each_fold,
        "resume_supported": settings.resume_supported,
        "output_directory": settings.output_directory,
        "report_directory": settings.report_directory,
    }

