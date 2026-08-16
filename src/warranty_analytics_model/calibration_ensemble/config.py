"""Fail-closed configuration for Phase 13."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..paths import discover_repository_root
from . import PHASE13_VERSION

TRACKS = ("T1", "T3")
CALIBRATION_METHODS = ("C0_NONE", "C1_SIGMOID", "C2_ISOTONIC")
ENSEMBLE_WEIGHTS = tuple(round(index / 10.0, 1) for index in range(11))
CALIBRATION_COMPLEXITY = {method: index for index, method in enumerate(CALIBRATION_METHODS)}
PHASE13_ERROR = RuntimeError


class CalibrationEnsembleError(ValueError):
    """Raised when Phase 13 input or policy is unsafe."""


@dataclass(frozen=True, slots=True)
class CalibrationEnsembleSettings:
    tracks: tuple[str, ...]
    calibration_methods: tuple[str, ...]
    sigmoid_epsilon: float
    sigmoid_solver: str
    sigmoid_max_iter: int
    isotonic_y_min: float
    isotonic_y_max: float
    isotonic_out_of_bounds: str
    isotonic_minimum_training_positives: int
    isotonic_minimum_training_negatives: int
    isotonic_minimum_unique_probabilities: int
    reliability_bins: int
    ensemble_weights: tuple[float, ...]
    selection_max_ap_drop: float
    selection_max_min_fold_ap_drop: float
    selection_max_roc_auc_drop: float
    none_log_loss_tolerance: float
    none_brier_tolerance: float
    ranking_minimum_ap_improvement: float
    ranking_max_min_fold_ap_drop: float
    ranking_max_roc_auc_drop: float
    calibration_route_max_ap_drop: float
    calibration_route_max_min_fold_ap_drop: float
    calibration_route_max_roc_auc_drop: float
    calibration_route_min_log_loss_improvement: float
    calibration_route_min_brier_improvement: float
    threshold_start: float
    threshold_stop: float
    threshold_step: float
    threshold_tie_tolerance: float
    validation_calibration_max_ap_drop: float
    validation_calibration_max_roc_auc_drop: float
    validation_calibration_max_log_loss_regression: float
    validation_calibration_max_brier_regression: float
    validation_calibration_min_log_loss_improvement: float
    validation_calibration_min_brier_improvement: float
    validation_ensemble_ap_improvement_tolerance: float
    validation_ensemble_max_ap_drop_for_calibration_route: float
    validation_ensemble_max_roc_auc_drop: float
    validation_ensemble_max_log_loss_regression: float
    validation_ensemble_min_log_loss_improvement: float
    validation_ensemble_min_brier_improvement: float
    reserve_logical_threads: int
    preferred_calibration_workers: int
    preferred_catboost_replay_threads: int
    checkpoint_each_calibration_fold: bool
    resume_supported: bool
    output_directory: str
    report_directory: str


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CalibrationEnsembleError(f"Phase 13 {label} must be a mapping.")
    return {str(key): item for key, item in value.items()}


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationEnsembleError(f"Phase 13 {label} must be numeric.")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise CalibrationEnsembleError(f"Phase 13 {label} must be finite.")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CalibrationEnsembleError(f"Phase 13 {label} must be a positive integer.")
    return int(value)


def load_calibration_ensemble_settings(
    project_root: Path | None = None,
) -> CalibrationEnsembleSettings:
    root = discover_repository_root(project_root)
    path = root / "configs" / "calibration_ensemble.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CalibrationEnsembleError(f"Could not read Phase 13 configuration: {path}") from exc
    root_payload = _mapping(raw, "configuration")
    payload = _mapping(root_payload.get("phase13_calibration_ensemble"), "configuration")
    tracks = tuple(str(item) for item in payload.get("tracks", []))
    if tracks != TRACKS:
        raise CalibrationEnsembleError("Phase 13 tracks are not exactly T1/T3.")
    calibration = _mapping(payload.get("calibration"), "calibration")
    methods = tuple(str(item) for item in calibration.get("methods", []))
    if methods != CALIBRATION_METHODS:
        raise CalibrationEnsembleError("Phase 13 calibration methods drifted.")
    sigmoid = _mapping(calibration.get("sigmoid"), "sigmoid")
    isotonic = _mapping(calibration.get("isotonic"), "isotonic")
    selection = _mapping(calibration.get("selection"), "selection")
    reliability = _mapping(payload.get("reliability"), "reliability")
    ensemble = _mapping(payload.get("ensemble"), "ensemble")
    ranking = _mapping(ensemble.get("ranking_route"), "ensemble ranking route")
    calibration_route = _mapping(ensemble.get("calibration_route"), "ensemble calibration route")
    threshold = _mapping(payload.get("threshold"), "threshold")
    validation = _mapping(payload.get("validation"), "validation")
    compute = _mapping(payload.get("compute"), "compute")
    weights = tuple(round(float(item), 1) for item in ensemble.get("t1_weights", []))
    if weights != ENSEMBLE_WEIGHTS:
        raise CalibrationEnsembleError("Phase 13 ensemble weights must be exactly 0.0..1.0 by 0.1.")
    if sigmoid.get("penalty") is not None or sigmoid.get("class_weight") is not None:
        raise CalibrationEnsembleError("Phase 13 sigmoid weighting policy drifted.")
    if sigmoid.get("solver") != "lbfgs":
        raise CalibrationEnsembleError("Phase 13 sigmoid solver drifted.")
    if isotonic.get("out_of_bounds") != "clip":
        raise CalibrationEnsembleError("Phase 13 isotonic out-of-bounds policy drifted.")
    if reliability.get("method") != "deterministic_equal_frequency":
        raise CalibrationEnsembleError("Phase 13 reliability bin policy drifted.")
    if threshold.get("objective") != "MCC":
        raise CalibrationEnsembleError("Phase 13 threshold objective must be MCC.")
    return CalibrationEnsembleSettings(
        tracks=tracks,
        calibration_methods=methods,
        sigmoid_epsilon=_number(sigmoid.get("epsilon"), "sigmoid epsilon"),
        sigmoid_solver=str(sigmoid["solver"]),
        sigmoid_max_iter=_positive_int(sigmoid.get("max_iter"), "sigmoid max_iter"),
        isotonic_y_min=_number(isotonic.get("y_min"), "isotonic y_min"),
        isotonic_y_max=_number(isotonic.get("y_max"), "isotonic y_max"),
        isotonic_out_of_bounds=str(isotonic["out_of_bounds"]),
        isotonic_minimum_training_positives=_positive_int(
            isotonic.get("minimum_training_positives"), "isotonic positive guard"
        ),
        isotonic_minimum_training_negatives=_positive_int(
            isotonic.get("minimum_training_negatives"), "isotonic negative guard"
        ),
        isotonic_minimum_unique_probabilities=_positive_int(
            isotonic.get("minimum_unique_probabilities"), "isotonic unique-probability guard"
        ),
        reliability_bins=_positive_int(reliability.get("bins"), "reliability bins"),
        ensemble_weights=weights,
        selection_max_ap_drop=_number(selection.get("max_ap_drop"), "selection AP drop"),
        selection_max_min_fold_ap_drop=_number(
            selection.get("max_min_fold_ap_drop"), "selection minimum-fold AP drop"
        ),
        selection_max_roc_auc_drop=_number(selection.get("max_roc_auc_drop"), "selection ROC drop"),
        none_log_loss_tolerance=_number(
            selection.get("none_log_loss_tolerance"), "NONE log-loss tolerance"
        ),
        none_brier_tolerance=_number(selection.get("none_brier_tolerance"), "NONE Brier tolerance"),
        ranking_minimum_ap_improvement=_number(
            ranking.get("minimum_ap_improvement"), "ensemble ranking AP improvement"
        ),
        ranking_max_min_fold_ap_drop=_number(
            ranking.get("max_min_fold_ap_drop"), "ensemble ranking minimum-fold AP drop"
        ),
        ranking_max_roc_auc_drop=_number(
            ranking.get("max_roc_auc_drop"), "ensemble ranking ROC drop"
        ),
        calibration_route_max_ap_drop=_number(
            calibration_route.get("max_ap_drop"), "ensemble calibration AP drop"
        ),
        calibration_route_max_min_fold_ap_drop=_number(
            calibration_route.get("max_min_fold_ap_drop"),
            "ensemble calibration minimum-fold AP drop",
        ),
        calibration_route_max_roc_auc_drop=_number(
            calibration_route.get("max_roc_auc_drop"), "ensemble calibration ROC drop"
        ),
        calibration_route_min_log_loss_improvement=_number(
            calibration_route.get("minimum_log_loss_improvement"), "ensemble log-loss improvement"
        ),
        calibration_route_min_brier_improvement=_number(
            calibration_route.get("minimum_brier_improvement"), "ensemble Brier improvement"
        ),
        threshold_start=_number(threshold.get("start"), "threshold start"),
        threshold_stop=_number(threshold.get("stop"), "threshold stop"),
        threshold_step=_number(threshold.get("step"), "threshold step"),
        threshold_tie_tolerance=_number(threshold.get("tie_tolerance"), "threshold tie tolerance"),
        validation_calibration_max_ap_drop=_number(
            validation.get("calibration_max_ap_drop"), "validation calibration AP drop"
        ),
        validation_calibration_max_roc_auc_drop=_number(
            validation.get("calibration_max_roc_auc_drop"), "validation calibration ROC drop"
        ),
        validation_calibration_max_log_loss_regression=_number(
            validation.get("calibration_max_log_loss_regression"),
            "validation calibration log-loss regression",
        ),
        validation_calibration_max_brier_regression=_number(
            validation.get("calibration_max_brier_regression"),
            "validation calibration Brier regression",
        ),
        validation_calibration_min_log_loss_improvement=_number(
            validation.get("calibration_min_log_loss_improvement"),
            "validation calibration log-loss improvement",
        ),
        validation_calibration_min_brier_improvement=_number(
            validation.get("calibration_min_brier_improvement"),
            "validation calibration Brier improvement",
        ),
        validation_ensemble_ap_improvement_tolerance=_number(
            validation.get("ensemble_ap_improvement_tolerance"), "validation ensemble AP tolerance"
        ),
        validation_ensemble_max_ap_drop_for_calibration_route=_number(
            validation.get("ensemble_max_ap_drop_for_calibration_route"),
            "validation ensemble calibration AP drop",
        ),
        validation_ensemble_max_roc_auc_drop=_number(
            validation.get("ensemble_max_roc_auc_drop"), "validation ensemble ROC drop"
        ),
        validation_ensemble_max_log_loss_regression=_number(
            validation.get("ensemble_max_log_loss_regression"),
            "validation ensemble log-loss regression",
        ),
        validation_ensemble_min_log_loss_improvement=_number(
            validation.get("ensemble_min_log_loss_improvement"),
            "validation ensemble log-loss improvement",
        ),
        validation_ensemble_min_brier_improvement=_number(
            validation.get("ensemble_min_brier_improvement"),
            "validation ensemble Brier improvement",
        ),
        reserve_logical_threads=_positive_int(
            compute.get("reserve_logical_threads"), "reserved logical threads"
        ),
        preferred_calibration_workers=_positive_int(
            compute.get("preferred_calibration_workers"), "calibration workers"
        ),
        preferred_catboost_replay_threads=_positive_int(
            compute.get("preferred_catboost_replay_threads"), "CatBoost replay threads"
        ),
        checkpoint_each_calibration_fold=bool(payload.get("checkpoint_each_calibration_fold")),
        resume_supported=bool(payload.get("resume_supported")),
        output_directory="artifacts/calibration_ensemble",
        report_directory="reports/phase13_calibration_ensemble",
    )


def settings_payload(settings: CalibrationEnsembleSettings) -> dict[str, Any]:
    """Return a stable configuration snapshot for manifests."""

    return {
        "phase": 13,
        "version": PHASE13_VERSION,
        "tracks": list(settings.tracks),
        "calibration_methods": list(settings.calibration_methods),
        "ensemble_weights": list(settings.ensemble_weights),
        "reliability_bins": settings.reliability_bins,
        "threshold": {
            "start": settings.threshold_start,
            "stop": settings.threshold_stop,
            "step": settings.threshold_step,
            "objective": "MCC",
            "tie_tolerance": settings.threshold_tie_tolerance,
        },
        "compute": {
            "reserve_logical_threads": settings.reserve_logical_threads,
            "preferred_calibration_workers": settings.preferred_calibration_workers,
            "preferred_catboost_replay_threads": settings.preferred_catboost_replay_threads,
        },
        "checkpoint_each_calibration_fold": settings.checkpoint_each_calibration_fold,
        "resume_supported": settings.resume_supported,
    }


__all__ = [
    "CALIBRATION_COMPLEXITY",
    "CALIBRATION_METHODS",
    "CalibrationEnsembleError",
    "CalibrationEnsembleSettings",
    "ENSEMBLE_WEIGHTS",
    "PHASE13_VERSION",
    "TRACKS",
    "load_calibration_ensemble_settings",
    "settings_payload",
]
