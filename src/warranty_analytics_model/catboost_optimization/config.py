"""Load and validate the closed Phase 10 CatBoost optimization configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..paths import discover_repository_root
from .models import OptimizationError, OptimizationSettings

TRACKS = ("T1", "T3")
TRACK_TO_EXPERIMENT = {"T1": "E1", "T3": "E3"}
SEARCH_PARAMETER_NAMES = (
    "iterations",
    "learning_rate",
    "depth",
    "l2_leaf_reg",
    "random_strength",
    "bagging_temperature",
    "border_count",
    "rsm",
)
FORBIDDEN_KEYS = {
    "class_weights",
    "auto_class_weights",
    "scale_pos_weight",
    "early_stopping_rounds",
    "od_type",
    "od_wait",
    "eval_set",
}


def _as_float_tuple(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise OptimizationError(f"Phase 10 {label} must be a non-empty list.")
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise OptimizationError(f"Phase 10 {label} must contain numbers.") from exc
    return values


def _validate_search_space(space: Any) -> dict[str, Any]:
    if not isinstance(space, dict) or set(space) != set(SEARCH_PARAMETER_NAMES):
        actual = sorted(space) if isinstance(space, dict) else type(space).__name__
        raise OptimizationError(
            f"Phase 10 search space must contain exactly the allowlisted parameters; got {actual}."
        )
    expected = {
        "iterations": {"type": "categorical", "values": [300, 500, 700, 1000, 1400]},
        "learning_rate": {"type": "float", "low": 0.015, "high": 0.15, "log": True},
        "depth": {"type": "int", "low": 4, "high": 9},
        "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 30.0, "log": True},
        "random_strength": {"type": "float", "low": 0.0, "high": 3.0},
        "bagging_temperature": {"type": "float", "low": 0.0, "high": 5.0},
        "border_count": {"type": "categorical", "values": [64, 128, 254]},
        "rsm": {"type": "float", "low": 0.70, "high": 1.00},
    }
    for name in SEARCH_PARAMETER_NAMES:
        actual = space[name]
        if actual != expected[name]:
            raise OptimizationError(
                f"Phase 10 search-space definition changed for {name}: {actual!r}."
            )
    return {str(key): value for key, value in space.items()}


def _validate_fixed_parameters(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise OptimizationError("Phase 10 fixed_parameters must be a mapping.")
    forbidden = sorted(FORBIDDEN_KEYS & set(raw))
    if forbidden:
        raise OptimizationError("Phase 10 forbids CatBoost options: " + ", ".join(forbidden))
    required = {
        "loss_function": "Logloss",
        "bootstrap_type": "Bayesian",
        "random_seed": 20260810,
        "task_type": "CPU",
        "thread_count": 10,
        "allow_writing_files": False,
        "verbose": False,
        "use_best_model": False,
    }
    for key, expected in required.items():
        if raw.get(key) != expected:
            raise OptimizationError(f"Phase 10 fixed CatBoost setting changed: {key}.")
    if set(raw) & set(SEARCH_PARAMETER_NAMES):
        raise OptimizationError("Phase 10 fixed parameters cannot contain search parameters.")
    return {str(key): value for key, value in raw.items()}


def load_optimization_settings(project_root: Path | None = None) -> OptimizationSettings:
    """Read configs/catboost_optimization.yaml and reject unsafe changes."""

    root = discover_repository_root(project_root)
    path = root / "configs" / "catboost_optimization.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise OptimizationError(f"Could not read Phase 10 configuration: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("catboost_optimization"), dict):
        raise OptimizationError("Phase 10 configuration must contain catboost_optimization.")
    raw = payload["catboost_optimization"]
    tracks = tuple(str(item) for item in raw.get("tracks", []))
    if tracks != TRACKS:
        raise OptimizationError(f"Phase 10 tracks must be exactly {TRACKS}; got {tracks}.")
    trials = raw.get("trials_per_track")
    if not isinstance(trials, int) or trials < 1:
        raise OptimizationError("Phase 10 trials_per_track must be a positive integer.")
    if raw.get("parallel_jobs") != 1 or raw.get("pruning") is not False:
        raise OptimizationError("Phase 10 requires sequential studies with pruning disabled.")
    threshold = float(raw.get("threshold", -1))
    if threshold != 0.5:
        raise OptimizationError("Phase 10 threshold must be fixed at 0.5.")
    seed = raw.get("random_seed")
    if not isinstance(seed, int) or seed != 20260810:
        raise OptimizationError("Phase 10 random_seed must be 20260810.")
    if raw.get("sampler") != "TPESampler" or raw.get("n_startup_trials") != 10:
        raise OptimizationError("Phase 10 sampler must be TPESampler with 10 startup trials.")
    fractions = _as_float_tuple(raw.get("inner_fold_fractions"), "inner_fold_fractions")
    if fractions != (0.55, 0.70, 0.85, 1.0):
        raise OptimizationError("Phase 10 inner fold fractions must be 0.55, 0.70, 0.85, 1.00.")
    fixed = _validate_fixed_parameters(raw.get("fixed_parameters"))
    space = _validate_search_space(raw.get("search_space"))
    return OptimizationSettings(
        tracks=tracks,
        trials_per_track=trials,
        parallel_jobs=1,
        pruning=False,
        threshold=threshold,
        random_seed=seed,
        sampler="TPESampler",
        n_startup_trials=10,
        fixed_parameters=fixed,
        search_space=space,
        inner_fold_fractions=fractions,
        minimum_train_positive=int(raw.get("minimum_train_positive", 40)),
        minimum_validation_positive=int(raw.get("minimum_validation_positive", 10)),
        instability_std_ap_warning=float(raw.get("instability_std_ap_warning", 0.04)),
        failure_warning_fraction=float(raw.get("failure_warning_fraction", 0.10)),
        minimum_completed_fraction=float(raw.get("minimum_completed_fraction", 0.80)),
        output_directory=str(raw.get("output_directory", "artifacts/catboost_optimization")),
        report_directory=str(raw.get("report_directory", "reports/phase10_catboost_optimization")),
        compression=str(raw.get("compression", "snappy")),
    )


def settings_payload(settings: OptimizationSettings) -> dict[str, Any]:
    """Return stable, secret-free configuration metadata."""

    return {
        "tracks": list(settings.tracks),
        "trials_per_track": settings.trials_per_track,
        "parallel_jobs": settings.parallel_jobs,
        "pruning": settings.pruning,
        "threshold": settings.threshold,
        "random_seed": settings.random_seed,
        "sampler": settings.sampler,
        "n_startup_trials": settings.n_startup_trials,
        "fixed_parameters": settings.fixed_parameters,
        "search_space": settings.search_space,
        "inner_fold_fractions": list(settings.inner_fold_fractions),
        "minimum_train_positive": settings.minimum_train_positive,
        "minimum_validation_positive": settings.minimum_validation_positive,
        "instability_std_ap_warning": settings.instability_std_ap_warning,
        "failure_warning_fraction": settings.failure_warning_fraction,
        "minimum_completed_fraction": settings.minimum_completed_fraction,
        "output_directory": settings.output_directory,
        "report_directory": settings.report_directory,
        "compression": settings.compression,
    }
