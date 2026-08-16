"""Versioned, offline Phase 10 policy and provenance contract validation."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

import yaml

from ..paths import discover_repository_root
from .config import SEARCH_PARAMETER_NAMES, TRACKS, load_optimization_settings, settings_payload
from .input import EXPECTED_FEATURE_SETS, EXPECTED_PHASE9_TARGET_HASHES

CONTRACT_VERSION = "phase10_catboost_optimization_v2"
REQUIRED_PHASE9_RUN_ID = "20260811T_PHASE9_FINAL"
OBJECTIVE_METRIC = "mean_average_precision"


def load_optimization_contract(
    project_root: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Load the committed contract and return its semantic SHA-256 checksum."""

    root = discover_repository_root(project_root)
    path = root / "contracts" / "catboost_optimization.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 10 contract must be a YAML object.")
    checksum = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return payload, checksum


def _expected_phase9_lock() -> dict[str, Any]:
    return {
        "required_run_id": REQUIRED_PHASE9_RUN_ID,
        "required_hardened_status": "HARDENED_PASS",
        "target_hashes": dict(EXPECTED_PHASE9_TARGET_HASHES),
        "feature_sets": {
            track: {
                "experiment_id": "E1" if track == "T1" else "E3",
                "feature_count": EXPECTED_FEATURE_SETS["E1" if track == "T1" else "E3"][0],
                "feature_set_sha256": EXPECTED_FEATURE_SETS["E1" if track == "T1" else "E3"][1],
            }
            for track in TRACKS
        },
    }


def _expected_dependency_policy() -> dict[str, Any]:
    return {
        "optimization_extra": "optimization",
        "requirements": {
            "python": ">=3.11",
            "optuna": ">=4,<5",
            "catboost": ">=1.2.10,<2",
            "pandas": ">=2.0,<3",
            "numpy": ">=1.24,<3",
            "pyarrow": ">=14.0,<20",
            "scikit-learn": ">=1.5,<2",
        },
    }


def _expected_policy(settings: Any) -> dict[str, Any]:
    return {
        "version": CONTRACT_VERSION,
        "phase9_lock": _expected_phase9_lock(),
        "tracks": list(TRACKS),
        "phase9_experiments": {"T1": "E1", "T3": "E3"},
        "search_parameters": list(SEARCH_PARAMETER_NAMES),
        "inner_validation": {
            "source_outer_split": "TRAIN",
            "fold_count": 3,
            "boundary_fractions": [0.55, 0.70, 0.85, 1.0],
            "same_date_groups": True,
            "strict_before": True,
            "fold_evidence": {
                "required_artifact": "trial_fold_metrics.parquet",
                "successful_trial_fold_count": 3,
                "fold_ids": [1, 2, 3],
                "aggregate_metric_reproduction": "required",
                "winning_trial_retraining": "required",
            },
        },
        "trial_budget": {
            "trials_per_track": 50,
            "minimum_completed_trials": 40,
            "minimum_completed_fraction": 0.80,
        },
        "search": {
            "sampler": settings.sampler,
            "n_startup_trials": settings.n_startup_trials,
            "parallel_jobs": settings.parallel_jobs,
            "pruning": settings.pruning,
            "objective_metric": OBJECTIVE_METRIC,
            "search_space": settings.search_space,
        },
        "fixed_catboost_policy": settings.fixed_parameters,
        "class_imbalance_policy": {
            "class_weights": "none",
            "auto_class_weights": "none",
            "scale_pos_weight": "none",
            "resampling": "prohibited",
        },
        "early_stopping_policy": "prohibited",
        "threshold_policy": {"value": settings.threshold, "tuning": "prohibited"},
        "calibration_policy": "prohibited",
        "feature_selection_policy": {"phase10": "prohibited", "owner": "phase11"},
        "objective_policy": {
            "primary_metric": OBJECTIVE_METRIC,
            "direction": "maximize",
            "tie_breakers": [
                "min_average_precision",
                "mean_roc_auc",
                "std_average_precision",
                "mean_log_loss",
            ],
        },
        "outer_validation": {
            "loaded_after_study_freeze_only": True,
            "finalist_count": 2,
            "finalist_models_only": True,
        },
        "dependency_policy": _expected_dependency_policy(),
        "test_target_access": {
            "policy": "FORBIDDEN_UNTIL_PHASE_15",
            "first_allowed_phase": 15,
        },
        "provenance": {
            "manifest_contract_checksum_field": "contract_checksum",
            "manifest_contract_snapshot_field": "contract_policy_snapshot",
        },
        "feature_selection_owner": "phase11",
    }


def _validate_declared_optimization_extra(root: Path) -> str | None:
    try:
        with (root / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
        optional = project.get("optional-dependencies", {})
        requirements = [str(item) for item in optional.get("optimization", [])]
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return f"Could not read pyproject optimization extra: {exc}"
    if requirements != ["optuna>=4,<5"]:
        return "pyproject optimization extra must contain exactly optuna>=4,<5."
    return None


def validate_optimization_contract(project_root: Path | None = None) -> dict[str, Any]:
    """Validate the complete Phase 10 policy and configuration without data access."""

    errors: list[str] = []
    warnings: list[str] = []
    checksum: str | None = None
    contract: dict[str, Any] = {}
    try:
        root = discover_repository_root(project_root)
        contract, checksum = load_optimization_contract(root)
        settings = load_optimization_settings(root)
        policy = contract.get("phase10")
        if not isinstance(policy, dict):
            errors.append("Phase 10 contract must contain phase10 mapping.")
        else:
            expected = _expected_policy(settings)
            for key, value in expected.items():
                if policy.get(key) != value:
                    errors.append(f"Phase 10 contract policy differs: {key}.")
            if settings.tracks != TRACKS:
                errors.append("Phase 10 configuration tracks are not exactly T1 and T3.")
            if settings.trials_per_track != 50:
                errors.append("Phase 10 configuration must declare exactly 50 trials per track.")
            if settings.search_space != expected["search"]["search_space"]:
                errors.append("Phase 10 configuration search space differs from the contract.")
            if settings_payload(settings)["fixed_parameters"] != expected["fixed_catboost_policy"]:
                errors.append("Phase 10 fixed CatBoost settings differ from the contract.")
        extra_error = _validate_declared_optimization_extra(root)
        if extra_error:
            errors.append(extra_error)
    except Exception as exc:
        errors.append(str(exc))
    phase10 = contract.get("phase10")
    return {
        "status": "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS"),
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "contract_version": phase10.get("version") if isinstance(phase10, dict) else None,
        "contract_checksum": checksum,
        "contract": contract,
    }


__all__ = [
    "CONTRACT_VERSION",
    "OBJECTIVE_METRIC",
    "REQUIRED_PHASE9_RUN_ID",
    "load_optimization_contract",
    "validate_optimization_contract",
]
