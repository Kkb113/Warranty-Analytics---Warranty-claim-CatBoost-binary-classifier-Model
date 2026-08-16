"""Offline validation for the locked Phase 13 policy contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ..paths import discover_repository_root
from . import PHASE13_VERSION
from .config import (
    CALIBRATION_METHODS,
    ENSEMBLE_WEIGHTS,
    TRACKS,
    load_calibration_ensemble_settings,
)


def load_calibration_ensemble_contract(
    project_root: Path | None = None,
) -> tuple[dict[str, Any], str]:
    root = discover_repository_root(project_root)
    path = root / "contracts" / "calibration_ensemble_v1.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read Phase 13 contract: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("phase13"), dict):
        raise ValueError("Phase 13 contract must contain a phase13 mapping.")
    checksum = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return payload, checksum


def validate_calibration_ensemble_contract(
    project_root: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    payload: dict[str, Any] = {}
    checksum: str | None = None
    try:
        root = discover_repository_root(project_root)
        payload, checksum = load_calibration_ensemble_contract(root)
        settings = load_calibration_ensemble_settings(root)
        policy = payload["phase13"]
        exact = {
            "phase": 13,
            "version": PHASE13_VERSION,
            "required_phase12_status": "HARDENED_PASS",
            "base_model_retraining": "prohibited",
            "feature_changes": "prohibited",
            "hyperparameter_retuning": "prohibited",
            "imbalance_reoptimization": "prohibited",
            "calibration_methods": ["NONE", "SIGMOID", "ISOTONIC"],
            "calibration_selection_target_access": "TRAIN_ONLY",
            "ensemble_type": "CONVEX_PROBABILITY_BLEND",
            "ensemble_weights": list(ENSEMBLE_WEIGHTS),
            "stacking": "prohibited",
            "meta_model": "prohibited",
            "threshold_selection": ["TRAIN_CROSSFIT_ONLY", "MCC_MAX"],
        }
        for key, expected in exact.items():
            if policy.get(key) != expected:
                errors.append(f"Phase 13 contract policy differs: {key}.")
        if policy.get("outer_validation") != {
            "after_phase13_freeze_only": True,
            "maximum_new_validation_score_sets": 3,
        }:
            errors.append("Phase 13 outer-validation policy differs.")
        if policy.get("test") != {"forbidden_until_phase": 15}:
            errors.append("Phase 13 TEST policy differs.")
        if settings.tracks != TRACKS or settings.calibration_methods != CALIBRATION_METHODS:
            errors.append("Phase 13 configuration inventory drifted.")
        if settings.ensemble_weights != ENSEMBLE_WEIGHTS:
            errors.append("Phase 13 configured ensemble weights drifted.")
    except Exception as exc:
        errors.append(str(exc))
    return {
        "status": "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS"),
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "contract_version": payload.get("phase13", {}).get("version"),
        "contract_checksum": checksum,
        "contract": payload,
    }


__all__ = [
    "load_calibration_ensemble_contract",
    "validate_calibration_ensemble_contract",
]
