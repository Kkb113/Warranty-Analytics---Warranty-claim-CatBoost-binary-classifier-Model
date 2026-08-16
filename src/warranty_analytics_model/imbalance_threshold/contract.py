"""Offline validation for the locked Phase 12 policy contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ..paths import discover_repository_root
from .config import (
    PHASE12_VERSION,
    STRATEGY_IDS,
    TRACKS,
    ImbalanceThresholdError,
    load_imbalance_threshold_settings,
)

CONTRACT_VERSION = PHASE12_VERSION


def load_imbalance_threshold_contract(
    project_root: Path | None = None,
) -> tuple[dict[str, Any], str]:
    root = discover_repository_root(project_root)
    path = root / "contracts" / "imbalance_threshold_v1.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ImbalanceThresholdError(f"Could not read Phase 12 contract: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("phase12"), dict):
        raise ImbalanceThresholdError("Phase 12 contract must contain a phase12 mapping.")
    checksum = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return payload, checksum


def validate_imbalance_threshold_contract(
    project_root: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    payload: dict[str, Any] = {}
    checksum: str | None = None
    try:
        root = discover_repository_root(project_root)
        payload, checksum = load_imbalance_threshold_contract(root)
        settings = load_imbalance_threshold_settings(root)
        policy = payload["phase12"]
        exact = {
            "phase": 12,
            "version": CONTRACT_VERSION,
            "required_phase9_hardened_status": "HARDENED_PASS",
            "required_phase10_hardened_status": "HARDENED_PASS",
            "required_phase11_hardened_status": "HARDENED_PASS",
            "parent_tracks": list(TRACKS),
            "feature_changes": "prohibited",
            "hyperparameter_retuning": "prohibited",
            "imbalance_methods": {"weighting": "allowed", "resampling": "prohibited"},
            "allowed_strategies": list(STRATEGY_IDS),
            "calibration": "prohibited_until_phase_13",
            "ensembling": "prohibited_until_phase_13",
            "optimization_target_access": "TRAIN_ONLY",
            "model_selection_primary_metric": "mean_average_precision",
            "threshold_selection_primary_metric": "matthews_correlation_coefficient",
            "outer_validation": {
                "after_phase12_freeze_only": True,
                "maximum_new_weighted_candidates": 2,
            },
            "test": {"forbidden_until_phase": 15},
        }
        for key, expected in exact.items():
            if policy.get(key) != expected:
                errors.append(f"Phase 12 contract policy differs: {key}.")
        if settings.tracks != TRACKS or settings.strategy_ids != STRATEGY_IDS:
            errors.append("Phase 12 configuration tracks or strategies drifted.")
    except Exception as exc:
        errors.append(str(exc))
    return {
        "status": "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS"),
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "contract_version": payload.get("phase12", {}).get("version"),
        "contract_checksum": checksum,
        "contract": payload,
    }


__all__ = [
    "CONTRACT_VERSION",
    "load_imbalance_threshold_contract",
    "validate_imbalance_threshold_contract",
]
