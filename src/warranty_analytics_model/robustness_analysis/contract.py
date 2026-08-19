"""Offline Phase 14 contract verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ..paths import discover_repository_root
from .config import LOCKED_CONFIGURATION, PHASE14_SEED, PHASE14_VERSION, configuration_sha256


def load_phase14_contract(project_root: Path | None = None) -> tuple[dict[str, Any], str]:
    root = discover_repository_root(project_root)
    path = root / "contracts" / "robustness_error_analysis_v1.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read Phase 14 contract: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Phase 14 contract must be a mapping.")
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    return payload, digest


def phase14_contract_check(project_root: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        payload, contract_sha = load_phase14_contract(project_root)
    except (OSError, ValueError) as exc:
        return {
            "phase": 14,
            "valid": False,
            "status": "BLOCKED",
            "errors": [str(exc)],
            "warnings": [],
        }
    root = payload.get("phase14_robustness_error_analysis")
    if not isinstance(root, dict):
        errors.append("Phase 14 contract root is missing.")
        root = {}
    expected = {
        "version": PHASE14_VERSION,
        "phase": 14,
        "required_phase13_status": "HARDENED_PASS",
        "model_retraining": "prohibited",
        "feature_changes": "prohibited",
        "hyperparameter_retuning": "prohibited",
        "class_weight_changes": "prohibited",
        "calibration_changes": "prohibited",
        "ensemble_changes": "prohibited",
        "threshold_reoptimization": "prohibited",
        "model_reselection": "prohibited",
        "validation_use": "DIAGNOSTIC_ONLY",
        "slice_definition_target_access": "PROHIBITED",
        "prediction_tolerance": 1.0e-10,
    }
    for key, value in expected.items():
        if root.get(key) != value:
            errors.append(f"Phase 14 contract drifted: {key}.")
    bootstrap = root.get("bootstrap")
    if bootstrap != {
        "seed": PHASE14_SEED,
        "overall_replicates": 2000,
        "material_slice_replicates": 1000,
        "confidence_level": 0.95,
        "method": "STRATIFIED_PERCENTILE",
    }:
        errors.append("Phase 14 bootstrap contract drifted.")
    if root.get("test") != {"forbidden_until_phase": 15}:
        errors.append("Phase 14 TEST seal contract drifted.")
    if (
        configuration_sha256()
        != hashlib.sha256(
            json.dumps(LOCKED_CONFIGURATION, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        errors.append("Phase 14 configuration hash is not self-consistent.")
    return {
        "phase": 14,
        "valid": not errors,
        "status": "PASS" if not errors else "BLOCKED",
        "contract_version": PHASE14_VERSION,
        "contract_sha256": contract_sha,
        "configuration_sha256": configuration_sha256(),
        "errors": errors,
        "warnings": warnings,
    }


__all__ = ["load_phase14_contract", "phase14_contract_check"]
