"""Offline, fail-closed Phase 15 contract verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ..paths import discover_repository_root
from .config import LOCKED_CONFIGURATION, PHASE15_SEED, PHASE15_VERSION, configuration_sha256


def load_phase15_contract(project_root: Path | None = None) -> tuple[dict[str, Any], str]:
    root = discover_repository_root(project_root)
    path = root / "contracts" / "final_test_evaluation_v1.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read Phase 15 contract: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Phase 15 contract must be a mapping.")
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    return payload, digest


def phase15_contract_check(project_root: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        payload, contract_sha = load_phase15_contract(project_root)
    except (OSError, ValueError) as exc:
        return {
            "phase": 15,
            "valid": False,
            "status": "BLOCKED",
            "errors": [str(exc)],
            "warnings": [],
        }
    root = payload.get("phase15_final_test_evaluation")
    if not isinstance(root, dict):
        errors.append("Phase 15 contract root is missing.")
        root = {}
    expected = {
        "version": PHASE15_VERSION,
        "phase": 15,
        "required_phase14_status": ["HARDENED_PASS", "HARDENED_PASS_WITH_WARNINGS"],
        "required_phase15_readiness": ["READY", "READY_WITH_WARNINGS"],
        "final_model_policy": "REUSE_FROZEN_PHASE14_CHAMPION",
        "model_retraining": "prohibited",
        "train_validation_refit": "prohibited",
        "model_reselection": "prohibited",
        "feature_changes": "prohibited",
        "hyperparameter_changes": "prohibited",
        "class_weight_changes": "prohibited",
        "calibration_changes": "prohibited",
        "ensemble_changes": "prohibited",
        "threshold_reoptimization": "prohibited",
        "prediction_tolerance": 1.0e-10,
    }
    for key, value in expected.items():
        if root.get(key) != value:
            errors.append(f"Phase 15 contract drifted: {key}.")
    bootstrap = root.get("bootstrap")
    if bootstrap != {
        "seed": PHASE15_SEED,
        "replicates": 2000,
        "confidence_level": 0.95,
        "method": "STRATIFIED_PERCENTILE",
    }:
        errors.append("Phase 15 bootstrap contract drifted.")
    if root.get("top_k") != [0.05, 0.10, 0.20, 0.30]:
        errors.append("Phase 15 Top-K contract drifted.")
    test_policy = root.get("test")
    if (
        not isinstance(test_policy, dict)
        or test_policy.get("first_allowed_target_access_phase") != 15
    ):
        errors.append("Phase 15 TEST target-access contract drifted.")
    if (
        configuration_sha256()
        != hashlib.sha256(
            json.dumps(LOCKED_CONFIGURATION, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        errors.append("Phase 15 configuration hash is not self-consistent.")
    return {
        "phase": 15,
        "valid": not errors,
        "status": "PASS" if not errors else "BLOCKED",
        "contract_version": PHASE15_VERSION,
        "contract_sha256": contract_sha,
        "configuration_sha256": configuration_sha256(),
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
    }


__all__ = ["load_phase15_contract", "phase15_contract_check"]
