"""Fail-closed Phase 11 policy contract validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ..paths import discover_repository_root
from .config import TRACKS, FeatureSelectionError, load_feature_selection_settings

CONTRACT_VERSION = "phase11_feature_selection_ablation_v1"
REQUIRED_PHASE9_RUN_ID = "20260811T_PHASE9_FINAL"
REQUIRED_PHASE10_RUN_ID = "20260811T_PHASE10"
REQUIRED_FEATURE_HASHES = {
    "T1": "4a8de5a69ce72bf6059f9856252d68465d464fecfb56242f5fa55646edae7b89",
    "T3": "13859692eec0494879712b6ac66a3ce06f64cd75ff93de5c80e2c0a67b701738",
}


def load_feature_selection_contract(project_root: Path | None = None) -> tuple[dict[str, Any], str]:
    root = discover_repository_root(project_root)
    path = root / "contracts" / "feature_selection_ablation_v1.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FeatureSelectionError(f"Could not read Phase 11 contract: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("phase11"), dict):
        raise FeatureSelectionError("Phase 11 contract must contain a phase11 mapping.")
    checksum = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return payload, checksum


def validate_feature_selection_contract(project_root: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    payload: dict[str, Any] = {}
    checksum: str | None = None
    try:
        root = discover_repository_root(project_root)
        payload, checksum = load_feature_selection_contract(root)
        settings = load_feature_selection_settings(root)
        policy = payload["phase11"]
        exact = {
            "phase": 11,
            "version": CONTRACT_VERSION,
            "required_phase9_run_id": REQUIRED_PHASE9_RUN_ID,
            "required_phase10_run_id": REQUIRED_PHASE10_RUN_ID,
            "required_phase9_hardened_status": "HARDENED_PASS",
            "required_phase10_hardened_status": "HARDENED_PASS",
            "required_phase10_contract_version": "phase10_catboost_optimization_v2",
            "parent_tracks": {"T1": "E1", "T3": "E3"},
            "required_feature_set_hashes": REQUIRED_FEATURE_HASHES,
            "parent_feature_counts": {"T1": 301, "T3": 536},
            "feature_addition": "prohibited",
            "feature_removal": "allowed",
            "hyperparameter_retuning": "prohibited",
            "class_weighting": "prohibited",
            "resampling": "prohibited",
            "early_stopping": "prohibited",
            "threshold_tuning": "prohibited",
            "calibration": "prohibited",
            "ensembling": "prohibited",
            "search_target_access": "TRAIN_ONLY",
            "primary_metric": "mean_average_precision",
        }
        for key, expected in exact.items():
            if policy.get(key) != expected:
                errors.append(f"Phase 11 contract policy differs: {key}.")
        if policy.get("outer_validation") != {
            "after_selection_freeze_only": True,
            "new_candidates_maximum": 2,
        }:
            errors.append("Phase 11 outer validation policy is not exactly locked.")
        if policy.get("test_target_access") != {
            "forbidden_until_phase": 15,
            "target_rows_loaded": 0,
            "predictions_created": 0,
            "metrics_computed": False,
        }:
            errors.append("Phase 11 TEST policy is not exactly locked.")
        if settings.tracks != TRACKS:
            errors.append("Phase 11 configuration tracks drifted.")
    except Exception as exc:
        errors.append(str(exc))
    return {
        "status": "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS"),
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "contract_version": payload.get("phase11", {}).get("version"),
        "contract_checksum": checksum,
        "contract": payload,
    }


__all__ = [
    "CONTRACT_VERSION",
    "REQUIRED_FEATURE_HASHES",
    "load_feature_selection_contract",
    "validate_feature_selection_contract",
]

