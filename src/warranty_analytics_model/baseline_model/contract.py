"""Versioned Phase 9 baseline-model contract and offline validation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from ..feature_mart.mart_contract import load_mart_contract
from ..paths import discover_repository_root
from ..policy.loader import load_phase4_contracts
from ..splits.split_contract import load_split_contract
from ..structured_features.contract import load_structured_feature_contract
from ..text_features.contract import load_text_feature_contract
from .config import load_baseline_settings
from .models import BaselineModelError

CONTRACT_NAME = "baseline_model_contract_v1.yaml"
EXPERIMENT_IDS = ("E0", "E1", "E2", "E3", "E4")
REQUIRED_KEYS = {
    "contract_version",
    "created_at",
    "target_contract_checksum",
    "feature_policy_checksum",
    "leakage_policy_checksum",
    "phase5_mart_contract_checksum",
    "phase6_split_contract_checksum",
    "phase7_structured_feature_contract_checksum",
    "phase8_text_feature_contract_checksum",
    "prediction_grain",
    "target",
    "allowed_target_splits",
    "prohibited_target_splits",
    "baseline_experiments",
    "primary_metric",
    "secondary_metrics",
    "fixed_catboost_parameters",
    "categorical_adapter_policy",
    "numeric_adapter_policy",
    "text_adapter_policy",
    "class_imbalance_policy",
    "threshold_policy",
    "calibration_policy",
    "feature_selection_policy",
    "test_access_policy",
    "reproducibility_policy",
    "artifact_layout",
    "development_status",
}


def contract_path(project_root: Path | None = None) -> Path:
    return discover_repository_root(project_root) / "contracts" / CONTRACT_NAME


def contract_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_baseline_contract(project_root: Path | None = None) -> tuple[dict[str, Any], str]:
    path = contract_path(project_root)
    if not path.is_file():
        raise BaselineModelError(f"Phase 9 contract is missing: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BaselineModelError(f"Could not read Phase 9 contract: {path}") from exc
    if not isinstance(payload, dict):
        raise BaselineModelError("Phase 9 contract must be a top-level mapping.")
    missing = sorted(REQUIRED_KEYS - set(payload))
    if missing:
        raise BaselineModelError("Phase 9 contract is missing: " + ", ".join(missing))
    return payload, contract_checksum(path)


def validate_baseline_contract(project_root: Path | None = None) -> dict[str, Any]:
    """Validate fixed experiments, policies, settings, and upstream checksums offline."""

    root = discover_repository_root(project_root)
    errors: list[str] = []
    warnings: list[str] = []
    try:
        contract, checksum = load_baseline_contract(root)
        phase4 = load_phase4_contracts(root)
        _, phase5_checksum = load_mart_contract(root)
        _, phase6_checksum = load_split_contract(root)
        _, phase7_checksum = load_structured_feature_contract(root)
        _, phase8_checksum = load_text_feature_contract(root)
        settings = load_baseline_settings(root)
    except Exception as exc:
        return {"status": "BLOCKED", "valid": False, "errors": [str(exc)], "warnings": []}
    expected = {
        "target_contract_checksum": phase4.target_checksum,
        "feature_policy_checksum": phase4.feature_policy_checksum,
        "leakage_policy_checksum": phase4.leakage_checksum,
        "phase5_mart_contract_checksum": phase5_checksum,
        "phase6_split_contract_checksum": phase6_checksum,
        "phase7_structured_feature_contract_checksum": phase7_checksum,
        "phase8_text_feature_contract_checksum": phase8_checksum,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            errors.append(f"Phase 9 {key} does not match the authoritative contract.")
    if contract.get("contract_version") != "1.0.0":
        errors.append("Phase 9 contract_version must be 1.0.0.")
    if contract.get("prediction_grain") != "one row = one eligible warranty claim":
        errors.append("Phase 9 prediction grain changed.")
    if contract.get("allowed_target_splits") != ["TRAIN", "VALIDATION"]:
        errors.append("Phase 9 target access must be TRAIN and VALIDATION only.")
    if contract.get("prohibited_target_splits") != ["TEST"]:
        errors.append("Phase 9 must explicitly prohibit TEST target access.")
    experiments = contract.get("baseline_experiments")
    experiment_ids = (
        tuple(item.get("id") for item in experiments if isinstance(item, dict))
        if isinstance(experiments, list)
        else ()
    )
    if experiment_ids != EXPERIMENT_IDS:
        errors.append("Phase 9 experiments must be exactly E0, E1, E2, E3, and E4.")
    if contract.get("primary_metric") != "average_precision":
        errors.append("Phase 9 primary metric must be average_precision.")
    fixed = contract.get("fixed_catboost_parameters", {})
    if not isinstance(fixed, dict) or any(
        fixed.get(key) != value for key, value in settings.catboost_parameters.items()
    ):
        errors.append("Phase 9 contract and configuration CatBoost parameters differ.")
    imbalance = contract.get("class_imbalance_policy", {})
    if not isinstance(imbalance, dict) or any(value != "none" for value in imbalance.values()):
        errors.append("Phase 9 class weighting and resampling must remain none.")
    threshold = contract.get("threshold_policy", {})
    if not isinstance(threshold, dict) or threshold.get("descriptive_threshold") != 0.5:
        errors.append("Phase 9 descriptive threshold must be fixed at 0.5.")
    if not isinstance(threshold, dict) or threshold.get("optimization") != "none":
        errors.append("Phase 9 threshold optimization is prohibited.")
    test_policy = contract.get("test_access_policy", {})
    if not isinstance(test_policy, dict) or test_policy.get("target_access_allowed") is not False:
        errors.append("Phase 9 TEST target access must be false.")
    if not isinstance(test_policy, dict) or test_policy.get("first_allowed_target_phase") != 15:
        errors.append("The first allowed TEST target phase must remain Phase 15.")
    development = contract.get("development_status", {})
    if not isinstance(development, dict) or development.get("production_approved") is not False:
        errors.append("Phase 9 production_approved must remain false.")
    warnings.extend(
        [
            "SYNTHETIC_POC: development metrics are not production performance.",
            "BUSINESS_TARGET_UNCONFIRMED: real-data target approval remains open.",
            "UNVERSIONED_FAILURE_DESCRIPTION_DIMENSION: production reapproval is required.",
        ]
    )
    return {
        "status": "BLOCKED" if errors else "PASS WITH WARNINGS",
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": warnings,
        "contract_version": contract.get("contract_version"),
        "contract_checksum": checksum,
        "contract": contract,
    }
