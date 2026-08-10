"""Versioned Phase 8 text-feature contract and offline validator."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from ..database.schema_contract import load_schema_contract
from ..feature_mart.mart_contract import load_mart_contract
from ..paths import discover_repository_root
from ..policy.loader import load_phase4_contracts
from ..splits.split_contract import load_split_contract
from ..structured_features.contract import load_structured_feature_contract
from .models import TextFeatureError

PHASE8_CONTRACT_NAME = "text_feature_contract_v1.yaml"
REQUIRED_KEYS = (
    "contract_version",
    "created_at",
    "schema_contract_checksum",
    "target_contract_checksum",
    "feature_policy_checksum",
    "leakage_policy_checksum",
    "phase5_mart_contract_checksum",
    "phase6_split_contract_checksum",
    "phase7_structured_feature_contract_checksum",
    "feature_grain",
    "prediction_reference",
    "approved_text_sources",
    "prohibited_text_sources",
    "text_normalization_policy",
    "historical_windows",
    "document_construction_policy",
    "lexical_feature_policy",
    "fitted_transform_policy",
    "target_independence_policy",
    "test_lock_policy",
    "dimension_versioning_warning",
    "artifact_layout",
    "development_status",
)


def phase8_contract_path(project_root: Path | None = None) -> Path:
    """Return the version-controlled Phase 8 contract path."""

    return discover_repository_root(project_root) / "contracts" / PHASE8_CONTRACT_NAME


def phase8_contract_checksum(path: Path) -> str:
    """Return the exact contract file SHA-256."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_text_feature_contract(
    project_root: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Load the Phase 8 contract and exact file checksum."""

    path = phase8_contract_path(project_root)
    if not path.is_file():
        raise TextFeatureError(f"Phase 8 contract is missing: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TextFeatureError(f"Could not read Phase 8 contract: {path}") from exc
    if not isinstance(payload, dict):
        raise TextFeatureError("Phase 8 contract must be a top-level mapping.")
    missing = sorted(set(REQUIRED_KEYS) - set(payload))
    if missing:
        raise TextFeatureError("Phase 8 contract is missing: " + ", ".join(missing))
    return payload, phase8_contract_checksum(path)


def validate_text_feature_contract(project_root: Path | None = None) -> dict[str, Any]:
    """Validate Phase 8 policy and all authoritative upstream checksums offline."""

    root = discover_repository_root(project_root)
    errors: list[str] = []
    warnings: list[str] = []
    try:
        contract, checksum = load_text_feature_contract(root)
        _, schema_checksum = load_schema_contract(root)
        phase4 = load_phase4_contracts(root)
        _, mart_checksum = load_mart_contract(root)
        _, split_checksum = load_split_contract(root)
        _, phase7_checksum = load_structured_feature_contract(root)
    except Exception as exc:
        return {"status": "BLOCKED", "valid": False, "errors": [str(exc)], "warnings": []}
    expected = {
        "schema_contract_checksum": schema_checksum,
        "target_contract_checksum": phase4.target_checksum,
        "feature_policy_checksum": phase4.feature_policy_checksum,
        "leakage_policy_checksum": phase4.leakage_checksum,
        "phase5_mart_contract_checksum": mart_checksum,
        "phase6_split_contract_checksum": split_checksum,
        "phase7_structured_feature_contract_checksum": phase7_checksum,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            errors.append(f"Phase 8 {key} does not match the authoritative contract.")
    if contract.get("contract_version") != "1.0.0":
        errors.append("Phase 8 contract_version must be 1.0.0.")
    if contract.get("feature_grain") != "one row = one eligible warranty claim":
        errors.append("Phase 8 feature grain is not the approved claim grain.")
    approved = contract.get("approved_text_sources")
    if not isinstance(approved, list) or len(approved) != 1:
        errors.append("Phase 8 must define exactly one approved text source.")
    else:
        source = approved[0]
        if (
            not isinstance(source, dict)
            or source.get("source") != "prior_failure__failure_description"
        ):
            errors.append("Phase 8 approved source must be prior_failure__failure_description.")
        if not isinstance(source, dict) or source.get("policy") != "ALLOW_HISTORICAL_POC":
            errors.append("Phase 8 approved source must be ALLOW_HISTORICAL_POC.")
    windows = contract.get("historical_windows", {})
    if not isinstance(windows, dict) or windows.get("fixed_months") != ["6m", "12m", "24m"]:
        errors.append("Phase 8 fixed windows must be exactly 6m, 12m, and 24m.")
    if not isinstance(windows, dict) or windows.get("include_all_history") is not True:
        errors.append("Phase 8 all-history document is required.")
    document = contract.get("document_construction_policy", {})
    if not isinstance(document, dict) or document.get("separator") != " [SEP] ":
        errors.append("Phase 8 document separator must be exactly ' [SEP] '.")
    if not isinstance(document, dict) or document.get("preserve_repeated_descriptions") is not True:
        errors.append("Phase 8 must preserve repeated descriptions.")
    normalization = contract.get("text_normalization_policy", {})
    if not isinstance(normalization, dict) or normalization.get("unicode_form") != "NFKC":
        errors.append("Phase 8 normalization must use NFKC.")
    fitted = contract.get("fitted_transform_policy", {})
    if isinstance(fitted, dict):
        for name, value in fitted.items():
            if name != "vocabulary_learning" and value is not False:
                errors.append(f"Phase 8 fitted transform {name} must be false.")
    target_policy = contract.get("target_independence_policy", {})
    if (
        not isinstance(target_policy, dict)
        or target_policy.get("target_column_excluded") is not True
    ):
        errors.append("Phase 8 target exclusion policy is not enabled.")
    test_lock = contract.get("test_lock_policy", {})
    if not isinstance(test_lock, dict) or test_lock.get("consume_existing_split") is not True:
        errors.append("Phase 8 must consume the existing split.")
    warning = contract.get("dimension_versioning_warning", {})
    if (
        not isinstance(warning, dict)
        or warning.get("code") != "UNVERSIONED_FAILURE_DESCRIPTION_DIMENSION"
    ):
        errors.append("The unversioned failure-description warning must be carried forward.")
    status = contract.get("development_status", {})
    if not isinstance(status, dict) or status.get("production_approved") is not False:
        errors.append("Phase 8 production_approved must remain false.")
    warnings.append("UNVERSIONED_FAILURE_DESCRIPTION_DIMENSION: real-data reapproval is required.")
    warnings.append(
        "Phase 8 is synthetic POC only; business target definition remains unconfirmed."
    )
    return {
        "status": "BLOCKED" if errors else "PASS WITH WARNINGS",
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "contract_version": contract.get("contract_version"),
        "contract_checksum": checksum,
        "contract": contract,
    }
