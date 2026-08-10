"""Versioned Phase 7 contract loading and fail-closed validation."""

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
from .models import StructuredFeatureError

PHASE7_CONTRACT_NAME = "structured_feature_contract_v1.yaml"
REQUIRED_KEYS = (
    "contract_version",
    "created_at",
    "schema_contract_checksum",
    "target_contract_checksum",
    "feature_policy_checksum",
    "leakage_policy_checksum",
    "phase5_mart_contract_checksum",
    "phase6_split_contract_checksum",
    "feature_grain",
    "prediction_reference",
    "direct_feature_policy",
    "structured_feature_families",
    "feature_tiers",
    "historical_windows",
    "aggregation_rules",
    "trend_rules",
    "recency_rules",
    "ratio_rules",
    "categorical_rules",
    "missingness_rules",
    "target_independence_policy",
    "test_lock_policy",
    "artifact_layout",
    "deferred_sources",
    "development_status",
)


def phase7_contract_path(project_root: Path | None = None) -> Path:
    """Return the version-controlled Phase 7 contract path."""

    return discover_repository_root(project_root) / "contracts" / PHASE7_CONTRACT_NAME


def phase7_contract_checksum(path: Path) -> str:
    """Return the exact file checksum used in manifests and reports."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_structured_feature_contract(
    project_root: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Load the Phase 7 YAML contract and its exact SHA-256."""

    path = phase7_contract_path(project_root)
    if not path.is_file():
        raise StructuredFeatureError(f"Phase 7 contract is missing: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StructuredFeatureError(f"Could not read Phase 7 contract: {path}") from exc
    if not isinstance(payload, dict):
        raise StructuredFeatureError("Phase 7 contract must be a top-level mapping.")
    missing = sorted(set(REQUIRED_KEYS) - set(payload))
    if missing:
        raise StructuredFeatureError("Phase 7 contract is missing: " + ", ".join(missing))
    return payload, phase7_contract_checksum(path)


def validate_structured_feature_contract(
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Validate Phase 7 policy, tiers, formulas, and source exclusions offline."""

    root = discover_repository_root(project_root)
    errors: list[str] = []
    warnings: list[str] = []
    try:
        contract, checksum = load_structured_feature_contract(root)
        schema, schema_checksum = load_schema_contract(root)
        phase4 = load_phase4_contracts(root)
        mart, mart_checksum = load_mart_contract(root)
        split, split_checksum = load_split_contract(root)
    except Exception as exc:
        return {"status": "BLOCKED", "valid": False, "errors": [str(exc)], "warnings": []}
    expected = {
        "schema_contract_checksum": schema_checksum,
        "target_contract_checksum": phase4.target_checksum,
        "feature_policy_checksum": phase4.feature_policy_checksum,
        "leakage_policy_checksum": phase4.leakage_checksum,
        "phase5_mart_contract_checksum": mart_checksum,
        "phase6_split_contract_checksum": split_checksum,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            errors.append(f"Phase 7 {key} does not match the authoritative contract.")
    if contract.get("contract_version") != "1.0.0":
        errors.append("Phase 7 contract_version must be 1.0.0.")
    if contract.get("feature_grain") != "one row = one eligible warranty claim":
        errors.append("Phase 7 feature grain is not the approved claim grain.")
    tiers = contract.get("feature_tiers")
    if not isinstance(tiers, dict) or set(tiers) != {"CORE", "EXTENDED"}:
        errors.append("Phase 7 must define exactly CORE and EXTENDED safe tiers.")
    if "RESTRICTED_EXPERIMENTAL" in str(contract.get("feature_tiers", {})):
        errors.append("RESTRICTED_EXPERIMENTAL cannot be a Phase 7 feature tier.")
    windows = contract.get("historical_windows")
    if windows != ["3m", "6m", "12m", "24m", "all"]:
        errors.append("Phase 7 historical windows must be 3m, 6m, 12m, 24m, and all.")
    independence = contract.get("target_independence_policy", {})
    if not isinstance(independence, dict) or independence.get("target_column_excluded") is not True:
        errors.append("Phase 7 target-independence policy must exclude the target column.")
    if independence.get("globally_fitted_transformations") not in ([], None):
        errors.append("Phase 7 cannot contain globally fitted transformations.")
    deferred = contract.get("deferred_sources", [])
    if not any("failure_description" in str(item) for item in deferred):
        errors.append("failure_description must be explicitly deferred to Phase 8.")
    if not any("repair_history_index" in str(item) for item in deferred):
        errors.append("repair_history_index must be explicitly deferred as control-only.")
    for forbidden in (
        "production_batch_id",
        "component_lot_no",
        "supplier_key",
        "service_center_key",
        "truck_key",
        "VIN",
        "serial",
        "technician",
        "inspector",
        "scenario fingerprint",
        "group hash",
    ):
        if forbidden.casefold() in str(contract.get("structured_feature_families", {})).casefold():
            errors.append(f"Restricted identifier appears in feature families: {forbidden}")
    if not schema.tables:
        warnings.append("Schema contract contains no included tables.")
    status = "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS")
    return {
        "status": status,
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "contract_version": contract.get("contract_version"),
        "contract_checksum": checksum,
        "schema_contract_checksum": schema_checksum,
        "phase5_mart_contract_checksum": mart_checksum,
        "phase6_split_contract_checksum": split_checksum,
    }
