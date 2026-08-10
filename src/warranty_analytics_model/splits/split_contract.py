"""Load and validate the versioned Phase 6 split contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ..database.schema_contract import load_schema_contract
from ..feature_mart.mart_contract import load_mart_contract, validate_mart_contract
from ..paths import discover_repository_root
from ..policy.loader import load_phase4_contracts
from .config import load_split_settings, validate_split_settings
from .models import ContractValidationResult, SplitContract, SplitError

SPLIT_CONTRACT_NAME = "claim_split_v1.yaml"


def split_contract_checksum(path: Path) -> str:
    """Return the exact file checksum for the versioned split contract."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise SplitError(f"Could not read Phase 6 split contract: {path}") from exc


def _contract_path(project_root: Path | None, path: Path | None) -> Path:
    return path or discover_repository_root(project_root) / "contracts" / SPLIT_CONTRACT_NAME


def load_split_contract(
    project_root: Path | None = None,
    *,
    path: Path | None = None,
) -> tuple[SplitContract, str]:
    """Load the YAML split contract and return its exact checksum."""

    contract_path = _contract_path(project_root, path)
    if not contract_path.is_file():
        raise SplitError(f"Phase 6 split contract is missing: {contract_path}")
    try:
        payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        contract = SplitContract.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise SplitError(f"Invalid Phase 6 split contract: {contract_path}") from exc
    return contract, split_contract_checksum(contract_path)


def validate_split_contract(
    contract: SplitContract,
    *,
    mart_contract: Any,
    mart_contract_checksum: str,
    schema_contract_checksum: str,
    phase4_bundle: Any,
    settings: Any,
    split_contract_checksum_value: str,
) -> ContractValidationResult:
    """Validate contract semantics and current Phase 4/5 checksum compatibility."""

    errors: list[str] = []
    warnings: list[str] = []
    fractions = {str(key): float(value) for key, value in contract.requested_fractions.items()}
    expected_names = {"TRAIN", "VALIDATION", "TEST"}
    if set(fractions) != expected_names:
        errors.append("requested_fractions must define exactly TRAIN, VALIDATION, and TEST.")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        errors.append("Split contract fractions must sum to 1.0.")
    if contract.split_strategy != "chronological":
        errors.append("Split contract strategy must be chronological.")
    if contract.tie_breaking_rule != "earlier_date":
        errors.append("Split contract tie-breaking rule must be earlier_date.")
    same_day = contract.same_day_policy
    if same_day.get("preserve_same_date") is not True:
        errors.append("Split contract must enable same-date preservation.")
    algorithm = contract.boundary_algorithm
    if algorithm.get("target_independent") is not True:
        errors.append("Boundary algorithm must declare target_independent: true.")
    if algorithm.get("target_column_used") is not False:
        errors.append("Boundary algorithm must explicitly set target_column_used: false.")
    algorithm_name = str(algorithm.get("name", "")).casefold()
    if "date" not in algorithm_name or "count" not in algorithm_name:
        errors.append("Boundary algorithm must be based on date groups and claim counts.")
    algorithm_inputs = algorithm.get("inputs", [])
    if isinstance(algorithm_inputs, list) and any(
        "target" in str(item).casefold() for item in algorithm_inputs
    ):
        errors.append("Boundary algorithm inputs must not include the stored target.")
    if contract.prediction_reference not in {"claim__claim_date", "claim_date"}:
        errors.append("Phase 6 prediction_reference must be the Phase 5 claim date.")
    settings_errors = validate_split_settings(settings)
    errors.extend(settings_errors)
    configured = {key: float(value) for key, value in settings.requested_fractions.items()}
    if fractions != configured:
        errors.append("Split contract fractions do not match configs/splits.yaml.")
    if contract.input_mart_contract_version != mart_contract.version:
        errors.append(
            "Split contract Phase 5 mart version does not match the current mart contract."
        )
    if contract.input_mart_contract_checksum != mart_contract_checksum:
        errors.append(
            "Split contract Phase 5 mart checksum does not match the current mart contract."
        )
    if contract.input_schema_contract_checksum != schema_contract_checksum:
        errors.append("Split contract schema checksum does not match the current schema contract.")
    if contract.input_target_contract_checksum != phase4_bundle.target_checksum:
        errors.append("Split contract target checksum does not match Phase 4.")
    if contract.input_feature_policy_checksum != phase4_bundle.feature_policy_checksum:
        errors.append("Split contract feature-policy checksum does not match Phase 4.")
    if contract.input_leakage_policy_checksum != phase4_bundle.leakage_checksum:
        errors.append("Split contract leakage-policy checksum does not match Phase 4.")
    test_access = contract.test_access_policy
    if int(test_access.get("allowed_first_target_evaluation_phase", 0)) != 15:
        errors.append("Test access policy must reserve the first target evaluation for Phase 15.")
    if test_access.get("phase_9_to_14_target_evaluation_for_development") is not False:
        errors.append(
            "Test access policy must prohibit target-based Phase 9-14 development evaluation."
        )
    if contract.group_exposure_policy.get("enabled") is not True:
        errors.append("Group exposure policy must be present and enabled.")
    scenario = contract.scenario_fingerprint_policy
    if scenario.get("fingerprint_clean_cohort_defined") is not True:
        errors.append("Scenario fingerprint policy must define a fingerprint-clean cohort.")
    if scenario.get("overlap_severity") != "WARNING":
        errors.append("Scenario fingerprint overlap severity must be WARNING.")
    if contract.artifact_layout.get("split_assignments") != "split_assignments.parquet":
        errors.append("Split contract artifact layout must define split_assignments.parquet.")
    if contract.validation_policy.get("claim_coverage_blocking") is not True:
        errors.append("Split contract validation policy must make claim coverage blocking.")
    if contract.development_status.get("development_mode") != "synthetic_poc":
        errors.append("Split contract must carry forward development_mode: synthetic_poc.")
    if contract.development_status.get("production_approved") is not False:
        errors.append("Split contract must not claim production approval.")
    if contract.development_status.get("real_data_reapproval_required") is not True:
        errors.append("Split contract must require real-data reapproval.")
    if contract.development_status.get("business_target_definition_confirmed") is not False:
        errors.append("Split contract must carry forward the unconfirmed business target status.")
    if contract.development_status.get("precise_submission_timestamp_available") is not False:
        errors.append(
            "Split contract must record that precise submission timestamps are unavailable."
        )
    return ContractValidationResult(
        valid=not errors,
        errors=list(dict.fromkeys(errors)),
        warnings=list(dict.fromkeys(warnings)),
        split_contract_checksum=split_contract_checksum_value,
        mart_contract_checksum=mart_contract_checksum,
        requested_fractions=fractions,
    )


def validate_current_split_contract(project_root: Path | None = None) -> ContractValidationResult:
    """Run the complete offline contract gate against current Phase 4/5 contracts."""

    root = discover_repository_root(project_root)
    contract, checksum = load_split_contract(root)
    settings = load_split_settings(root)
    schema_contract, schema_checksum = load_schema_contract(root)
    phase4_bundle = load_phase4_contracts(root)
    mart_contract, mart_checksum = load_mart_contract(root)
    mart_plan = validate_mart_contract(
        schema_contract,
        phase4_bundle,
        schema_contract_checksum=schema_checksum,
        contract=mart_contract,
        contract_checksum=mart_checksum,
    )
    if not mart_plan.valid:
        raise SplitError("Phase 5 mart contract is invalid: " + "; ".join(mart_plan.errors))
    return validate_split_contract(
        contract,
        mart_contract=mart_contract,
        mart_contract_checksum=mart_checksum,
        schema_contract_checksum=schema_checksum,
        phase4_bundle=phase4_bundle,
        settings=settings,
        split_contract_checksum_value=checksum,
    )
