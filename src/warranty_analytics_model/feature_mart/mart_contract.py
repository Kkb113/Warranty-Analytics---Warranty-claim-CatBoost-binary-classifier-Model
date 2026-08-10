"""Load and fail-closed validate the Phase 5 mart contract."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ..database.models import SchemaContract
from ..paths import discover_repository_root
from ..policy.coverage import build_future_allowlists
from ..policy.models import Phase4ContractBundle
from ..policy.validator import validate_historical_source_rules, validate_phase4_contracts
from .models import (
    FeatureMartError,
    FieldMapping,
    MartContract,
    MartPlanValidationResult,
)

MART_CONTRACT_NAME = "claim_feature_mart_v1.yaml"


def mart_contract_checksum(path: Path) -> str:
    """Return the exact SHA-256 checksum of a versioned mart contract."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise FeatureMartError(f"Could not read Phase 5 mart contract: {path}") from exc


def _contract_path(project_root: Path | None, path: Path | None) -> Path:
    if path is not None:
        return path
    return discover_repository_root(project_root) / "contracts" / MART_CONTRACT_NAME


def load_mart_contract(
    project_root: Path | None = None,
    *,
    path: Path | None = None,
) -> tuple[MartContract, str]:
    """Load the YAML mart contract and return it with its exact file checksum."""

    contract_path = _contract_path(project_root, path)
    if not contract_path.is_file():
        raise FeatureMartError(f"Phase 5 mart contract is missing: {contract_path}")
    try:
        payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        contract = MartContract.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise FeatureMartError(f"Invalid Phase 5 mart contract: {contract_path}") from exc
    return contract, mart_contract_checksum(contract_path)


def leakage_rule_matches(field: str, rule: str) -> bool:
    """Match exact or table-prefix leakage rules, including ``table.*``."""

    if rule.endswith(".*"):
        return field.startswith(rule[:-1])
    return field == rule


def _target_mapping(contract: MartContract) -> FieldMapping:
    payload = dict(contract.target)
    payload.setdefault("artifact", "claim_snapshot")
    payload.setdefault("is_model_feature", False)
    payload.setdefault("is_target", True)
    payload.setdefault("is_lineage", False)
    payload.setdefault("is_control", False)
    payload.setdefault("transform_type", "rename")
    payload.setdefault("join_path", [])
    payload.setdefault("as_of_rule", "stored target is copied from the eligible claim")
    return FieldMapping.model_validate(payload)


def iter_contract_mappings(contract: MartContract) -> Iterable[FieldMapping]:
    """Yield every dataset-column mapping declared by the mart contract."""

    yield from contract.direct_feature_mappings
    yield from contract.lineage_mappings
    yield _target_mapping(contract)
    yield from contract.artifact_column_mappings
    for bridge in contract.historical_bridge_definitions:
        yield from bridge.field_mappings
        yield from bridge.control_mappings


def mappings_by_artifact(contract: MartContract) -> dict[str, list[FieldMapping]]:
    """Group declared mappings by artifact in stable declaration order."""

    grouped: dict[str, list[FieldMapping]] = {}
    for mapping in iter_contract_mappings(contract):
        grouped.setdefault(mapping.artifact, []).append(mapping)
    return grouped


def _source_name(mapping: FieldMapping) -> str:
    return f"{mapping.source_table}.{mapping.source_column}"


def _contains_any(value: Any, needles: set[str]) -> bool:
    rendered = json.dumps(value, sort_keys=True).casefold()
    return any(needle.casefold() in rendered for needle in needles)


def validate_mart_contract(
    schema_contract: SchemaContract,
    phase4_bundle: Phase4ContractBundle,
    *,
    schema_contract_checksum: str,
    contract: MartContract,
    contract_checksum: str,
) -> MartPlanValidationResult:
    """Validate Phase 5 against the authoritative schema and Phase 4 contracts."""

    errors: list[str] = []
    warnings: list[str] = []
    phase4_result = validate_phase4_contracts(
        schema_contract,
        phase4_bundle.target,
        phase4_bundle.feature_policy,
        phase4_bundle.leakage,
        checksums={
            "high_cost_target_v1.yaml": phase4_bundle.target_checksum,
            "claim_time_feature_policy_v1.yaml": phase4_bundle.feature_policy_checksum,
            "leakage_policy_v1.yaml": phase4_bundle.leakage_checksum,
        },
        schema_contract_checksum=schema_contract_checksum,
    )
    errors.extend(phase4_result.errors)
    warnings.extend(phase4_result.warnings)
    source_rules = validate_historical_source_rules(schema_contract, phase4_bundle.feature_policy)
    errors.extend(source_rules["errors"])

    if contract.version != contract.contract_version:
        errors.append("Phase 5 mart contract version and contract_version must match.")
    if contract.schema_contract_version != schema_contract.contract_version:
        errors.append("Phase 5 mart contract schema version does not match the schema contract.")
    if contract.schema_contract_checksum != schema_contract_checksum:
        errors.append("Phase 5 mart contract schema checksum does not match the schema contract.")
    if contract.target_contract_version != phase4_bundle.target.version:
        errors.append("Phase 5 target contract version does not match Phase 4.")
    if contract.target_contract_checksum != phase4_bundle.target_checksum:
        errors.append("Phase 5 target contract checksum does not match Phase 4.")
    if contract.feature_policy_version != phase4_bundle.feature_policy.version:
        errors.append("Phase 5 feature-policy version does not match Phase 4.")
    if contract.feature_policy_checksum != phase4_bundle.feature_policy_checksum:
        errors.append("Phase 5 feature-policy checksum does not match Phase 4.")
    if contract.leakage_policy_version != phase4_bundle.leakage.version:
        errors.append("Phase 5 leakage-policy version does not match Phase 4.")
    if contract.leakage_policy_checksum != phase4_bundle.leakage_checksum:
        errors.append("Phase 5 leakage-policy checksum does not match Phase 4.")
    if contract.mart_grain != "one row = one eligible warranty claim":
        errors.append("Phase 5 mart grain must be one row = one eligible warranty claim.")
    if contract.target.get("source_table") != phase4_bundle.target.source_table:
        errors.append("Phase 5 target source table does not match the Phase 4 target.")
    if contract.target.get("source_column") != phase4_bundle.target.source_column:
        errors.append("Phase 5 target source column does not match the Phase 4 target.")
    if contract.target.get("output_column") != "target__high_cost_claim_flag":
        errors.append("Phase 5 target output must be target__high_cost_claim_flag.")
    if contract.prediction_reference.get("source_column") != "claim_date":
        errors.append("Phase 5 prediction reference must be claim_date.")

    allowlists = build_future_allowlists(phase4_bundle.feature_policy)
    direct_expected_fields = set(allowlists["tier_a_direct_baseline"])
    historical_expected_fields = set(allowlists["tier_a_historical"])
    direct_fields = [_source_name(mapping) for mapping in contract.direct_feature_mappings]
    if len(direct_fields) != len(set(direct_fields)):
        errors.append("Duplicate direct source fields are not allowed.")
    if len({mapping.output_column for mapping in contract.direct_feature_mappings}) != len(
        contract.direct_feature_mappings
    ):
        errors.append("Duplicate direct output columns are not allowed.")
    missing_direct = sorted(direct_expected_fields - set(direct_fields))
    unknown_direct = sorted(set(direct_fields) - direct_expected_fields)
    if missing_direct:
        errors.append(f"Direct Tier A fields are unmapped: {', '.join(missing_direct)}")
    if unknown_direct:
        errors.append(f"Non-baseline fields entered direct mappings: {', '.join(unknown_direct)}")

    direct_deferred = sum(
        mapping.mapping_status == "DEFERRED_WITH_REASON"
        for mapping in contract.direct_feature_mappings
    )
    direct_materialized = len(contract.direct_feature_mappings) - direct_deferred
    for mapping in contract.direct_feature_mappings:
        source = _source_name(mapping)
        if mapping.policy != "ALLOW_BASELINE_POC" or not mapping.is_model_feature:
            errors.append(f"Direct mapping is not an ALLOW_BASELINE_POC model feature: {source}")
        if mapping.is_target:
            errors.append(f"Target cannot be present in direct feature mappings: {source}")
        if mapping.mapping_status == "DEFERRED_WITH_REASON" and not mapping.defer_reason:
            errors.append(f"Deferred direct field lacks a reason: {source}")

    historical_fields: list[str] = []
    bridge_names = {bridge.name for bridge in contract.historical_bridge_definitions}
    if len(bridge_names) != len(contract.historical_bridge_definitions):
        errors.append("Historical bridge names must be unique.")
    for bridge in contract.historical_bridge_definitions:
        if not bridge.as_of_rule or bridge.same_day_policy != "exclude":
            errors.append(f"Historical bridge lacks a strict same-day-safe rule: {bridge.name}")
        if bridge.source_table in schema_contract.excluded_tables:
            errors.append(f"Excluded ML table appears in historical bridge: {bridge.source_table}")
        for mapping in [*bridge.field_mappings, *bridge.control_mappings]:
            if mapping.artifact != bridge.artifact:
                errors.append(f"Bridge mapping artifact mismatch: {mapping.output_column}")
            if mapping.is_model_feature:
                historical_fields.append(_source_name(mapping))
                if mapping.policy != "ALLOW_HISTORICAL_POC":
                    errors.append(
                        f"Historical model field is not ALLOW_HISTORICAL_POC: {_source_name(mapping)}"
                    )
    historical_fields = list(dict.fromkeys(historical_fields))
    missing_historical = sorted(historical_expected_fields - set(historical_fields))
    unknown_historical = sorted(set(historical_fields) - historical_expected_fields)
    if missing_historical:
        errors.append(f"Historical Tier A fields are unmapped: {', '.join(missing_historical)}")
    if unknown_historical:
        errors.append(
            f"Non-historical fields entered historical model mappings: {', '.join(unknown_historical)}"
        )
    historical_deferred = sum(
        item.get("mapping_status") == "DEFERRED_WITH_REASON"
        for item in contract.deferred_fields
        if item.get("tier") == "historical"
    )

    entries = {entry.field_name: entry for entry in phase4_bundle.feature_policy.field_policies}
    identifiers = set(phase4_bundle.leakage.identifier_fields)
    hard_blacklist = [rule.field for rule in phase4_bundle.leakage.hard_blacklist]
    all_outputs: set[tuple[str, str]] = set()
    for mapping in iter_contract_mappings(contract):
        output_key = (mapping.artifact, mapping.output_column)
        if output_key in all_outputs:
            errors.append(f"Duplicate materialized output column: {mapping.output_column}")
        all_outputs.add(output_key)
        source = _source_name(mapping)
        if mapping.source_table in schema_contract.excluded_tables:
            errors.append(f"Excluded ML table appears in mapping: {source}")
        entry = entries.get(source)
        if mapping.is_model_feature:
            if entry is None:
                errors.append(f"Model mapping source is absent from Phase 4 policy: {source}")
            elif entry.policy not in {"ALLOW_BASELINE_POC", "ALLOW_HISTORICAL_POC"}:
                errors.append(f"Excluded policy is materialized as a model feature: {source}")
            if source in identifiers or any(
                leakage_rule_matches(source, rule) for rule in hard_blacklist
            ):
                errors.append(f"Identifier or leakage rule overlaps a model feature: {source}")
        if mapping.is_target and mapping.is_model_feature:
            errors.append(f"Target is marked as a model feature: {mapping.output_column}")
        if (
            mapping.policy
            in {
                "TARGET_ONLY",
                "CONTROL_ONLY",
                "RESTRICTED_EXPERIMENTAL",
                "REQUIRES_CONFIRMATION",
                "PROHIBITED",
            }
            and mapping.is_model_feature
        ):
            errors.append(f"Non-Tier-A policy is marked as a model feature: {source}")

    if _contains_any(
        contract.source_join_paths.get("service_location", {}), {"customer", "dim_customer"}
    ):
        errors.append("Customer-to-location path cannot be used for direct service location.")
    component_bridge = next(
        (
            bridge
            for bridge in contract.historical_bridge_definitions
            if bridge.name == "component_installation"
        ),
        None,
    )
    component_text = json.dumps(
        [
            {
                "source": _source_name(mapping),
                "join_path": mapping.join_path,
            }
            for mapping in (
                [*component_bridge.field_mappings, *component_bridge.control_mappings]
                if component_bridge is not None
                else []
            )
        ],
        sort_keys=True,
    ).casefold()
    if "causal_component_key" in component_text:
        errors.append("Component history must not depend on current causal_component_key.")
    if not any(
        rule.field == "dbo.fact_repair_line.*" for rule in phase4_bundle.leakage.hard_blacklist
    ):
        errors.append("The Phase 5 plan requires the Phase 4 repair-line wildcard blacklist.")
    serialized = json.dumps(contract.model_dump(mode="json"), sort_keys=True).casefold()
    for excluded in schema_contract.excluded_tables:
        if excluded.casefold() in serialized:
            errors.append(f"Excluded ML table appears in the Phase 5 contract: {excluded}")
    if "lookback" in serialized or "rolling" in serialized or "target_rate" in serialized:
        errors.append(
            "Phase 5 contract must not introduce lookback or target-derived feature rules."
        )
    if contract.safety_rules.get("no_imputation") is not True:
        errors.append("Phase 5 safety rules must explicitly forbid imputation.")
    if contract.safety_rules.get("no_target_derived_features") is not True:
        errors.append("Phase 5 safety rules must explicitly forbid target-derived features.")

    return MartPlanValidationResult(
        valid=not errors,
        errors=list(dict.fromkeys(errors)),
        warnings=list(dict.fromkeys(warnings)),
        mart_contract_checksum=contract_checksum,
        direct_expected=len(direct_expected_fields),
        direct_materialized=direct_materialized,
        direct_deferred=direct_deferred,
        direct_fields=sorted(direct_fields),
        historical_expected=len(historical_expected_fields),
        historical_mapped=len(set(historical_fields) & historical_expected_fields),
        historical_deferred=historical_deferred,
        historical_fields=sorted(historical_fields),
        excluded_tables=list(schema_contract.excluded_tables),
    )


def assert_mart_contract_valid(result: MartPlanValidationResult) -> None:
    """Raise an actionable exception when the offline Phase 5 plan is unsafe."""

    if not result.valid:
        raise FeatureMartError("; ".join(result.errors))
