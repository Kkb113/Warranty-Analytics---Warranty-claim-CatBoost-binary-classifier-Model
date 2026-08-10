"""Cross-contract Phase 4 enforcement and fail-closed validation."""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..database.models import SchemaContract
from .coverage import schema_field_names, validate_feature_policy_coverage
from .models import (
    FEATURE_POLICY_NAMES,
    FeaturePolicyContract,
    LeakagePolicyContract,
    Phase4ContractBundle,
    Phase4ContractError,
    PolicyValidationResult,
    TargetContract,
)
from .target_contract import validate_target_contract

KNOWN_POST_OUTCOME_FIELDS = frozenset(
    {
        "dbo.fact_warranty_claim.total_claim_cost",
        "dbo.fact_warranty_claim.labor_cost",
        "dbo.fact_warranty_claim.parts_cost",
        "dbo.fact_warranty_claim.diagnostic_cost",
        "dbo.fact_warranty_claim.towing_cost",
        "dbo.fact_warranty_claim.other_cost",
        "dbo.fact_warranty_claim.approved_amount",
        "dbo.fact_warranty_claim.rejected_amount",
        "dbo.fact_warranty_claim.customer_paid_amount",
        "dbo.fact_warranty_claim.repair_end_date",
        "dbo.fact_warranty_claim.days_to_repair",
        "dbo.fact_warranty_claim.claim_status",
        "dbo.fact_warranty_claim.root_cause_category",
        "dbo.fact_warranty_claim.repeat_claim_flag",
        "dbo.fact_warranty_claim.potential_recall_flag",
    }
)

KNOWN_IDENTIFIER_COLUMNS = frozenset(
    {
        "warranty_claim_key",
        "claim_id",
        "truck_key",
        "vin",
        "service_event_key",
        "service_event_id",
        "component_key",
        "component_id",
        "component_serial_no",
        "engine_serial_no",
        "transmission_serial_no",
        "technician_id",
        "inspector_id",
        "customer_key",
        "customer_id",
        "supplier_key",
        "supplier_id",
        "service_center_key",
        "service_center_id",
        "location_key",
        "warranty_policy_key",
        "failure_code_key",
        "installation_key",
        "maintenance_event_key",
        "repair_line_key",
        "telemetry_month_key",
        "date_key",
        "truck_model_key",
    }
)

REQUIRED_HISTORICAL_SOURCES = {
    "telemetry": (
        "dbo.fact_telemetry_monthly",
        "end_of_month(month_start_date) < claim_date",
    ),
    "maintenance": (
        "dbo.fact_maintenance_event",
        "maintenance_date < claim_date",
    ),
    "service": (
        "dbo.fact_service_event",
        "service_date < claim_date",
    ),
    "repair": (
        "dbo.fact_repair_line",
        "prior_claim.repair_end_date < claim_date",
    ),
    "prior_claim": (
        "dbo.fact_warranty_claim",
        "prior_claim.claim_date < claim_date",
    ),
    "component_installation": (
        "dbo.fact_component_installation",
        "installed_date < claim_date",
    ),
}


def _base_result(
    schema_contract: SchemaContract,
    feature_policy: FeaturePolicyContract,
) -> PolicyValidationResult:
    try:
        return validate_feature_policy_coverage(schema_contract, feature_policy)
    except Phase4ContractError as exc:
        fields = [entry.field_name for entry in feature_policy.field_policies]
        counts = {
            str(policy): int(count)
            for policy, count in sorted(
                Counter(entry.policy for entry in feature_policy.field_policies).items()
            )
        }
        return PolicyValidationResult(
            valid=False,
            errors=[str(exc)],
            warnings=[],
            schema_columns=len(schema_field_names(schema_contract)),
            classified_columns=len(set(fields) & schema_field_names(schema_contract)),
            unclassified_columns=max(0, len(schema_field_names(schema_contract) - set(fields))),
            policy_counts=counts,
            safe_baseline_allowlist=[],
            historical_allowlist=[],
            restricted_experimental_list=[],
            requires_confirmation_list=[],
            lineage_fields=list(feature_policy.lineage_fields),
        )


def _validate_feature_invariants(
    schema_contract: SchemaContract,
    feature_policy: FeaturePolicyContract,
    leakage_policy: LeakagePolicyContract,
    result: PolicyValidationResult,
) -> tuple[list[str], list[str]]:
    errors = list(result.errors)
    warnings: list[str] = []
    entries = {entry.field_name: entry for entry in feature_policy.field_policies}
    schema_fields = schema_field_names(schema_contract)

    if feature_policy.schema_contract_version != schema_contract.contract_version:
        errors.append("Feature policy schema_contract_version does not match the schema contract.")
    if feature_policy.schema_contract_checksum != "":
        # The exact checksum is checked by the caller that loaded the contract.
        pass
    if set(feature_policy.feature_policy_enum) != FEATURE_POLICY_NAMES:
        errors.append("Feature policy enum is incomplete or contains an unsupported value.")
    if set(feature_policy.excluded_tables) != set(schema_contract.excluded_tables):
        errors.append("Feature policy excluded tables do not match the schema contract.")
    if set(leakage_policy.excluded_tables) != set(schema_contract.excluded_tables):
        errors.append("Leakage policy excluded tables do not match the schema contract.")

    target_field = "dbo.fact_warranty_claim.high_cost_claim_flag"
    target_entry = entries.get(target_field)
    if target_entry is None:
        errors.append("Stored target is missing from the field policy.")
    elif target_entry.policy != "TARGET_ONLY" or target_entry.is_model_feature:
        errors.append("The stored target must be TARGET_ONLY and never a model feature.")

    for field in KNOWN_POST_OUTCOME_FIELDS:
        entry = entries.get(field)
        if entry is None:
            continue
        if entry.policy != "PROHIBITED" or entry.is_model_feature:
            errors.append(f"Known post-outcome field is not prohibited: {field}")

    leakage_fields = {rule.field for rule in leakage_policy.hard_blacklist}
    for field in KNOWN_POST_OUTCOME_FIELDS:
        if field not in leakage_fields:
            errors.append(f"Known post-outcome field is missing from leakage blacklist: {field}")

    safe = set(result.safe_baseline_allowlist)
    for field in safe:
        entry = entries[field]
        if entry.policy not in {"ALLOW_BASELINE_POC", "ALLOW_HISTORICAL_POC"}:
            errors.append(f"Non-allowlisted policy entered Tier A: {field}")
        if field.rsplit(".", 1)[-1] in KNOWN_IDENTIFIER_COLUMNS:
            errors.append(f"Identifier entered the safe baseline: {field}")
        if field in leakage_fields:
            errors.append(f"Blacklisted field entered the safe baseline: {field}")
        if entry.policy == "ALLOW_HISTORICAL_POC" and not entry.as_of_rule:
            errors.append(f"Historical field lacks an as-of rule: {field}")

    for entry in feature_policy.field_policies:
        if entry.table not in {table.name for table in schema_contract.tables}:
            errors.append(f"Policy source table is not included: {entry.table}")
        if entry.policy not in FEATURE_POLICY_NAMES:
            errors.append(f"Unknown feature policy: {entry.policy}")
        if (
            entry.policy
            in {
                "TARGET_ONLY",
                "CONTROL_ONLY",
                "RESTRICTED_EXPERIMENTAL",
                "REQUIRES_CONFIRMATION",
                "PROHIBITED",
            }
            and entry.is_model_feature
        ):
            errors.append(f"Excluded policy is marked as a model feature: {entry.field_name}")
        if (
            entry.policy in {"CONTROL_ONLY", "REQUIRES_CONFIRMATION", "PROHIBITED"}
            and entry.synthetic_poc_allowed
        ):
            errors.append(f"Excluded policy is allowed in synthetic POC: {entry.field_name}")
        if entry.policy == "ALLOW_HISTORICAL_POC" and not entry.as_of_rule:
            errors.append(f"Historical field lacks an enforceable as-of rule: {entry.field_name}")
        if entry.policy == "RESTRICTED_EXPERIMENTAL" and entry.field_name in safe:
            errors.append(f"Restricted field entered Tier A: {entry.field_name}")

    unknown_lineage = [
        field
        for field in feature_policy.lineage_fields
        if field not in schema_fields
        and field
        not in {
            "duplicate_scenario_fingerprint",
            "feature_policy_version",
            "target_contract_version",
        }
    ]
    if unknown_lineage:
        errors.append(f"Unknown lineage fields: {', '.join(sorted(unknown_lineage))}")

    missing_sources = [
        f"{name}={expected[0]}"
        for name, expected in REQUIRED_HISTORICAL_SOURCES.items()
        if name not in feature_policy.historical_sources
        or feature_policy.historical_sources[name].source_table != expected[0]
    ]
    if missing_sources:
        errors.append(
            f"Missing or mismatched historical source policies: {', '.join(missing_sources)}"
        )
    for name, expected in REQUIRED_HISTORICAL_SOURCES.items():
        source = feature_policy.historical_sources.get(name)
        if source is not None and expected[1] not in source.qualification_rule:
            errors.append(
                f"Historical source rule is too permissive for {name}: {source.qualification_rule}"
            )
        if source is not None and source.same_day_policy != "exclude":
            errors.append(f"Same-day records are not excluded for {name}.")
    if feature_policy.prediction_time_policy.get("event_date_comparison") != "<":
        errors.append("Prediction-time event policy must use strict event_date < claim_date.")
    if feature_policy.prediction_time_policy.get("telemetry_completed_month_rule") != (
        "end_of_month(month_start_date) < claim_date"
    ):
        errors.append("Telemetry must use the completed-month as-of rule.")
    if feature_policy.same_day_policy.get("default") != "exclude":
        errors.append("Same-day historical events must be excluded by default.")

    for excluded in schema_contract.excluded_tables:
        if any(entry.table == excluded for entry in feature_policy.field_policies):
            errors.append(f"Excluded ML table appears in feature policy: {excluded}")

    if not feature_policy.development_status.production_approved:
        warnings.append(
            "Feature policy is synthetic-development-only pending real-data reapproval."
        )
    if not feature_policy.development_status.exact_submission_timestamp_available:
        warnings.append(
            "claim_date is date-level only; strict-before same-day policy remains mandatory."
        )
    return errors, warnings


def _validate_leakage_invariants(
    schema_contract: SchemaContract,
    feature_policy: FeaturePolicyContract,
    leakage_policy: LeakagePolicyContract,
) -> list[str]:
    errors: list[str] = []
    entries = {entry.field_name: entry for entry in feature_policy.field_policies}
    blacklist = {rule.field for rule in leakage_policy.hard_blacklist}
    schema_fields = schema_field_names(schema_contract)
    unknown_blacklist = sorted(
        field for field in blacklist if "*" not in field and field not in schema_fields
    )
    if unknown_blacklist:
        errors.append(f"Leakage policy contains unknown fields: {', '.join(unknown_blacklist)}")
    unknown_identifiers = sorted(
        field for field in leakage_policy.identifier_fields if field not in schema_fields
    )
    if unknown_identifiers:
        errors.append(
            f"Leakage identifier policy contains unknown fields: {', '.join(unknown_identifiers)}"
        )
    unknown_groups = sorted(
        field
        for field in leakage_policy.high_cardinality_group_fields
        if field not in schema_fields
    )
    if unknown_groups:
        errors.append(
            f"High-cardinality group policy contains unknown fields: {', '.join(unknown_groups)}"
        )
    if "dbo.fact_warranty_claim.high_cost_claim_flag" not in blacklist:
        errors.append("Stored target is missing from the hard leakage blacklist.")
    if "dbo.fact_repair_line.*" not in blacklist:
        errors.append(
            "Current-claim repair-line wildcard is missing from the hard leakage blacklist."
        )
    if set(leakage_policy.excluded_tables) != set(schema_contract.excluded_tables):
        errors.append("Leakage policy does not preserve the excluded ML-table scope.")
    safe = {
        entry.field_name
        for entry in feature_policy.field_policies
        if entry.policy in {"ALLOW_BASELINE_POC", "ALLOW_HISTORICAL_POC"} and entry.is_model_feature
    }
    overlaps = sorted(safe & blacklist)
    if overlaps:
        errors.append(f"Hard-blacklisted fields are allowed in Tier A: {', '.join(overlaps)}")
    for field in leakage_policy.identifier_fields:
        entry = entries.get(field)
        if entry is not None and entry.policy in {"ALLOW_BASELINE_POC", "ALLOW_HISTORICAL_POC"}:
            errors.append(f"Identifier is allowed as a model feature: {field}")
    if "synthetic-generator-leakage" not in leakage_policy.leakage_categories:
        errors.append("Leakage categories must include synthetic-generator-leakage.")
    return errors


def validate_phase4_contracts(
    schema_contract: SchemaContract,
    target_contract: TargetContract,
    feature_policy: FeaturePolicyContract,
    leakage_policy: LeakagePolicyContract,
    *,
    checksums: dict[str, str] | None = None,
    schema_contract_checksum: str | None = None,
) -> PolicyValidationResult:
    """Validate all offline Phase 4 contracts and return derived inventories."""

    result = _base_result(schema_contract, feature_policy)
    errors, warnings = _validate_feature_invariants(
        schema_contract, feature_policy, leakage_policy, result
    )
    errors.extend(validate_target_contract(schema_contract, target_contract))
    errors.extend(_validate_leakage_invariants(schema_contract, feature_policy, leakage_policy))
    if target_contract.schema_contract_version != schema_contract.contract_version:
        errors.append("Target contract schema_contract_version does not match the schema contract.")
    if target_contract.source_schema_contract_version != schema_contract.contract_version:
        errors.append("Target source_schema_contract_version does not match the schema contract.")
    if target_contract.version != target_contract.contract_version:
        errors.append("Target contract version and contract_version must match.")
    if feature_policy.version != feature_policy.contract_version:
        errors.append("Feature policy version and contract_version must match.")
    if leakage_policy.version != leakage_policy.contract_version:
        errors.append("Leakage policy version and contract_version must match.")
    if schema_contract_checksum is not None:
        for label, actual in (
            ("target", target_contract.schema_contract_checksum),
            ("target source", target_contract.source_schema_contract_checksum),
            ("feature", feature_policy.schema_contract_checksum),
            ("leakage", leakage_policy.schema_contract_checksum),
        ):
            if actual != schema_contract_checksum:
                errors.append(
                    f"{label.title()} policy schema checksum does not match the schema contract."
                )
    if checksums:
        result.target_checksum = checksums.get("high_cost_target_v1.yaml")
        result.feature_policy_checksum = checksums.get("claim_time_feature_policy_v1.yaml")
        result.leakage_policy_checksum = checksums.get("leakage_policy_v1.yaml")
    result.errors = list(dict.fromkeys(errors))
    result.warnings = list(dict.fromkeys(warnings))
    result.valid = not result.errors
    return result


def assert_phase4_contracts_valid(
    schema_contract: SchemaContract,
    bundle: Phase4ContractBundle,
) -> PolicyValidationResult:
    """Raise a safe, actionable exception when the offline policy is invalid."""

    result = validate_phase4_contracts(
        schema_contract,
        bundle.target,
        bundle.feature_policy,
        bundle.leakage,
        checksums={
            "high_cost_target_v1.yaml": bundle.target_checksum,
            "claim_time_feature_policy_v1.yaml": bundle.feature_policy_checksum,
            "leakage_policy_v1.yaml": bundle.leakage_checksum,
        },
    )
    if not result.valid:
        raise Phase4ContractError("; ".join(result.errors))
    return result


def validate_historical_source_rules(
    schema_contract: SchemaContract,
    feature_policy: FeaturePolicyContract,
) -> dict[str, Any]:
    """Check that every required historical source/date column exists."""

    table_map = schema_contract.table_map
    checks: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for name, rule in feature_policy.historical_sources.items():
        source = table_map.get(rule.source_table)
        columns = set(source.column_map) if source is not None else set()
        date_exists = rule.event_date_column is None or rule.event_date_column in columns
        checks[name] = {
            "source_table": rule.source_table,
            "event_date_column": rule.event_date_column,
            "source_table_exists": source is not None,
            "event_date_column_exists": date_exists,
            "qualification_rule": rule.qualification_rule,
            "same_day_policy": rule.same_day_policy,
        }
        if source is None:
            errors.append(f"Historical source table is not included: {rule.source_table}")
        if not date_exists:
            errors.append(
                f"Historical source date column is not included: "
                f"{rule.source_table}.{rule.event_date_column}"
            )
    return {"valid": not errors, "errors": errors, "sources": checks}
