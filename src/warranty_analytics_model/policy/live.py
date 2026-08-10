"""Read-only live Phase 4 target and policy audits."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

import pandas as pd
from sqlalchemy import text

from ..config import Settings
from ..database.connection import DatabaseConnection
from ..database.models import SchemaContract
from ..profiling.extractor import quote_identifier
from .models import (
    EligibilityValidationResult,
    Phase4ContractBundle,
    Phase4ValidationResult,
)
from .target_contract import validate_claim_eligibility
from .validator import validate_historical_source_rules, validate_phase4_contracts


def _select_sql(table_name: str, columns: list[str]) -> str:
    """Build a reviewed select over a table already present in the schema contract."""

    schema, table = table_name.split(".", 1)
    selected = ", ".join(quote_identifier(column) for column in columns)
    return f"SELECT {selected} FROM {quote_identifier(schema)}.{quote_identifier(table)}"


def _read_claim_inputs(
    settings: Settings,
    schema_contract: SchemaContract,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    claim_table = schema_contract.table_map["dbo.fact_warranty_claim"]
    truck_table = schema_contract.table_map["dbo.dim_truck"]
    claim_columns = [
        "warranty_claim_key",
        "claim_date",
        "high_cost_claim_flag",
        "truck_key",
        "total_claim_cost",
    ]
    claim_names = set(claim_table.column_map)
    selected_claim_columns = [column for column in claim_columns if column in claim_names]
    connection = DatabaseConnection(settings.database)
    try:
        with connection.connect() as db_connection:
            claim_rows = list(
                db_connection.execute(
                    text(_select_sql(claim_table.name, selected_claim_columns))
                ).mappings()
            )
            truck_rows = list(
                db_connection.execute(text(_select_sql(truck_table.name, ["truck_key"]))).mappings()
            )
            claims = pd.DataFrame(
                [dict(row) for row in claim_rows],
                columns=selected_claim_columns,
            )
            trucks = pd.DataFrame([dict(row) for row in truck_rows], columns=["truck_key"])
            return claims, trucks
    finally:
        connection.dispose()


def _leakage_summary(bundle: Phase4ContractBundle, safe_allowlist: list[str]) -> dict[str, Any]:
    blacklist = [rule.field for rule in bundle.leakage.hard_blacklist]
    overlap = sorted(set(blacklist) & set(safe_allowlist))
    identifier_overlap = sorted(set(bundle.leakage.identifier_fields) & set(safe_allowlist))
    return {
        "valid": not overlap and not identifier_overlap,
        "hard_blacklist_count": len(blacklist),
        "hard_blacklist": blacklist,
        "identifier_field_count": len(bundle.leakage.identifier_fields),
        "identifier_safe_baseline_overlap": identifier_overlap,
        "blacklist_safe_baseline_overlap": overlap,
        "current_claim_exclusions": bundle.leakage.current_claim_exclusions,
        "excluded_tables": bundle.leakage.excluded_tables,
    }


def run_live_phase4(
    settings: Settings,
    schema_contract: SchemaContract,
    bundle: Phase4ContractBundle,
    *,
    schema_validation: dict[str, Any] | None = None,
    schema_contract_checksum: str | None = None,
) -> Phase4ValidationResult:
    """Run the bounded live Phase 4 audit without writing to SQL Server."""

    contract_validation = validate_phase4_contracts(
        schema_contract,
        bundle.target,
        bundle.feature_policy,
        bundle.leakage,
        checksums={
            "high_cost_target_v1.yaml": bundle.target_checksum,
            "claim_time_feature_policy_v1.yaml": bundle.feature_policy_checksum,
            "leakage_policy_v1.yaml": bundle.leakage_checksum,
        },
        schema_contract_checksum=schema_contract_checksum,
    )
    source_validation = validate_historical_source_rules(
        schema_contract,
        bundle.feature_policy,
    )
    errors = list(contract_validation.errors) + list(source_validation["errors"])
    warnings = list(contract_validation.warnings)
    claims, trucks = _read_claim_inputs(settings, schema_contract)
    eligibility: EligibilityValidationResult = validate_claim_eligibility(
        claims,
        trucks,
        target_column=bundle.target.source_column,
        claim_key_column="warranty_claim_key",
        claim_date_column=bundle.target.prediction_reference,
        truck_key_column="truck_key",
    )
    target_validation = eligibility.model_dump(mode="json")
    target_validation["target_name"] = f"{bundle.target.source_table}.{bundle.target.source_column}"
    target_validation["positive_value"] = bundle.target.positive_value
    target_validation["negative_value"] = bundle.target.negative_value
    target_validation["business_definition_confirmed"] = (
        bundle.target.development_status.business_target_definition_confirmed
    )
    target_validation[
        "synthetic_development_only"
    ] = not bundle.target.development_status.production_approved
    target_validation["prediction_reference"] = bundle.target.prediction_reference
    if not eligibility.target_valid:
        errors.append("Live target contains null or values outside the approved binary set {0, 1}.")
    if eligibility.eligible_claims == 0:
        errors.append("No claim rows satisfy the Phase 4 eligibility policy.")
    if schema_validation is not None and schema_validation.get("status") != "passed":
        errors.append("Live schema validation did not pass.")
    if not bundle.target.development_status.business_target_definition_confirmed:
        warnings.append(
            "Business target definition remains unconfirmed; live validity is technical only."
        )
    if not bundle.target.development_status.real_data_reapproval_required:
        errors.append("Target contract must require real-data reapproval.")
    leakage_validation = _leakage_summary(
        bundle,
        contract_validation.safe_baseline_allowlist,
    )
    if not leakage_validation["valid"]:
        errors.append("Leakage or identifier fields overlap the safe baseline allowlist.")
    errors = list(dict.fromkeys(errors))
    warnings = list(dict.fromkeys(warnings))
    status: Literal["READY", "READY WITH WARNINGS", "BLOCKED"] = (
        "BLOCKED" if errors else ("READY WITH WARNINGS" if warnings else "READY")
    )
    return Phase4ValidationResult(
        status=status,
        errors=errors,
        warnings=warnings,
        contract_validation=contract_validation,
        target_validation=target_validation,
        schema_validation=schema_validation,
        source_policy_validation=source_validation,
        leakage_policy_validation=leakage_validation,
        checksums={
            "high_cost_target_v1.yaml": bundle.target_checksum,
            "claim_time_feature_policy_v1.yaml": bundle.feature_policy_checksum,
            "leakage_policy_v1.yaml": bundle.leakage_checksum,
        },
        execution_timestamp=datetime.now(UTC).isoformat(),
    )
