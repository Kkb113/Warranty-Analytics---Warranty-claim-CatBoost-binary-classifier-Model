"""Database-independent Phase 4 contract and eligibility tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from warranty_analytics_model.cli import main
from warranty_analytics_model.database.schema_contract import load_schema_contract
from warranty_analytics_model.policy.coverage import (
    build_future_allowlists,
    validate_feature_policy_coverage,
)
from warranty_analytics_model.policy.loader import load_phase4_contracts, policy_checksum
from warranty_analytics_model.policy.models import FeaturePolicyEntry, Phase4ContractError
from warranty_analytics_model.policy.target_contract import validate_claim_eligibility
from warranty_analytics_model.policy.validator import validate_phase4_contracts

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def schema_contract():
    return load_schema_contract(REPOSITORY_ROOT)[0]


@pytest.fixture
def phase4_bundle():
    return load_phase4_contracts(REPOSITORY_ROOT)


def test_all_phase4_contracts_load_and_validate(schema_contract, phase4_bundle) -> None:
    """The checked-in target, feature, and leakage contracts pass together."""

    result = validate_phase4_contracts(
        schema_contract,
        phase4_bundle.target,
        phase4_bundle.feature_policy,
        phase4_bundle.leakage,
    )

    assert result.valid is True
    assert result.schema_columns == 209
    assert result.classified_columns == 209
    assert result.unclassified_columns == 0
    assert result.policy_counts["TARGET_ONLY"] == 1
    assert result.policy_counts["PROHIBITED"] == 16


def test_target_contract_is_stored_binary_target_only(phase4_bundle) -> None:
    """The contract points at the stored flag and records cost only as evidence."""

    target = phase4_bundle.target
    assert target.source_table == "dbo.fact_warranty_claim"
    assert target.source_column == "high_cost_claim_flag"
    assert (target.positive_value, target.negative_value) == (1, 0)
    assert "total_claim_cost" in target.prohibited_derivation_fields
    assert target.target_generation_evidence.candidate_separator == 9999.525
    assert target.target_generation_evidence.business_rule_approved is False


def test_target_contract_is_not_production_approved(phase4_bundle) -> None:
    """Synthetic evidence does not silently become business approval."""

    assert phase4_bundle.target.development_status.production_approved is False
    assert phase4_bundle.target.development_status.business_target_definition_confirmed is False
    assert phase4_bundle.target.prediction_reference == "claim_date"
    assert phase4_bundle.target.prediction_reference_status == "provisional_date_level"


def test_target_contract_records_source_schema_provenance(schema_contract, phase4_bundle) -> None:
    """The target contract carries both required source-schema provenance fields."""

    _, checksum = load_schema_contract(REPOSITORY_ROOT)
    target = phase4_bundle.target
    assert target.source_schema_contract_version == schema_contract.contract_version
    assert target.source_schema_contract_checksum == checksum


def test_source_schema_checksum_mismatch_blocks_validation(schema_contract, phase4_bundle) -> None:
    """A policy tied to a different schema revision cannot pass silently."""

    target = phase4_bundle.target.model_copy(update={"source_schema_contract_checksum": "0" * 64})
    result = validate_phase4_contracts(
        schema_contract,
        target,
        phase4_bundle.feature_policy,
        phase4_bundle.leakage,
        schema_contract_checksum=load_schema_contract(REPOSITORY_ROOT)[1],
    )

    assert result.valid is False
    assert any("schema checksum" in error.lower() for error in result.errors)


def test_feature_policy_covers_exactly_all_schema_columns(schema_contract, phase4_bundle) -> None:
    """Coverage is exact and does not rely on a permissive default."""

    result = validate_feature_policy_coverage(schema_contract, phase4_bundle.feature_policy)

    assert result.valid is True
    assert result.schema_columns == result.classified_columns == 209
    assert result.unclassified_columns == 0


def test_unknown_schema_column_fails_closed(schema_contract, phase4_bundle) -> None:
    """A future schema field without a policy entry blocks validation."""

    entry = phase4_bundle.feature_policy.field_policies[0]
    unknown = entry.model_copy(update={"table": "dbo.future_table", "column": "future_column"})
    policy = phase4_bundle.feature_policy.model_copy(
        update={"field_policies": [*phase4_bundle.feature_policy.field_policies, unknown]}
    )

    with pytest.raises(Phase4ContractError, match="unknown schema columns"):
        validate_feature_policy_coverage(schema_contract, policy)


def test_missing_policy_column_fails_closed(schema_contract, phase4_bundle) -> None:
    """Removing one policy entry is a blocking unclassified-field error."""

    policy = phase4_bundle.feature_policy.model_copy(
        update={"field_policies": phase4_bundle.feature_policy.field_policies[:-1]}
    )

    with pytest.raises(Phase4ContractError, match="Unclassified schema columns"):
        validate_feature_policy_coverage(schema_contract, policy)


def test_duplicate_policy_entry_fails(schema_contract, phase4_bundle) -> None:
    """Duplicate field policy records cannot be collapsed silently."""

    policy = phase4_bundle.feature_policy.model_copy(
        update={
            "field_policies": [
                *phase4_bundle.feature_policy.field_policies,
                phase4_bundle.feature_policy.field_policies[0],
            ]
        }
    )

    with pytest.raises(Phase4ContractError, match="Duplicate policy entries"):
        validate_feature_policy_coverage(schema_contract, policy)


def test_target_cannot_be_a_model_feature(schema_contract, phase4_bundle) -> None:
    """The stored label remains target-only."""

    entries = [
        entry.model_copy(update={"is_model_feature": True})
        if entry.field_name == "dbo.fact_warranty_claim.high_cost_claim_flag"
        else entry
        for entry in phase4_bundle.feature_policy.field_policies
    ]
    policy = phase4_bundle.feature_policy.model_copy(update={"field_policies": entries})

    result = validate_phase4_contracts(
        schema_contract, phase4_bundle.target, policy, phase4_bundle.leakage
    )
    assert result.valid is False
    assert any("stored target" in error.lower() for error in result.errors)


def test_total_claim_cost_cannot_be_allowed(schema_contract, phase4_bundle) -> None:
    """Outcome cost cannot enter either future feature tier."""

    entries = [
        entry.model_copy(update={"policy": "ALLOW_BASELINE_POC", "is_model_feature": True})
        if entry.field_name == "dbo.fact_warranty_claim.total_claim_cost"
        else entry
        for entry in phase4_bundle.feature_policy.field_policies
    ]
    policy = phase4_bundle.feature_policy.model_copy(update={"field_policies": entries})

    result = validate_phase4_contracts(
        schema_contract, phase4_bundle.target, policy, phase4_bundle.leakage
    )
    assert result.valid is False
    assert any(
        "post-outcome" in error.lower() or "blacklist" in error.lower() for error in result.errors
    )


def test_known_post_outcome_and_repair_fields_are_excluded(schema_contract, phase4_bundle) -> None:
    """Known Phase 3 leakage findings remain prohibited or current-claim excluded."""

    entries = {entry.field_name: entry for entry in phase4_bundle.feature_policy.field_policies}
    for field in (
        "dbo.fact_warranty_claim.total_claim_cost",
        "dbo.fact_warranty_claim.claim_status",
        "dbo.fact_warranty_claim.root_cause_category",
        "dbo.fact_warranty_claim.repair_end_date",
        "dbo.fact_warranty_claim.potential_recall_flag",
    ):
        assert entries[field].policy == "PROHIBITED"
    repair_entry = entries["dbo.fact_repair_line.labor_hours"]
    assert repair_entry.synthetic_poc_allowed is False
    assert repair_entry.current_claim_use == "PROHIBITED"

    result = validate_phase4_contracts(
        schema_contract,
        phase4_bundle.target,
        phase4_bundle.feature_policy,
        phase4_bundle.leakage,
    )
    assert result.valid is True


def test_identifiers_and_high_cardinality_fields_are_not_tier_a(phase4_bundle) -> None:
    """Raw IDs are control-only and group fields are isolated."""

    allowlists = build_future_allowlists(phase4_bundle.feature_policy)
    safe = set(allowlists["tier_a_safe_baseline"])
    restricted = set(allowlists["tier_b_restricted_experimental"])
    assert "dbo.dim_truck.vin" not in safe
    assert "dbo.dim_truck.engine_serial_no" not in safe
    assert "dbo.fact_warranty_claim.warranty_claim_key" not in safe
    assert "dbo.dim_truck.production_batch_id" in restricted
    assert "dbo.fact_component_installation.component_lot_no" in restricted
    assert "dbo.dim_service_center.service_center_key" in restricted


def test_unresolved_current_claim_fields_remain_confirmation_required(phase4_bundle) -> None:
    """Diagnostic success in Phase 3 does not prove submission-time availability."""

    entries = {entry.field_name: entry for entry in phase4_bundle.feature_policy.field_policies}
    assert entries["dbo.fact_warranty_claim.causal_component_key"].policy == "REQUIRES_CONFIRMATION"
    assert entries["dbo.fact_warranty_claim.failure_code_key"].policy == "REQUIRES_CONFIRMATION"
    assert entries["dbo.fact_warranty_claim.root_cause_category"].policy == "PROHIBITED"
    assert entries["dbo.fact_service_event.complaint_description"].policy == "REQUIRES_CONFIRMATION"
    assert entries["dbo.fact_service_event.diagnostic_summary"].policy == "REQUIRES_CONFIRMATION"


def test_historical_rules_are_strict_and_source_specific(phase4_bundle) -> None:
    """Every historical source carries an enforceable same-day-safe rule."""

    rules = phase4_bundle.feature_policy.historical_sources
    assert rules["telemetry"].qualification_rule == "end_of_month(month_start_date) < claim_date"
    assert rules["maintenance"].qualification_rule == "maintenance_date < claim_date"
    assert "service_date < claim_date" in rules["service"].qualification_rule
    assert (
        "service_event_key != current_claim.service_event_key"
        in rules["service"].current_record_exclusion
    )
    assert "repair_end_date < claim_date" in rules["repair"].qualification_rule
    assert "installed_date < claim_date" in rules["component_installation"].qualification_rule
    assert all(rule.same_day_policy == "exclude" for rule in rules.values())


def test_lineage_fields_are_not_automatically_model_features(phase4_bundle) -> None:
    """Lineage is explicit metadata and does not imply feature inclusion."""

    entries = {entry.field_name: entry for entry in phase4_bundle.feature_policy.field_policies}
    for field in (
        "dbo.fact_warranty_claim.warranty_claim_key",
        "dbo.dim_truck.production_batch_id",
        "dbo.fact_component_installation.component_lot_no",
    ):
        assert field in phase4_bundle.feature_policy.lineage_fields
        assert entries[field].is_model_feature is False


def test_excluded_ml_tables_remain_outside_policy(schema_contract, phase4_bundle) -> None:
    """The three excluded ML tables cannot become policy sources."""

    assert set(phase4_bundle.feature_policy.excluded_tables) == set(schema_contract.excluded_tables)
    assert set(phase4_bundle.leakage.excluded_tables) == set(schema_contract.excluded_tables)
    assert all(
        entry.table not in schema_contract.excluded_tables
        for entry in phase4_bundle.feature_policy.field_policies
    )


def test_policy_checksums_are_deterministic(phase4_bundle) -> None:
    """Checksums are stable for unchanged contract bytes."""

    target_path = REPOSITORY_ROOT / "contracts" / "high_cost_target_v1.yaml"
    feature_path = REPOSITORY_ROOT / "contracts" / "claim_time_feature_policy_v1.yaml"
    leakage_path = REPOSITORY_ROOT / "contracts" / "leakage_policy_v1.yaml"
    assert phase4_bundle.target_checksum == policy_checksum(target_path)
    assert phase4_bundle.feature_policy_checksum == policy_checksum(feature_path)
    assert phase4_bundle.leakage_checksum == policy_checksum(leakage_path)


def test_claim_eligibility_is_explicit_and_does_not_require_history() -> None:
    """Only claim identity/date/target/truck-link rules determine eligibility."""

    claims = pd.DataFrame(
        {
            "warranty_claim_key": [1, 1, None, 4, 5, 6, 7],
            "claim_date": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
                None,
                "2026-01-05",
                "2026-01-06",
                "2026-01-07",
            ],
            "high_cost_claim_flag": [0, 1, 1, 0, 1, 0, None],
            "truck_key": [10, 10, 10, 10, 10, 999, 10],
            "total_claim_cost": [1.0, 2.0, 3.0, 4.0, 10_001.0, 5.0, 6.0],
        }
    )
    trucks = pd.DataFrame({"truck_key": [10]})

    result = validate_claim_eligibility(claims, trucks)

    assert result.total_claims == 7
    assert result.eligible_claims == 1
    assert result.excluded_claims == 6
    assert result.category_counts["ELIGIBLE"] == 1
    assert result.duplicate_claim_key_claims == 2
    assert result.missing_claim_date_claims == 1
    assert result.unresolved_truck_link_claims == 1
    assert result.null_target_claims == 1
    assert result.positive_claims == 1
    assert result.negative_claims == 0
    assert result.target_valid is False


def test_invalid_target_values_are_reported() -> None:
    """Invalid target values are not silently coerced into eligibility."""

    claims = pd.DataFrame(
        {
            "warranty_claim_key": [1, 2],
            "claim_date": ["2026-01-01", "2026-01-02"],
            "high_cost_claim_flag": [0, 2],
            "truck_key": [10, 10],
        }
    )
    trucks = pd.DataFrame({"truck_key": [10]})

    result = validate_claim_eligibility(claims, trucks)

    assert result.target_valid is False
    assert result.invalid_target_claims == 1
    assert result.category_counts["INELIGIBLE_INVALID_TARGET"] == 1


def test_phase4_contract_cli_is_offline(capsys) -> None:
    """The required contract-check command succeeds without database settings."""

    assert main(["phase4-contract-check"]) == 0
    output = capsys.readouterr().out
    assert "209/209" in output
    assert "checksum" in output.lower()


def test_contract_serialization_is_deterministic(phase4_bundle) -> None:
    """Stable sorted JSON is suitable for future policy lineage checks."""

    first = json.dumps(phase4_bundle.feature_policy.model_dump(mode="json"), sort_keys=True)
    second = json.dumps(phase4_bundle.feature_policy.model_dump(mode="json"), sort_keys=True)
    assert first == second


def test_unknown_policy_enum_is_rejected(phase4_bundle) -> None:
    """The feature policy taxonomy has no unclassified or permissive state."""

    entry = phase4_bundle.feature_policy.field_policies[0]
    with pytest.raises(ValueError):
        FeaturePolicyEntry.model_validate(
            entry.model_dump(mode="json") | {"policy": "UNCLASSIFIED"}
        )
