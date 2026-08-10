"""Offline Phase 5 contract, plan, and safety tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from warranty_analytics_model.cli import main
from warranty_analytics_model.database.schema_contract import load_schema_contract
from warranty_analytics_model.feature_mart.extraction_plan import (
    build_extraction_plan,
    explicit_select_sql,
    plan_columns,
)
from warranty_analytics_model.feature_mart.mart_contract import (
    load_mart_contract,
    mart_contract_checksum,
    validate_mart_contract,
)
from warranty_analytics_model.feature_mart.models import FeatureMartError
from warranty_analytics_model.policy.loader import load_phase4_contracts

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def phase5_contracts():
    schema, schema_checksum = load_schema_contract(REPOSITORY_ROOT)
    phase4 = load_phase4_contracts(REPOSITORY_ROOT)
    mart, mart_checksum = load_mart_contract(REPOSITORY_ROOT)
    return schema, schema_checksum, phase4, mart, mart_checksum


def test_mart_contract_references_phase4_checksums(phase5_contracts) -> None:
    """The mart contract is pinned to the exact Phase 4 contract bytes."""

    schema, schema_checksum, phase4, mart, mart_checksum = phase5_contracts
    result = validate_mart_contract(
        schema,
        phase4,
        schema_contract_checksum=schema_checksum,
        contract=mart,
        contract_checksum=mart_checksum,
    )
    assert result.valid is True
    assert result.direct_expected == result.direct_materialized == 41
    assert result.historical_expected == result.historical_mapped == 43
    assert (
        mart_contract_checksum(REPOSITORY_ROOT / "contracts" / "claim_feature_mart_v1.yaml")
        == mart_checksum
    )


def test_phase5_plan_check_is_offline(capsys) -> None:
    """The plan gate succeeds without database settings."""

    assert main(["phase5-plan-check"]) == 0
    output = capsys.readouterr().out
    assert "direct=41/41" in output
    assert "historical=43/43" in output


def test_extraction_plan_is_explicit_and_excludes_ml_tables(phase5_contracts) -> None:
    """The planner reads only contract columns and never the excluded ML objects."""

    schema, _, phase4, mart, _ = phase5_contracts
    plan = build_extraction_plan(schema, phase4, mart)
    assert all(table not in plan.excluded_tables for table in plan.table_names)
    assert "SELECT *" not in explicit_select_sql(
        "dbo.fact_warranty_claim", plan_columns(plan, "dbo.fact_warranty_claim")[:1]
    )
    claim_columns = {item.column for item in plan.columns_by_table["dbo.fact_warranty_claim"]}
    assert "total_claim_cost" not in claim_columns
    assert "high_cost_claim_flag" in claim_columns
    assert "repair_end_date" in claim_columns


def test_unknown_direct_field_fails_closed(phase5_contracts) -> None:
    """A source field outside the Phase 4 direct allowlist blocks the plan."""

    schema, schema_checksum, phase4, mart, mart_checksum = phase5_contracts
    mapping = mart.direct_feature_mappings[0].model_copy(
        update={"source_column": "total_claim_cost"}
    )
    changed = mart.model_copy(
        update={"direct_feature_mappings": [mapping, *mart.direct_feature_mappings[1:]]}
    )
    result = validate_mart_contract(
        schema,
        phase4,
        schema_contract_checksum=schema_checksum,
        contract=changed,
        contract_checksum=mart_checksum,
    )
    assert result.valid is False
    assert any("baseline" in error.lower() or "direct" in error.lower() for error in result.errors)


@pytest.mark.parametrize(
    "policy", ["PROHIBITED", "REQUIRES_CONFIRMATION", "RESTRICTED_EXPERIMENTAL"]
)
def test_non_tier_a_direct_field_fails_closed(phase5_contracts, policy: str) -> None:
    """Prohibited, confirmation, and restricted fields cannot enter direct Tier A."""

    schema, schema_checksum, phase4, mart, mart_checksum = phase5_contracts
    mapping = mart.direct_feature_mappings[0].model_copy(update={"policy": policy})
    changed = mart.model_copy(
        update={"direct_feature_mappings": [mapping, *mart.direct_feature_mappings[1:]]}
    )
    result = validate_mart_contract(
        schema,
        phase4,
        schema_contract_checksum=schema_checksum,
        contract=changed,
        contract_checksum=mart_checksum,
    )
    assert result.valid is False


def test_target_cannot_be_model_feature(phase5_contracts) -> None:
    """The label remains separate from the direct feature allowlist."""

    schema, schema_checksum, phase4, mart, mart_checksum = phase5_contracts
    target = dict(mart.target) | {"is_model_feature": True}
    changed = mart.model_copy(update={"target": target})
    result = validate_mart_contract(
        schema,
        phase4,
        schema_contract_checksum=schema_checksum,
        contract=changed,
        contract_checksum=mart_checksum,
    )
    assert result.valid is False


def test_customer_location_path_is_rejected(phase5_contracts) -> None:
    """Direct service location cannot silently switch to customer context."""

    schema, schema_checksum, phase4, mart, mart_checksum = phase5_contracts
    changed_paths = dict(mart.source_join_paths)
    changed_paths["service_location"] = [
        "dbo.dim_customer.location_key",
        "dbo.dim_location.location_key",
    ]
    changed = mart.model_copy(update={"source_join_paths": changed_paths})
    result = validate_mart_contract(
        schema,
        phase4,
        schema_contract_checksum=schema_checksum,
        contract=changed,
        contract_checksum=mart_checksum,
    )
    assert result.valid is False
    assert any("customer" in error.lower() for error in result.errors)


def test_wildcard_repair_rule_matches_table_prefix() -> None:
    """Wildcard leakage policy rules are matched at table scope."""

    from warranty_analytics_model.feature_mart.mart_contract import leakage_rule_matches

    assert leakage_rule_matches("dbo.fact_repair_line.line_cost", "dbo.fact_repair_line.*")
    assert not leakage_rule_matches("dbo.fact_service_event.service_type", "dbo.fact_repair_line.*")


def test_unknown_extraction_column_fails(phase5_contracts) -> None:
    """The explicit source planner rejects a mapping that does not exist in the schema."""

    schema, _, phase4, mart, _ = phase5_contracts
    mapping = mart.lineage_mappings[0].model_copy(update={"source_column": "unknown_key"})
    changed = mart.model_copy(update={"lineage_mappings": [mapping, *mart.lineage_mappings[1:]]})
    with pytest.raises(FeatureMartError, match="unknown source column"):
        build_extraction_plan(schema, phase4, changed)
