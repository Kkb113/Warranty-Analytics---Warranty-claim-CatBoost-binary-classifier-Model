"""Contract loading and self-validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from warranty_analytics_model.database.exceptions import SchemaContractError
from warranty_analytics_model.database.models import ForeignKeySpec, SQLTypeSpec
from warranty_analytics_model.database.schema_contract import (
    EXPECTED_EXCLUDED_TABLES,
    EXPECTED_SUMMARY,
    contract_checksum,
    load_schema_contract,
    validate_contract,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _contract():
    return load_schema_contract(REPOSITORY_ROOT)[0]


def _replace_table(contract, table):
    tables = [table if item.name == table.name else item for item in contract.tables]
    return contract.model_copy(update={"tables": tables})


def test_authoritative_contract_reconciles_source_metadata() -> None:
    """The checked-in YAML matches the approved DOCX totals and exclusions."""

    contract, checksum = load_schema_contract(REPOSITORY_ROOT)

    assert contract.contract_version == "1.0.0"
    assert contract.database.name == "warranty_analytics"
    assert contract.source.document == "warranty_analytics_schema_document.docx"
    assert (
        contract.source.sha256 == "13b749388c5c1ab94b6507832b3d0e6b37c9a7b016aec8cb71c57186498b7628"
    )
    assert contract.summary.model_dump() == EXPECTED_SUMMARY
    assert set(contract.excluded_tables) == EXPECTED_EXCLUDED_TABLES
    assert contract.included_column_count == 209
    assert contract.foreign_key_count == 22
    assert checksum == contract_checksum(
        REPOSITORY_ROOT / "contracts" / "warranty_analytics_schema_v1.yaml"
    )
    assert contract.table_map["dbo.fact_warranty_claim"].column_count == 34


def test_contract_captures_ordered_keys_types_and_provenance() -> None:
    """Primary keys, foreign keys, decimal metadata, and source properties are typed."""

    contract = _contract()
    claim = contract.table_map["dbo.fact_warranty_claim"]
    total_cost = claim.column_map["total_claim_cost"]
    foreign_key = claim.foreign_keys[-1]

    assert claim.primary_key is not None
    assert claim.primary_key.columns == ["warranty_claim_key"]
    assert total_cost.sql_type.precision == 12
    assert total_cost.sql_type.scale == 2
    assert total_cost.nullable is False
    assert foreign_key.referenced_table == "dbo.fact_service_event"
    assert foreign_key.parent_columns == ["service_event_key"]
    assert foreign_key.trusted is True
    assert claim.properties.temporal is False
    assert claim.properties.storage_classification == "NON_TEMPORAL_TABLE"


@pytest.mark.parametrize(
    "mutator, message",
    [
        (
            lambda contract: contract.model_copy(update={"contract_version": "bad"}),
            "semantic version",
        ),
        (
            lambda contract: contract.model_copy(
                update={"summary": contract.summary.model_copy(update={"estimated_rows": 1})}
            ),
            "summary totals",
        ),
        (
            lambda contract: contract.model_copy(
                update={"excluded_tables": [*contract.excluded_tables, "dbo.dim_component"]}
            ),
            "excluded_tables",
        ),
    ],
)
def test_contract_rejects_invalid_top_level_metadata(mutator, message: str) -> None:
    """Top-level version, totals, and scope errors are blocking."""

    with pytest.raises(SchemaContractError, match=message):
        validate_contract(mutator(_contract()))


def test_contract_rejects_duplicate_tables_columns_and_ordinals() -> None:
    """Duplicate identity metadata cannot pass self-validation."""

    contract = _contract()
    duplicate_table = contract.model_copy(update={"tables": [*contract.tables, contract.tables[0]]})
    with pytest.raises(SchemaContractError, match="duplicate fully qualified"):
        validate_contract(duplicate_table)

    first = contract.tables[0]
    duplicate_column = first.columns[0]
    bad_columns = first.model_copy(update={"columns": [*first.columns, duplicate_column]})
    with pytest.raises(SchemaContractError, match="column count"):
        validate_contract(_replace_table(contract, bad_columns))

    bad_ordinals = first.columns[1].model_copy(update={"ordinal": 1})
    ordinal_columns = [
        bad_ordinals if column.name == bad_ordinals.name else column for column in first.columns
    ]
    with pytest.raises(SchemaContractError, match="ordinals"):
        validate_contract(
            _replace_table(contract, first.model_copy(update={"columns": ordinal_columns}))
        )


def test_contract_rejects_bad_primary_key_foreign_key_and_type_metadata() -> None:
    """Key references and SQL type metadata must refer to the contract inventory."""

    contract = _contract()
    first = contract.tables[0]
    assert first.primary_key is not None
    with pytest.raises(SchemaContractError, match="primary key column is unknown"):
        validate_contract(
            _replace_table(
                contract,
                first.model_copy(
                    update={
                        "primary_key": first.primary_key.model_copy(update={"columns": ["missing"]})
                    }
                ),
            )
        )

    bad_fk = ForeignKeySpec(
        name="FK_bad",
        referenced_table="dbo.not_in_contract",
        parent_columns=["component_key"],
        referenced_columns=["missing"],
        on_delete="no_action",
        on_update="no_action",
        trusted=True,
    )
    with pytest.raises(SchemaContractError, match="referenced table is unknown"):
        validate_contract(
            _replace_table(contract, first.model_copy(update={"foreign_keys": [bad_fk]}))
        )

    bad_type = SQLTypeSpec(base_type="decimal", precision=None, scale=None)
    bad_column = first.columns[0].model_copy(update={"sql_type": bad_type})
    bad_columns = [
        bad_column if column.name == bad_column.name else column for column in first.columns
    ]
    with pytest.raises(SchemaContractError, match="decimal type requires"):
        validate_contract(
            _replace_table(contract, first.model_copy(update={"columns": bad_columns}))
        )


def test_contract_load_errors_are_safe(tmp_path: Path) -> None:
    """Missing and malformed contract files fail without database access."""

    with pytest.raises(SchemaContractError, match="missing"):
        load_schema_contract(tmp_path / "missing.yaml")
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(SchemaContractError, match="top-level mapping"):
        load_schema_contract(malformed)
