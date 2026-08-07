"""Contract-versus-live metadata diff tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from warranty_analytics_model.database.models import (
    ColumnSpec,
    LiveSchemaMetadata,
    SQLTypeSpec,
    TableProperties,
)
from warranty_analytics_model.database.schema_contract import load_schema_contract
from warranty_analytics_model.database.schema_diff import diff_schema

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _contract():
    return load_schema_contract(REPOSITORY_ROOT)[0]


def _live(contract, tables=None, **updates):
    payload = {
        "database_name": "warranty_analytics",
        "server_name": "sql.example.test",
        "sql_version": "Microsoft SQL Server 2022",
        "tables": tables
        if tables is not None
        else [table.model_copy(deep=True) for table in contract.tables],
        "all_table_names": [table.name for table in contract.tables],
        "schemas": ["dbo"],
        "views": [],
        "sequences": [],
        "excluded_objects": [],
        "catalog_readable": True,
    }
    payload.update(updates)
    return LiveSchemaMetadata(**payload)


def _codes(result) -> set[str]:
    return {issue.code for issue in result.issues}


def _replace_table(live, table):
    return live.model_copy(
        update={"tables": [table if item.name == table.name else item for item in live.tables]}
    )


def test_matching_schema_is_non_blocking_and_reports_excluded_status() -> None:
    """An exact catalog match has no errors or warnings."""

    contract = _contract()
    result = diff_schema(contract, _live(contract))

    assert result.row_count_differences == 0
    assert not [issue for issue in result.issues if issue.severity in {"ERROR", "WARNING"}]
    assert "EXCLUDED_OBJECT_ABSENT" in _codes(result)


@pytest.mark.parametrize(
    "mutate, expected_code",
    [
        (lambda live: live.model_copy(update={"tables": live.tables[1:]}), "MISSING_TABLE"),
        (
            lambda live: _replace_table(
                live,
                live.tables[0].model_copy(
                    update={"columns": live.tables[0].columns[1:], "column_count": 9}
                ),
            ),
            "MISSING_COLUMN",
        ),
        (
            lambda live: live.model_copy(
                update={"all_table_names": [*live.all_table_names, "dbo.extra"]}
            ),
            "EXTRA_TABLE",
        ),
        (
            lambda live: _replace_table(
                live,
                live.tables[0].model_copy(
                    update={
                        "columns": [
                            ColumnSpec(
                                ordinal=11,
                                name="added_column",
                                sql_type=SQLTypeSpec(base_type="int"),
                                nullable=True,
                            ),
                            *live.tables[0].columns,
                        ],
                        "column_count": 11,
                    }
                ),
            ),
            "EXTRA_COLUMN",
        ),
    ],
)
def test_table_and_column_scope_drift_is_reported(mutate, expected_code: str) -> None:
    """Missing and additional scope elements use the documented severity policy."""

    contract = _contract()
    result = diff_schema(contract, mutate(_live(contract)))

    assert expected_code in _codes(result)


@pytest.mark.parametrize(
    "column_name, actual_type, expected_code",
    [
        (
            "component_id",
            SQLTypeSpec(
                base_type="varchar", max_length=50, unicode=False, length_unit="characters"
            ),
            "COLUMN_TYPE_MISMATCH",
        ),
        (
            "component_id",
            SQLTypeSpec(
                base_type="nvarchar", max_length=40, unicode=True, length_unit="characters"
            ),
            "COLUMN_LENGTH_NARROWER",
        ),
        (
            "component_id",
            SQLTypeSpec(
                base_type="nvarchar", max_length=100, unicode=True, length_unit="characters"
            ),
            "COLUMN_LENGTH_WIDENED",
        ),
        (
            "unit_cost",
            SQLTypeSpec(base_type="decimal", precision=10, scale=2),
            "COLUMN_PRECISION_SCALE_MISMATCH",
        ),
    ],
)
def test_sql_type_normalization_and_compatibility(
    column_name: str,
    actual_type: SQLTypeSpec,
    expected_code: str,
) -> None:
    """Unicode, logical lengths, and decimal precision/scale are not ignored."""

    contract = _contract()
    table = contract.tables[0]
    column = table.column_map[column_name]
    actual_column = column.model_copy(update={"sql_type": actual_type})
    actual_table = table.model_copy(
        update={
            "columns": [
                actual_column if item.name == column_name else item for item in table.columns
            ]
        }
    )
    result = diff_schema(contract, _replace_table(_live(contract), actual_table))

    assert expected_code in _codes(result)


def test_nullability_identity_and_computed_drift_are_blocking() -> None:
    """Column structural flags remain part of the compatibility contract."""

    contract = _contract()
    table = contract.tables[0]
    column = table.columns[0]
    actual_column = column.model_copy(update={"nullable": True, "identity": True, "computed": True})
    actual_table = table.model_copy(
        update={
            "columns": [
                actual_column if item.name == column.name else item for item in table.columns
            ]
        }
    )
    codes = _codes(diff_schema(contract, _replace_table(_live(contract), actual_table)))

    assert {
        "COLUMN_NULLABILITY_MISMATCH",
        "COLUMN_IDENTITY_MISMATCH",
        "COLUMN_COMPUTED_MISMATCH",
    } <= codes


def test_primary_key_foreign_key_and_index_drift_are_classified() -> None:
    """Keys and indexes compare ordered mappings and use blocking/non-blocking severities."""

    contract = _contract()
    table = contract.tables[0]
    assert table.primary_key is not None
    assert table.foreign_keys
    assert table.indexes
    bad_table = table.model_copy(
        update={
            "primary_key": None,
            "foreign_keys": [],
            "indexes": [],
        }
    )
    result = diff_schema(contract, _replace_table(_live(contract), bad_table))
    codes = _codes(result)

    assert "MISSING_PRIMARY_KEY" in codes
    assert "MISSING_FOREIGN_KEY" in codes
    assert "MISSING_INDEX" in codes

    bad_fk = table.foreign_keys[0].model_copy(update={"referenced_table": "dbo.dim_customer"})
    mismatch = table.model_copy(update={"foreign_keys": [bad_fk]})
    assert "FOREIGN_KEY_MAPPING_MISMATCH" in _codes(
        diff_schema(contract, _replace_table(_live(contract), mismatch))
    )


def test_warnings_properties_defaults_and_row_estimates_support_strict_mode() -> None:
    """Non-blocking metadata drift passes by default and becomes blocking in strict mode."""

    contract = _contract()
    table = contract.tables[0]
    column = table.columns[1]
    widened_column = column.model_copy(
        update={"default": "(0)", "collation": "Latin1_General_100_BIN2"}
    )
    widened_table = table.model_copy(
        update={
            "estimated_rows": table.estimated_rows + 1,
            "properties": TableProperties(
                storage_classification="SYSTEM_VERSIONED_TEMPORAL_TABLE",
                temporal=True,
                memory_optimized=True,
                file_table=False,
            ),
            "columns": [
                widened_column if item.name == column.name else item for item in table.columns
            ],
        }
    )
    normal = diff_schema(contract, _replace_table(_live(contract), widened_table))
    strict = diff_schema(contract, _replace_table(_live(contract), widened_table), strict=True)

    assert normal.row_count_differences == 1
    assert "ROW_COUNT_ESTIMATE_DIFF" in _codes(normal)
    assert "COLUMN_DEFAULT_MISMATCH" in _codes(normal)
    assert "TABLE_PROPERTY_MISMATCH" in _codes(normal)
    assert any(issue.severity == "WARNING" for issue in normal.issues)
    assert all(issue.severity != "WARNING" for issue in strict.issues)


def test_wrong_database_unreadable_catalog_and_excluded_presence_are_explicit() -> None:
    """Database identity, catalog readability, and excluded names are reported safely."""

    contract = _contract()
    live = _live(
        contract,
        database_name="other_database",
        catalog_readable=False,
        all_table_names=[*live_names(contract), contract.excluded_tables[0]],
    )
    codes = _codes(diff_schema(contract, live))

    assert "DATABASE_NAME_MISMATCH" in codes
    assert "CATALOG_METADATA_UNREADABLE" in codes
    assert "EXCLUDED_OBJECT_PRESENT" in codes


def live_names(contract):
    """Return included names for a keyword-safe test fixture expression."""

    return [table.name for table in contract.tables]
