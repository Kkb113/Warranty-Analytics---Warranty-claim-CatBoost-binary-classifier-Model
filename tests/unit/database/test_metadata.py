"""Catalog metadata normalization and collection tests with a fully mocked connection."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from warranty_analytics_model.config import DatabaseSettings
from warranty_analytics_model.database import metadata
from warranty_analytics_model.database.models import SQLTypeSpec
from warranty_analytics_model.database.schema_contract import load_schema_contract

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_sql_server_type_normalization_preserves_unicode_and_decimal_metadata() -> None:
    """SQL Server byte lengths are normalized to logical character lengths."""

    nvarchar = metadata._normalize_type(
        {
            "sql_type": "nvarchar",
            "max_length": 200,
            "precision": 0,
            "scale": 0,
        }
    )
    decimal = metadata._normalize_type(
        {
            "sql_type": "decimal",
            "max_length": 0,
            "precision": 12,
            "scale": 2,
        }
    )
    nvarchar_max = metadata._normalize_type(
        {
            "sql_type": "nvarchar",
            "max_length": -1,
            "precision": 0,
            "scale": 0,
        }
    )

    assert nvarchar == SQLTypeSpec(
        base_type="nvarchar",
        max_length=100,
        unicode=True,
        length_unit="characters",
    )
    assert decimal.precision == 12
    assert decimal.scale == 2
    assert nvarchar_max.is_max is True
    assert nvarchar_max.max_length is None


class _Result:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def mappings(self) -> _Result:
        return self

    def __iter__(self):
        return iter(self.rows)


class _CatalogConnection:
    def __init__(self, contract) -> None:
        self.contract = contract
        self.queries: list[tuple[str, object | None]] = []

    def execute(self, statement: object, parameters: object | None = None) -> _Result:
        query = str(statement)
        self.queries.append((query, parameters))
        values = parameters if isinstance(parameters, dict) else {}
        table_name = str(values.get("table_name", ""))
        contract_table = self.contract.table_map.get(f"dbo.{table_name}")
        if "FROM sys.schemas AS s" in query and "TOP" not in query:
            return _Result([{"schema_name": "dbo"}, {"schema_name": "sys"}])
        if "FROM sys.tables AS t" in query and "t.name AS table_name" in query:
            names = [table.name for table in self.contract.tables]
            names.extend(self.contract.excluded_tables)
            names.append("dbo.extra_table")
            return _Result(
                [{"schema_name": name.split(".")[0], "table_name": name.split(".")[1]} for name in names]
            )
        if "FROM sys.views AS v" in query:
            return _Result([{"schema_name": "dbo", "view_name": "documented_view"}])
        if "FROM sys.sequences AS seq" in query:
            return _Result([{"schema_name": "dbo", "sequence_name": "documented_sequence"}])
        if "SERVERPROPERTY" in query:
            return _Result(
                [
                    {
                        "database_name": "warranty_analytics",
                        "server_name": "sql.example.test",
                        "product_version": "16.0",
                        "sql_version": "Microsoft SQL Server 2022",
                    }
                ]
            )
        if "TOP" in query and "sys.schemas" in query:
            return _Result([{"schema_name": "dbo"}])
        if "t.is_memory_optimized" in query:
            return _Result(
                [
                    {
                        "schema_name": "dbo",
                        "table_name": table_name,
                        "is_memory_optimized": False,
                        "temporal_type_desc": "NON_TEMPORAL_TABLE",
                        "is_filetable": False,
                    }
                ]
            )
        assert contract_table is not None
        if "FROM sys.columns AS c" in query:
            rows = []
            for column in contract_table.columns:
                base = column.sql_type.base_type
                if column.sql_type.is_max:
                    max_length = -1
                elif base.startswith("n") and column.sql_type.max_length is not None:
                    max_length = column.sql_type.max_length * 2
                else:
                    max_length = column.sql_type.max_length or 0
                rows.append(
                    {
                        "ordinal": column.ordinal,
                        "column_name": column.name,
                        "sql_type": base,
                        "max_length": max_length,
                        "precision": column.sql_type.precision,
                        "scale": column.sql_type.scale,
                        "is_nullable": column.nullable,
                        "is_identity": column.identity,
                        "is_computed": column.computed,
                        "default_definition": column.default,
                        "computed_definition": None,
                        "collation_name": column.collation,
                    }
                )
            return _Result(rows)
        if "FROM sys.key_constraints AS kc" in query:
            assert contract_table.primary_key is not None
            return _Result(
                [
                    {
                        "constraint_name": contract_table.primary_key.name,
                        "is_system_named": False,
                        "column_name": column,
                        "key_ordinal": index + 1,
                    }
                    for index, column in enumerate(contract_table.primary_key.columns)
                ]
            )
        if "FROM sys.foreign_keys AS fk" in query:
            rows = []
            for foreign_key in contract_table.foreign_keys:
                for index, parent_column in enumerate(foreign_key.parent_columns):
                    referenced_table = self.contract.table_map[foreign_key.referenced_table]
                    rows.append(
                        {
                            "constraint_name": foreign_key.name,
                            "parent_schema_name": "dbo",
                            "parent_table_name": table_name,
                            "parent_column_name": parent_column,
                            "referenced_schema_name": referenced_table.schema,
                            "referenced_table_name": referenced_table.table,
                            "referenced_column_name": foreign_key.referenced_columns[index],
                            "on_delete": "NO_ACTION",
                            "on_update": "NO_ACTION",
                            "trusted": True,
                            "constraint_column_id": index + 1,
                        }
                    )
            return _Result(rows)
        if "FROM sys.indexes AS i" in query:
            rows = []
            for index in contract_table.indexes:
                rows.extend(
                    {
                        "index_name": index.name,
                        "index_type": index.index_type.upper(),
                        "is_unique": index.unique,
                        "is_disabled": False,
                        "filter_definition": None,
                        "column_name": column,
                        "key_ordinal": ordinal + 1,
                        "is_included_column": False,
                    }
                    for ordinal, column in enumerate(index.key_columns)
                )
            return _Result(rows)
        if "FROM sys.partitions AS p" in query:
            return _Result([{"estimated_rows": contract_table.estimated_rows}])
        raise AssertionError(f"Unhandled catalog query: {query}")


class _CatalogDatabaseConnection:
    def __init__(self, settings: DatabaseSettings, fake: _CatalogConnection) -> None:
        self.fake = fake

    @contextmanager
    def connect(self):
        yield self.fake

    def dispose(self) -> None:
        return None


def test_collect_schema_metadata_reads_only_catalog_and_excluded_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The collector reconstructs the contract shape without reading business rows."""

    contract = load_schema_contract(REPOSITORY_ROOT)[0]
    fake = _CatalogConnection(contract)
    monkeypatch.setattr(metadata, "validate_driver", lambda settings: None)
    monkeypatch.setattr(
        metadata,
        "DatabaseConnection",
        lambda settings: _CatalogDatabaseConnection(settings, fake),
    )

    live = metadata.collect_schema_metadata(DatabaseSettings(server="sql.example.test"), contract)

    assert len(live.tables) == 16
    assert live.included_column_count == 209
    assert live.foreign_key_count == 22
    assert "dbo.ml_truck_failure_risk_dataset" in live.excluded_objects
    assert "dbo.extra_table" in live.all_table_names
    assert live.views == ["dbo.documented_view"]
    assert live.sequences == ["dbo.documented_sequence"]
    assert all("SELECT *" not in query.upper() for query, _ in fake.queries)
    assert not any(
        any(excluded in query for excluded in contract.excluded_tables)
        for query, _ in fake.queries
    )
