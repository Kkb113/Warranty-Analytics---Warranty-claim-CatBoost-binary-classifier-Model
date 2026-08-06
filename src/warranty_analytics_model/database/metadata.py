"""Read-only SQL Server catalog inspection for the approved table scope."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from sqlalchemy import Connection, text

from ..config import DatabaseSettings
from .connection import DatabaseConnection, load_sql_resource, validate_driver
from .models import (
    ColumnSpec,
    ForeignKeySpec,
    IndexSpec,
    LiveSchemaMetadata,
    PrimaryKeySpec,
    SchemaContract,
    SQLTypeSpec,
    TableProperties,
    TableSpec,
)


def _rows(
    connection: Connection, query_name: str, parameters: Mapping[str, object] | None = None
) -> list[Mapping[str, Any]]:
    result = connection.execute(text(load_sql_resource(query_name)), parameters or {})
    return [dict(row) for row in result.mappings()]


def _qualified(schema: object, table: object) -> str:
    return f"{schema}.{table}"


def _normalize_type(row: Mapping[str, Any]) -> SQLTypeSpec:
    base_type = str(row["sql_type"]).casefold()
    raw_length = int(row["max_length"] or 0)
    string_types = {"char", "varchar", "nchar", "nvarchar"}
    is_string = base_type in string_types
    is_max = is_string and raw_length == -1
    if is_string and not is_max:
        logical_length = raw_length // 2 if base_type.startswith("n") else raw_length
    else:
        logical_length = None
    return SQLTypeSpec(
        base_type=base_type,
        max_length=logical_length,
        precision=int(row["precision"]) if row["precision"] is not None else None,
        scale=int(row["scale"]) if row["scale"] is not None else None,
        unicode=base_type.startswith("n"),
        length_unit=("characters" if base_type not in {"binary", "varbinary"} else "bytes")
        if is_string
        else None,
        is_max=is_max,
    )


def _collect_columns(connection: Connection, schema: str, table: str) -> list[ColumnSpec]:
    columns: list[ColumnSpec] = []
    for row in _rows(
        connection,
        "column_metadata.sql",
        {"schema_name": schema, "table_name": table},
    ):
        computed = bool(row["is_computed"])
        columns.append(
            ColumnSpec(
                ordinal=int(row["ordinal"]),
                name=str(row["column_name"]),
                sql_type=_normalize_type(row),
                nullable=bool(row["is_nullable"]),
                identity=bool(row["is_identity"]),
                computed=computed,
                default=None if computed else _optional_text(row["default_definition"]),
                collation=_optional_text(row["collation_name"]),
            )
        )
    return columns


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _collect_primary_key(
    connection: Connection,
    schema: str,
    table: str,
) -> PrimaryKeySpec | None:
    rows = _rows(
        connection,
        "primary_key_metadata.sql",
        {"schema_name": schema, "table_name": table},
    )
    if not rows:
        return None
    first = rows[0]
    return PrimaryKeySpec(
        name=str(first["constraint_name"]),
        columns=[str(row["column_name"]) for row in rows],
        system_named=bool(first["is_system_named"]),
    )


def _collect_foreign_keys(
    connection: Connection,
    schema: str,
    table: str,
) -> list[ForeignKeySpec]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in _rows(
        connection,
        "foreign_key_metadata.sql",
        {"schema_name": schema, "table_name": table},
    ):
        grouped[str(row["constraint_name"])].append(row)
    foreign_keys: list[ForeignKeySpec] = []
    for name, rows in grouped.items():
        first = rows[0]
        foreign_keys.append(
            ForeignKeySpec(
                name=name,
                referenced_table=_qualified(
                    first["referenced_schema_name"], first["referenced_table_name"]
                ),
                parent_columns=[str(row["parent_column_name"]) for row in rows],
                referenced_columns=[str(row["referenced_column_name"]) for row in rows],
                on_delete=str(first["on_delete"]).casefold(),
                on_update=str(first["on_update"]).casefold(),
                trusted=bool(first["trusted"]),
            )
        )
    return foreign_keys


def _collect_indexes(connection: Connection, schema: str, table: str) -> list[IndexSpec]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in _rows(
        connection,
        "index_metadata.sql",
        {"schema_name": schema, "table_name": table},
    ):
        grouped[str(row["index_name"])].append(row)
    indexes: list[IndexSpec] = []
    for name, rows in grouped.items():
        first = rows[0]
        key_rows = [row for row in rows if not bool(row["is_included_column"])]
        included_rows = [row for row in rows if bool(row["is_included_column"])]
        indexes.append(
            IndexSpec(
                name=name,
                index_type=str(first["index_type"]).casefold(),
                unique=bool(first["is_unique"]),
                key_columns=[str(row["column_name"]) for row in key_rows],
                included_columns=[str(row["column_name"]) for row in included_rows],
                disabled=bool(first["is_disabled"]),
                filter_definition=_optional_text(first["filter_definition"]),
            )
        )
    return indexes


def _collect_table(
    connection: Connection,
    contract_table: TableSpec,
) -> TableSpec | None:
    schema = contract_table.schema
    table = contract_table.table
    metadata_rows = _rows(
        connection,
        "table_metadata.sql",
        {"schema_name": schema, "table_name": table},
    )
    if not metadata_rows:
        return None
    metadata = metadata_rows[0]
    row_count = _rows(
        connection,
        "row_count_estimate.sql",
        {"schema_name": schema, "table_name": table},
    )[0]
    temporal_description = str(metadata["temporal_type_desc"]).casefold()
    properties = TableProperties(
        storage_classification=str(metadata["temporal_type_desc"]),
        temporal=temporal_description != "non_temporal_table",
        memory_optimized=bool(metadata["is_memory_optimized"]),
        file_table=bool(metadata["is_filetable"]),
    )
    columns = _collect_columns(connection, schema, table)
    return TableSpec(
        name=contract_table.name,
        schema=schema,
        table=table,
        estimated_rows=int(row_count["estimated_rows"]),
        column_count=len(columns),
        primary_key=_collect_primary_key(connection, schema, table),
        foreign_keys=_collect_foreign_keys(connection, schema, table),
        indexes=_collect_indexes(connection, schema, table),
        properties=properties,
        columns=columns,
    )


def collect_schema_metadata(
    settings: DatabaseSettings,
    contract: SchemaContract,
) -> LiveSchemaMetadata:
    """Collect catalog metadata for included tables and names-only extra objects."""

    settings.validate_for_connection()
    validate_driver(settings)
    connection = DatabaseConnection(settings)
    try:
        with connection.connect() as db_connection:
            schemas = [str(row["schema_name"]) for row in _rows(db_connection, "schema_names.sql")]
            table_name_rows = _rows(
                db_connection,
                "table_names.sql",
                {"schema_name": "dbo"},
            )
            all_table_names = {
                _qualified(row["schema_name"], row["table_name"]) for row in table_name_rows
            }
            excluded_objects = [
                name for name in contract.excluded_tables if name in all_table_names
            ]
            views = [
                _qualified(row["schema_name"], row["view_name"])
                for row in _rows(db_connection, "view_names.sql", {"schema_name": "dbo"})
            ]
            sequences = [
                _qualified(row["schema_name"], row["sequence_name"])
                for row in _rows(db_connection, "sequence_names.sql", {"schema_name": "dbo"})
            ]
            tables = [
                live_table
                for contract_table in contract.tables
                if (live_table := _collect_table(db_connection, contract_table)) is not None
            ]
            info = _rows(db_connection, "connection_info.sql")[0]
            return LiveSchemaMetadata(
                database_name=str(info["database_name"]),
                server_name=_optional_text(info["server_name"]),
                sql_version=_optional_text(info["sql_version"]),
                tables=tables,
                all_table_names=sorted(all_table_names),
                schemas=schemas,
                views=views,
                sequences=sequences,
                excluded_objects=excluded_objects,
                catalog_readable=True,
            )
    finally:
        connection.dispose()
