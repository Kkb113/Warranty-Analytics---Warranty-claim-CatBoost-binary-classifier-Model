"""Load and self-validate the version-controlled SQL Server schema contract."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ..paths import discover_repository_root
from .exceptions import SchemaContractError
from .models import SchemaContract, TableSpec

EXPECTED_SUMMARY = {
    "included_tables": 16,
    "included_columns": 209,
    "documented_foreign_keys": 22,
    "estimated_rows": 392_352,
}
EXPECTED_EXCLUDED_TABLES = {
    "dbo.ml_region_terrain_warranty_risk_dataset",
    "dbo.ml_truck_failure_risk_dataset",
    "dbo.ml_truck_region_terrain_failure_risk_dataset",
}
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_STRING_TYPES = {"char", "varchar", "nchar", "nvarchar"}
_DECIMAL_TYPES = {"decimal", "numeric"}
_SUPPORTED_TYPES = {
    "bigint",
    "bit",
    "char",
    "date",
    "datetime",
    "datetime2",
    "decimal",
    "float",
    "int",
    "nchar",
    "nvarchar",
    "real",
    "smallint",
    "smalldatetime",
    "smallmoney",
    "time",
    "tinyint",
    "uniqueidentifier",
    "varbinary",
    "varchar",
    "money",
}


def default_contract_path(project_root: Path | None = None) -> Path:
    """Return the repository contract path without creating directories."""

    return (
        discover_repository_root(project_root) / "contracts" / "warranty_analytics_schema_v1.yaml"
    )


def contract_checksum(path: Path) -> str:
    """Return the SHA-256 checksum of the exact contract bytes."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise SchemaContractError(f"Could not read schema contract: {path}") from exc


def _format_validation_errors(error: ValidationError) -> str:
    details: list[str] = []
    for item in error.errors():
        location = ".".join(str(part) for part in item.get("loc", ())) or "contract"
        details.append(f"{location}: {item.get('msg', 'invalid value')}")
    return "; ".join(details)


def _check_type_metadata(table: TableSpec, errors: list[str]) -> None:
    for column in table.columns:
        type_spec = column.sql_type
        base = type_spec.base_type
        if base not in _SUPPORTED_TYPES:
            errors.append(f"{table.name}.{column.name}: unsupported SQL type {base}")
        elif base in _STRING_TYPES:
            if type_spec.is_max:
                if type_spec.max_length is not None:
                    errors.append(
                        f"{table.name}.{column.name}: max string type cannot have a length"
                    )
            elif type_spec.max_length is None or type_spec.max_length < 1:
                errors.append(f"{table.name}.{column.name}: string type requires a positive length")
            expected_unicode = base.startswith("n")
            if type_spec.unicode != expected_unicode:
                errors.append(f"{table.name}.{column.name}: unicode flag disagrees with SQL type")
            if type_spec.length_unit != "characters":
                errors.append(f"{table.name}.{column.name}: string length unit must be characters")
        elif base in _DECIMAL_TYPES:
            if type_spec.precision is None or type_spec.scale is None:
                errors.append(
                    f"{table.name}.{column.name}: decimal type requires precision and scale"
                )
            elif type_spec.scale > type_spec.precision:
                errors.append(f"{table.name}.{column.name}: decimal scale exceeds precision")
            if type_spec.max_length is not None or type_spec.length_unit is not None:
                errors.append(
                    f"{table.name}.{column.name}: decimal type has string length metadata"
                )
        else:
            if type_spec.max_length is not None or type_spec.precision is not None:
                errors.append(
                    f"{table.name}.{column.name}: scalar type has incompatible size metadata"
                )
            if type_spec.scale is not None or type_spec.length_unit is not None:
                errors.append(
                    f"{table.name}.{column.name}: scalar type has incompatible scale metadata"
                )
            if type_spec.unicode or type_spec.is_max:
                errors.append(
                    f"{table.name}.{column.name}: scalar type has incompatible Unicode metadata"
                )


def _check_table(table: TableSpec, tables: dict[str, TableSpec], errors: list[str]) -> None:
    if table.name != f"{table.schema}.{table.table}":
        errors.append(f"Table identity does not match fully qualified name: {table.name}")
    if table.column_count != len(table.columns):
        errors.append(f"{table.name}: column count does not match the column inventory")
    column_names = [column.name for column in table.columns]
    if len(column_names) != len(set(column_names)):
        errors.append(f"{table.name}: duplicate column names")
    ordinals = [column.ordinal for column in table.columns]
    if len(ordinals) != len(set(ordinals)) or ordinals != list(range(1, len(ordinals) + 1)):
        errors.append(f"{table.name}: duplicate or non-contiguous column ordinals")
    _check_type_metadata(table, errors)

    if table.primary_key is None:
        errors.append(f"{table.name}: primary key is missing")
    else:
        if len(table.primary_key.columns) != len(set(table.primary_key.columns)):
            errors.append(f"{table.name}: primary key has duplicate columns")
        for column in table.primary_key.columns:
            if column not in column_names:
                errors.append(f"{table.name}: primary key column is unknown: {column}")

    fk_names: set[str] = set()
    for foreign_key in table.foreign_keys:
        if foreign_key.name in fk_names:
            errors.append(f"{table.name}: duplicate foreign key name: {foreign_key.name}")
        fk_names.add(foreign_key.name)
        if len(foreign_key.parent_columns) != len(foreign_key.referenced_columns):
            errors.append(f"{table.name}.{foreign_key.name}: column mapping lengths differ")
        for column in foreign_key.parent_columns:
            if column not in column_names:
                errors.append(
                    f"{table.name}.{foreign_key.name}: parent column is unknown: {column}"
                )
        referenced = tables.get(foreign_key.referenced_table)
        if referenced is None:
            errors.append(
                f"{table.name}.{foreign_key.name}: referenced table is unknown: "
                f"{foreign_key.referenced_table}"
            )
        else:
            for column in foreign_key.referenced_columns:
                if column not in referenced.column_map:
                    errors.append(
                        f"{table.name}.{foreign_key.name}: referenced column is unknown: "
                        f"{foreign_key.referenced_table}.{column}"
                    )

    index_names: set[str] = set()
    for index in table.indexes:
        if index.name in index_names:
            errors.append(f"{table.name}: duplicate index name: {index.name}")
        index_names.add(index.name)
        for column in [*index.key_columns, *index.included_columns]:
            if column not in column_names:
                errors.append(f"{table.name}.{index.name}: index column is unknown: {column}")


def validate_contract(contract: SchemaContract) -> None:
    """Raise ``SchemaContractError`` when the parsed contract is not trustworthy."""

    errors: list[str] = []
    if not _VERSION_RE.fullmatch(contract.contract_version):
        errors.append("contract_version must use semantic version form MAJOR.MINOR.PATCH")
    if not contract.database.name.strip():
        errors.append("database.name must not be empty")
    if not _SHA256_RE.fullmatch(contract.source.sha256):
        errors.append("source.sha256 must be a 64-character hexadecimal checksum")
    if Path(contract.source.document).is_absolute():
        errors.append("source.document must be repository-relative")
    if len(contract.excluded_tables) != len(set(contract.excluded_tables)):
        errors.append("excluded_tables contains duplicate names")
    if set(contract.excluded_tables) != EXPECTED_EXCLUDED_TABLES:
        errors.append("excluded_tables does not match the approved exclusion list")

    table_names = [table.name for table in contract.tables]
    if len(table_names) != len(set(table_names)):
        errors.append("tables contains duplicate fully qualified names")
    table_map = contract.table_map
    if set(table_map).intersection(contract.excluded_tables):
        errors.append("included and excluded table scopes overlap")
    for table in contract.tables:
        _check_table(table, table_map, errors)

    totals = {
        "included_tables": len(contract.tables),
        "included_columns": contract.included_column_count,
        "documented_foreign_keys": contract.foreign_key_count,
        "estimated_rows": sum(table.estimated_rows or 0 for table in contract.tables),
    }
    if contract.summary.model_dump() != totals:
        errors.append(f"summary totals do not reconcile: expected {totals}")
    if totals != EXPECTED_SUMMARY:
        errors.append(
            f"summary totals must reconcile to the approved source totals: {EXPECTED_SUMMARY}"
        )
    if contract.source.views_documented != 0 or contract.source.sequences_documented != 0:
        errors.append("the source document records zero views and zero sequences")

    if errors:
        raise SchemaContractError("Invalid schema contract: " + "; ".join(errors))


def load_schema_contract(
    path: Path | None = None,
    project_root: Path | None = None,
) -> tuple[SchemaContract, str]:
    """Load, validate, and checksum the YAML contract without database access."""

    if path is not None and path.is_dir():
        contract_path = (path / "contracts" / "warranty_analytics_schema_v1.yaml").resolve()
    else:
        contract_path = (path or default_contract_path(project_root)).resolve()
    if not contract_path.is_file():
        raise SchemaContractError(f"Schema contract is missing: {contract_path}")
    try:
        with contract_path.open("r", encoding="utf-8") as stream:
            payload: Any = yaml.safe_load(stream)
    except OSError as exc:
        raise SchemaContractError(f"Could not read schema contract: {contract_path}") from exc
    except yaml.YAMLError as exc:
        raise SchemaContractError(f"Schema contract YAML is invalid: {contract_path}") from exc
    if not isinstance(payload, dict):
        raise SchemaContractError("Schema contract must contain a top-level mapping.")
    try:
        contract = SchemaContract.model_validate(payload)
    except ValidationError as exc:
        raise SchemaContractError(
            f"Schema contract fields are invalid: {_format_validation_errors(exc)}"
        ) from exc
    validate_contract(contract)
    return contract, contract_checksum(contract_path)
