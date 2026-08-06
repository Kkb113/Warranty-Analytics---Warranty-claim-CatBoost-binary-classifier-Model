"""Compare the approved contract with read-only live catalog metadata."""

from __future__ import annotations

from typing import Any

from .exceptions import SchemaContractError
from .models import (
    ColumnSpec,
    ForeignKeySpec,
    LiveSchemaMetadata,
    SchemaContract,
    SchemaDiffResult,
    SchemaIssue,
    SQLTypeSpec,
    TableSpec,
)
from .schema_contract import validate_contract


def _issue(
    severity: str,
    code: str,
    object_type: str,
    message: str,
    remediation: str,
    *,
    table: TableSpec | None = None,
    column: str | None = None,
    constraint: str | None = None,
    expected: Any = None,
    actual: Any = None,
) -> SchemaIssue:
    return SchemaIssue(
        severity=severity,  # type: ignore[arg-type]
        code=code,
        object_type=object_type,
        schema=table.schema if table else None,
        table=table.table if table else None,
        column=column,
        constraint=constraint,
        expected=expected,
        actual=actual,
        message=message,
        remediation=remediation,
    )


def _type_value(type_spec: SQLTypeSpec) -> dict[str, object]:
    return type_spec.model_dump(exclude_none=True)


def _compare_type(
    expected: ColumnSpec,
    actual: ColumnSpec,
    table: TableSpec,
) -> list[SchemaIssue]:
    expected_type = expected.sql_type
    actual_type = actual.sql_type
    issues: list[SchemaIssue] = []
    if (
        expected_type.base_type != actual_type.base_type
        or expected_type.unicode != actual_type.unicode
    ):
        issues.append(
            _issue(
                "ERROR",
                "COLUMN_TYPE_MISMATCH",
                "column",
                f"Column {table.name}.{expected.name} has an incompatible SQL type.",
                "Review the live column type and restore the approved contract type before validation.",
                table=table,
                column=expected.name,
                expected=_type_value(expected_type),
                actual=_type_value(actual_type),
            )
        )
        return issues
    if expected_type.base_type in {"nvarchar", "varchar", "nchar", "char"}:
        if expected_type.is_max and not actual_type.is_max:
            issues.append(
                _issue(
                    "ERROR",
                    "COLUMN_LENGTH_NARROWER",
                    "column",
                    f"Column {table.name}.{expected.name} is narrower than the contract.",
                    "Restore the contract length or obtain an explicitly approved contract revision.",
                    table=table,
                    column=expected.name,
                    expected=_type_value(expected_type),
                    actual=_type_value(actual_type),
                )
            )
        elif not expected_type.is_max and actual_type.is_max:
            issues.append(
                _issue(
                    "WARNING",
                    "COLUMN_LENGTH_WIDENED",
                    "column",
                    f"Column {table.name}.{expected.name} is wider than the contract.",
                    "Confirm the widening is intentional; update the contract only through an explicit versioned change.",
                    table=table,
                    column=expected.name,
                    expected=_type_value(expected_type),
                    actual=_type_value(actual_type),
                )
            )
        elif (
            expected_type.max_length is not None
            and actual_type.max_length is not None
            and actual_type.max_length < expected_type.max_length
        ):
            issues.append(
                _issue(
                    "ERROR",
                    "COLUMN_LENGTH_NARROWER",
                    "column",
                    f"Column {table.name}.{expected.name} is narrower than the contract.",
                    "Restore the contract length or obtain an explicitly approved contract revision.",
                    table=table,
                    column=expected.name,
                    expected=_type_value(expected_type),
                    actual=_type_value(actual_type),
                )
            )
        elif (
            expected_type.max_length is not None
            and actual_type.max_length is not None
            and actual_type.max_length > expected_type.max_length
        ):
            issues.append(
                _issue(
                    "WARNING",
                    "COLUMN_LENGTH_WIDENED",
                    "column",
                    f"Column {table.name}.{expected.name} is wider than the contract.",
                    "Confirm the widening is intentional; update the contract only through an explicit versioned change.",
                    table=table,
                    column=expected.name,
                    expected=_type_value(expected_type),
                    actual=_type_value(actual_type),
                )
            )
    if expected_type.base_type in {"decimal", "numeric"} and (
        expected_type.precision != actual_type.precision or expected_type.scale != actual_type.scale
    ):
        issues.append(
            _issue(
                "ERROR",
                "COLUMN_PRECISION_SCALE_MISMATCH",
                "column",
                f"Column {table.name}.{expected.name} has incompatible decimal precision or scale.",
                "Restore the documented precision and scale before validation.",
                table=table,
                column=expected.name,
                expected=_type_value(expected_type),
                actual=_type_value(actual_type),
            )
        )
    return issues


def _compare_columns(expected: TableSpec, actual: TableSpec, issues: list[SchemaIssue]) -> None:
    expected_columns = expected.column_map
    actual_columns = actual.column_map
    for name, expected_column in expected_columns.items():
        actual_column = actual_columns.get(name)
        if actual_column is None:
            issues.append(
                _issue(
                    "ERROR",
                    "MISSING_COLUMN",
                    "column",
                    f"Required column is missing: {expected.name}.{name}.",
                    "Restore the missing column or create an explicitly approved contract revision.",
                    table=expected,
                    column=name,
                    expected=expected_column.model_dump(),
                )
            )
            continue
        issues.extend(_compare_type(expected_column, actual_column, expected))
        if expected_column.nullable != actual_column.nullable:
            issues.append(
                _issue(
                    "ERROR",
                    "COLUMN_NULLABILITY_MISMATCH",
                    "column",
                    f"Column {expected.name}.{name} has different nullability.",
                    "Restore the documented nullability before validation.",
                    table=expected,
                    column=name,
                    expected=expected_column.nullable,
                    actual=actual_column.nullable,
                )
            )
        if expected_column.identity != actual_column.identity:
            issues.append(
                _issue(
                    "ERROR",
                    "COLUMN_IDENTITY_MISMATCH",
                    "column",
                    f"Column {expected.name}.{name} has different identity metadata.",
                    "Restore the documented identity property before validation.",
                    table=expected,
                    column=name,
                    expected=expected_column.identity,
                    actual=actual_column.identity,
                )
            )
        if expected_column.computed != actual_column.computed:
            issues.append(
                _issue(
                    "ERROR",
                    "COLUMN_COMPUTED_MISMATCH",
                    "column",
                    f"Column {expected.name}.{name} has different computed metadata.",
                    "Restore the documented computed property before validation.",
                    table=expected,
                    column=name,
                    expected=expected_column.computed,
                    actual=actual_column.computed,
                )
            )
        if expected_column.collation != actual_column.collation:
            issues.append(
                _issue(
                    "WARNING",
                    "COLUMN_COLLATION_MISMATCH",
                    "column",
                    f"Column {expected.name}.{name} has a different collation.",
                    "Confirm collation compatibility and update the contract only through an explicit revision.",
                    table=expected,
                    column=name,
                    expected=expected_column.collation,
                    actual=actual_column.collation,
                )
            )
        if expected_column.default != actual_column.default:
            issues.append(
                _issue(
                    "WARNING",
                    "COLUMN_DEFAULT_MISMATCH",
                    "column",
                    f"Column {expected.name}.{name} has a different default definition.",
                    "Confirm the default is intentional; update the contract only through an explicit revision.",
                    table=expected,
                    column=name,
                    expected=expected_column.default,
                    actual=actual_column.default,
                )
            )
    for name, actual_column in actual_columns.items():
        if name not in expected_columns:
            issues.append(
                _issue(
                    "WARNING",
                    "EXTRA_COLUMN",
                    "column",
                    f"Live table has an additional column: {actual.name}.{name}.",
                    "Assess whether the column is approved and update the contract explicitly if it is in scope.",
                    table=expected,
                    column=name,
                    actual=actual_column.model_dump(),
                )
            )


def _compare_primary_keys(
    expected: TableSpec, actual: TableSpec, issues: list[SchemaIssue]
) -> None:
    expected_key = expected.primary_key
    actual_key = actual.primary_key
    if expected_key is None:
        return
    if actual_key is None:
        issues.append(
            _issue(
                "ERROR",
                "MISSING_PRIMARY_KEY",
                "primary_key",
                f"Required primary key is missing: {expected.name}.",
                "Restore the documented primary key before validation.",
                table=expected,
                constraint=expected_key.name,
                expected=expected_key.model_dump(),
            )
        )
        return
    if expected_key.name != actual_key.name or expected_key.columns != actual_key.columns:
        issues.append(
            _issue(
                "ERROR",
                "PRIMARY_KEY_MISMATCH",
                "primary_key",
                f"Primary key differs for {expected.name}.",
                "Restore the documented primary-key name and ordered columns.",
                table=expected,
                constraint=expected_key.name,
                expected=expected_key.model_dump(),
                actual=actual_key.model_dump(),
            )
        )


def _foreign_key_signature(foreign_key: ForeignKeySpec) -> tuple[object, ...]:
    return (
        foreign_key.referenced_table,
        tuple(foreign_key.parent_columns),
        tuple(foreign_key.referenced_columns),
        foreign_key.on_delete,
        foreign_key.on_update,
    )


def _compare_foreign_keys(
    expected: TableSpec, actual: TableSpec, issues: list[SchemaIssue]
) -> None:
    expected_keys = {foreign_key.name: foreign_key for foreign_key in expected.foreign_keys}
    actual_keys = {foreign_key.name: foreign_key for foreign_key in actual.foreign_keys}
    for name, expected_key in expected_keys.items():
        actual_key = actual_keys.get(name)
        if actual_key is None:
            issues.append(
                _issue(
                    "ERROR",
                    "MISSING_FOREIGN_KEY",
                    "foreign_key",
                    f"Required foreign key is missing: {expected.name}.{name}.",
                    "Restore the documented foreign-key mapping before validation.",
                    table=expected,
                    constraint=name,
                    expected=expected_key.model_dump(),
                )
            )
            continue
        if _foreign_key_signature(expected_key) != _foreign_key_signature(actual_key):
            issues.append(
                _issue(
                    "ERROR",
                    "FOREIGN_KEY_MAPPING_MISMATCH",
                    "foreign_key",
                    f"Foreign-key mapping differs: {expected.name}.{name}.",
                    "Restore the documented parent and referenced columns/actions.",
                    table=expected,
                    constraint=name,
                    expected=expected_key.model_dump(),
                    actual=actual_key.model_dump(),
                )
            )
        if expected_key.trusted != actual_key.trusted:
            issues.append(
                _issue(
                    "WARNING",
                    "FOREIGN_KEY_TRUST_MISMATCH",
                    "foreign_key",
                    f"Foreign-key trust metadata differs: {expected.name}.{name}.",
                    "Confirm constraint trust state and remediate before relying on referential assumptions.",
                    table=expected,
                    constraint=name,
                    expected=expected_key.trusted,
                    actual=actual_key.trusted,
                )
            )
    for name, actual_key in actual_keys.items():
        if name not in expected_keys:
            issues.append(
                _issue(
                    "WARNING",
                    "EXTRA_FOREIGN_KEY",
                    "foreign_key",
                    f"Live table has an additional foreign key: {actual.name}.{name}.",
                    "Assess whether the constraint is approved and update the contract explicitly if needed.",
                    table=expected,
                    constraint=name,
                    actual=actual_key.model_dump(),
                )
            )


def _compare_indexes(expected: TableSpec, actual: TableSpec, issues: list[SchemaIssue]) -> None:
    expected_indexes = {index.name: index for index in expected.indexes}
    actual_indexes = {index.name: index for index in actual.indexes}
    for name, expected_index in expected_indexes.items():
        actual_index = actual_indexes.get(name)
        if actual_index is None:
            issues.append(
                _issue(
                    "WARNING",
                    "MISSING_INDEX",
                    "index",
                    f"Documented index is missing: {expected.name}.{name}.",
                    "Confirm performance and constraint implications; restore or explicitly revise the contract.",
                    table=expected,
                    constraint=name,
                    expected=expected_index.model_dump(),
                )
            )
        elif expected_index.model_dump(exclude={"name"}) != actual_index.model_dump(
            exclude={"name"}
        ):
            issues.append(
                _issue(
                    "WARNING",
                    "INDEX_MISMATCH",
                    "index",
                    f"Index metadata differs: {expected.name}.{name}.",
                    "Confirm the live index definition and update the contract only through an explicit revision.",
                    table=expected,
                    constraint=name,
                    expected=expected_index.model_dump(),
                    actual=actual_index.model_dump(),
                )
            )
    for name, actual_index in actual_indexes.items():
        if name not in expected_indexes:
            issues.append(
                _issue(
                    "WARNING",
                    "EXTRA_INDEX",
                    "index",
                    f"Live table has an additional index: {actual.name}.{name}.",
                    "Assess whether the index is approved and record it through an explicit contract revision if needed.",
                    table=expected,
                    constraint=name,
                    actual=actual_index.model_dump(),
                )
            )


def _compare_table(expected: TableSpec, actual: TableSpec, issues: list[SchemaIssue]) -> int:
    _compare_columns(expected, actual, issues)
    _compare_primary_keys(expected, actual, issues)
    _compare_foreign_keys(expected, actual, issues)
    _compare_indexes(expected, actual, issues)
    if expected.properties.model_dump() != actual.properties.model_dump():
        issues.append(
            _issue(
                "WARNING",
                "TABLE_PROPERTY_MISMATCH",
                "table",
                f"Storage or temporal properties differ: {expected.name}.",
                "Confirm the live table properties and update the contract only through an explicit revision.",
                table=expected,
                expected=expected.properties.model_dump(),
                actual=actual.properties.model_dump(),
            )
        )
    if expected.estimated_rows != actual.estimated_rows:
        issues.append(
            _issue(
                "WARNING",
                "ROW_COUNT_ESTIMATE_DIFF",
                "table",
                f"Partition row estimate differs for {expected.name}; this is non-blocking.",
                "Investigate the estimate if needed; row-count drift does not block schema compatibility.",
                table=expected,
                expected=expected.estimated_rows,
                actual=actual.estimated_rows,
            )
        )
        return 1
    return 0


def diff_schema(
    contract: SchemaContract,
    live: LiveSchemaMetadata,
    *,
    strict: bool = False,
) -> SchemaDiffResult:
    """Return all contract-versus-live issues without reading business rows."""

    issues: list[SchemaIssue] = []
    try:
        validate_contract(contract)
    except SchemaContractError as exc:
        issues.append(
            _issue(
                "ERROR",
                "CONTRACT_INVALID",
                "contract",
                "The schema contract failed self-validation.",
                "Repair the version-controlled contract and rerun the offline contract check.",
                actual=str(exc),
            )
        )
        return SchemaDiffResult(issues=issues)

    if live.database_name.casefold() != contract.database.name.casefold():
        issues.append(
            _issue(
                "ERROR",
                "DATABASE_NAME_MISMATCH",
                "database",
                "Connected database does not match the approved contract database.",
                "Connect to warranty_analytics or obtain an explicit approved override.",
                expected=contract.database.name,
                actual=live.database_name,
            )
        )
    if not live.catalog_readable:
        issues.append(
            _issue(
                "ERROR",
                "CATALOG_METADATA_UNREADABLE",
                "catalog",
                "Required SQL Server catalog metadata could not be read.",
                "Grant read access to the catalog views required by Phase 2.",
            )
        )

    expected_tables = contract.table_map
    actual_tables = live.table_map
    row_count_differences = 0
    for name, expected_table in expected_tables.items():
        actual_table = actual_tables.get(name)
        if actual_table is None:
            issues.append(
                _issue(
                    "ERROR",
                    "MISSING_TABLE",
                    "table",
                    f"Required table is missing: {name}.",
                    "Restore the included table or stop until an approved contract revision exists.",
                    table=expected_table,
                    expected=name,
                )
            )
            continue
        row_count_differences += _compare_table(expected_table, actual_table, issues)

    actual_names = set(live.all_table_names or actual_tables)
    for name in sorted(actual_names - set(expected_tables)):
        if name in contract.excluded_tables:
            issues.append(
                _issue(
                    "INFO",
                    "EXCLUDED_OBJECT_PRESENT",
                    "excluded_object",
                    f"Excluded ML object is present by name only: {name}.",
                    "Keep the object excluded; do not inspect, validate, or use its columns.",
                    actual=name,
                )
            )
        else:
            issues.append(
                _issue(
                    "WARNING",
                    "EXTRA_TABLE",
                    "table",
                    f"Live schema contains a table outside the approved scope: {name}.",
                    "Assess scope explicitly; do not add the table to modeling inputs implicitly.",
                    actual=name,
                )
            )
    for name in sorted(set(contract.excluded_tables) - actual_names):
        issues.append(
            _issue(
                "INFO",
                "EXCLUDED_OBJECT_ABSENT",
                "excluded_object",
                f"Excluded ML object is absent by name: {name}.",
                "Keep the object excluded unless the project scope is explicitly revised.",
                actual=name,
            )
        )
    for schema in sorted(set(live.schemas) - {table.schema for table in contract.tables}):
        issues.append(
            _issue(
                "INFO",
                "ADDITIONAL_SCHEMA",
                "schema",
                f"Additional database schema is present: {schema}.",
                "Keep additional schemas outside the approved modeling scope unless explicitly reviewed.",
                actual=schema,
            )
        )
    for view in live.views:
        issues.append(
            _issue(
                "INFO",
                "ADDITIONAL_VIEW",
                "view",
                f"Additional view is present: {view}.",
                "Do not use the view for Phase 2 validation unless scope is explicitly revised.",
                actual=view,
            )
        )
    for sequence in live.sequences:
        issues.append(
            _issue(
                "INFO",
                "ADDITIONAL_SEQUENCE",
                "sequence",
                f"Additional sequence is present: {sequence}.",
                "Do not use the sequence for Phase 2 validation unless scope is explicitly revised.",
                actual=sequence,
            )
        )

    if strict:
        issues = [
            issue.model_copy(update={"severity": "ERROR"}) if issue.severity == "WARNING" else issue
            for issue in issues
        ]
    return SchemaDiffResult(issues=issues, row_count_differences=row_count_differences)
