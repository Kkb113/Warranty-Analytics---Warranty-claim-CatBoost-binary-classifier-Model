"""Orchestrate schema diffing into a stable validation result."""

from __future__ import annotations

from datetime import UTC, datetime
from time import monotonic
from uuid import uuid4

from .models import LiveSchemaMetadata, SchemaContract, ValidationResult
from .schema_diff import diff_schema


def validate_schema(
    contract: SchemaContract,
    live: LiveSchemaMetadata,
    contract_checksum: str,
    *,
    environment: str,
    strict: bool = False,
    server: str | None = None,
    started: float | None = None,
) -> ValidationResult:
    """Return a pass/fail result; schema errors are represented, not hidden."""

    timer = started if started is not None else monotonic()
    diff = diff_schema(contract, live, strict=strict)
    errors = sum(issue.severity == "ERROR" for issue in diff.issues)
    warnings = sum(issue.severity == "WARNING" for issue in diff.issues)
    infos = sum(issue.severity == "INFO" for issue in diff.issues)
    return ValidationResult(
        validation_id=uuid4().hex,
        contract_version=contract.contract_version,
        contract_checksum=contract_checksum,
        execution_timestamp=datetime.now(UTC),
        environment=environment,
        server=server or live.server_name,
        database=live.database_name,
        sql_version=live.sql_version,
        status="failed" if errors else "passed",
        included_table_count=len(contract.tables),
        included_column_count=contract.included_column_count,
        included_foreign_key_count=contract.foreign_key_count,
        actual_table_count=len(live.tables),
        actual_column_count=live.included_column_count,
        actual_foreign_key_count=live.foreign_key_count,
        error_count=errors,
        warning_count=warnings,
        info_count=infos,
        duration_seconds=monotonic() - timer,
        issues=diff.issues,
        excluded_objects=live.excluded_objects,
        row_count_differences=diff.row_count_differences,
    )
