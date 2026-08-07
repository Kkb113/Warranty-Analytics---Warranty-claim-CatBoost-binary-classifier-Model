"""Validation result and report writer tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from warranty_analytics_model.database.models import (
    LiveSchemaMetadata,
    SchemaIssue,
    ValidationResult,
)
from warranty_analytics_model.database.reporting import write_validation_reports
from warranty_analytics_model.database.schema_contract import load_schema_contract
from warranty_analytics_model.database.schema_validator import validate_schema

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _matching_live(contract) -> LiveSchemaMetadata:
    return LiveSchemaMetadata(
        database_name="warranty_analytics",
        server_name="sql.example.test",
        sql_version="Microsoft SQL Server 2022",
        tables=[table.model_copy(deep=True) for table in contract.tables],
        all_table_names=[table.name for table in contract.tables],
        schemas=["dbo"],
    )


def test_validation_result_and_reports_are_secret_free(tmp_path: Path) -> None:
    """JSON and Markdown reports contain stable metadata and no credentials."""

    contract, checksum = load_schema_contract(REPOSITORY_ROOT)
    live = _matching_live(contract)
    result = validate_schema(
        contract,
        live,
        checksum,
        environment="development",
        server="sql.example.test",
    )
    paths = write_validation_reports(result, tmp_path)

    assert result.status == "passed"
    assert len(paths) == 2
    json_text = next(path.read_text(encoding="utf-8") for path in paths if path.suffix == ".json")
    markdown_text = next(path.read_text(encoding="utf-8") for path in paths if path.suffix == ".md")
    assert checksum in json_text
    assert "sql.example.test" in markdown_text
    assert "password" not in json_text.casefold()
    assert "fictional" not in json_text.casefold()
    assert "Schema validation" in markdown_text


def test_report_writer_supports_selected_formats_and_rejects_unknown(tmp_path: Path) -> None:
    """Report output is explicit and format selection is validated."""

    result = ValidationResult(
        validation_id="abc123",
        contract_version="1.0.0",
        contract_checksum="a" * 64,
        execution_timestamp=datetime.now(UTC),
        environment="test",
        server="sql.example.test",
        database="warranty_analytics",
        status="failed",
        included_table_count=16,
        included_column_count=209,
        included_foreign_key_count=22,
        actual_table_count=15,
        actual_column_count=208,
        actual_foreign_key_count=21,
        error_count=1,
        warning_count=0,
        info_count=0,
        duration_seconds=0.25,
        issues=[
            SchemaIssue(
                severity="ERROR",
                code="MISSING_TABLE",
                object_type="table",
                schema="dbo",
                table="dim_component",
                expected="dbo.dim_component",
                message="Required table is missing.",
                remediation="Restore the table.",
            )
        ],
    )
    paths = write_validation_reports(result, tmp_path, ("markdown",))
    assert len(paths) == 1
    assert paths[0].suffix == ".md"
    with pytest.raises(ValueError, match="Unsupported report"):
        write_validation_reports(result, tmp_path, ("xml",))


def test_diff_result_can_be_promoted_to_failed_validation() -> None:
    """The validator translates a blocking diff into the CLI/report status model."""

    contract, checksum = load_schema_contract(REPOSITORY_ROOT)
    live = _matching_live(contract).model_copy(update={"tables": []})
    result = validate_schema(contract, live, checksum, environment="test")
    assert result.status == "failed"
    assert result.error_count == len(
        [issue for issue in result.issues if issue.severity == "ERROR"]
    )
    assert result.actual_table_count == 0
