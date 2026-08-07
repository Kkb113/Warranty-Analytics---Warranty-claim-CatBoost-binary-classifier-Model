"""CLI tests for offline and mocked live Phase 2 commands."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from warranty_analytics_model import cli
from warranty_analytics_model.config import DatabaseSettings
from warranty_analytics_model.database.models import ConnectivityResult, LiveSchemaMetadata
from warranty_analytics_model.database.schema_contract import load_schema_contract

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _live(contract, **updates) -> LiveSchemaMetadata:
    payload = {
        "database_name": contract.database.name,
        "server_name": "sql.example.test",
        "sql_version": "Microsoft SQL Server 2022",
        "tables": [table.model_copy(deep=True) for table in contract.tables],
        "all_table_names": [table.name for table in contract.tables],
        "schemas": ["dbo"],
    }
    payload.update(updates)
    return LiveSchemaMetadata(**payload)


def test_schema_contract_check_is_offline_and_successful(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public contract check succeeds without a configured database server."""

    assert cli.main(["schema-contract-check"]) == 0
    output = capsys.readouterr().out
    assert "16 tables" in output
    assert "209 columns" in output
    assert "22 foreign keys" in output


def test_schema_contract_check_invalid_path_returns_schema_exit_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An invalid offline contract is a schema/contract failure, not a connection failure."""

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("contract_version: bad\n", encoding="utf-8")
    assert cli.main(["schema-contract-check", "--contract", str(invalid)]) == 1
    assert "Schema contract error" in capsys.readouterr().err


def test_db_check_without_server_is_safe(capsys: pytest.CaptureFixture[str]) -> None:
    """A live command reports missing server configuration without attempting a driver load."""

    assert cli.main(["db-check"]) == 2
    error = capsys.readouterr().err
    assert "WARRANTY_DB_SERVER" in error
    assert "password" not in error.casefold()


def test_db_check_mocked_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI renders only safe connectivity fields on a mocked success."""

    monkeypatch.setenv("WARRANTY_DB_SERVER", "sql.example.test")
    monkeypatch.setattr(
        cli,
        "check_database_connection",
        lambda settings: ConnectivityResult(
            checked_at=datetime.now(UTC),
            server=settings.server,
            port=settings.port,
            expected_database=settings.database,
            actual_database=settings.database,
            sql_version="Microsoft SQL Server 2022",
            catalog_readable=True,
            duration_seconds=0.01,
        ),
    )

    assert cli.main(["db-check"]) == 0
    output = capsys.readouterr().out
    assert "database=warranty_analytics" in output
    assert "sql.example.test" not in output


def test_schema_validate_mocked_matching_writes_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A mocked live metadata match produces reports through the public command."""

    monkeypatch.setenv("WARRANTY_DB_SERVER", "sql.example.test")
    contract, checksum = load_schema_contract(REPOSITORY_ROOT)
    live = _live(contract)
    monkeypatch.setattr(cli, "check_database_connection", lambda settings: None)
    monkeypatch.setattr(cli, "collect_schema_metadata", lambda settings, approved: live)

    assert cli.main(["schema-validate", "--output-dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "Schema validation PASSED" in output
    reports = list(tmp_path.iterdir())
    assert {path.suffix for path in reports} == {".json", ".md"}
    assert checksum in reports[0].read_text(encoding="utf-8") or checksum in reports[1].read_text(
        encoding="utf-8"
    )


def test_schema_validate_blocking_and_strict_warning_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Blocking mismatches and strict warnings return the documented non-zero code."""

    monkeypatch.setenv("WARRANTY_DB_SERVER", "sql.example.test")
    contract, checksum = load_schema_contract(REPOSITORY_ROOT)
    monkeypatch.setattr(cli, "check_database_connection", lambda settings: None)

    monkeypatch.setattr(
        cli, "collect_schema_metadata", lambda settings, approved: _live(contract, tables=[])
    )
    assert cli.main(["schema-validate", "--no-report"]) == 1
    assert "FAILED" in capsys.readouterr().out

    changed = _live(
        contract,
        tables=[
            table.model_copy(update={"estimated_rows": (table.estimated_rows or 0) + 1})
            if table.name == contract.tables[0].name
            else table
            for table in contract.tables
        ],
    )
    monkeypatch.setattr(cli, "collect_schema_metadata", lambda settings, approved: changed)
    assert cli.main(["schema-validate", "--strict", "--no-report"]) == 1
    assert "errors=" in capsys.readouterr().out
    assert checksum


def test_schema_validate_does_not_need_database_for_invalid_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Contract failure is returned before any live operation is invoked."""

    monkeypatch.setenv("WARRANTY_DB_SERVER", "sql.example.test")
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("contract_version: bad\n", encoding="utf-8")
    called = False

    def fail_if_called(settings):
        nonlocal called
        called = True
        raise AssertionError("database should not be called")

    monkeypatch.setattr(cli, "check_database_connection", fail_if_called)
    assert cli.main(["schema-validate", "--contract", str(invalid), "--no-report"]) == 1
    assert called is False
    assert "Schema contract error" in capsys.readouterr().err


def test_importing_database_wrapper_does_not_connect() -> None:
    """A typed settings object can be created without any live side effect."""

    settings = DatabaseSettings()
    assert settings.server is None
