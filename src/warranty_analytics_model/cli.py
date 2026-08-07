"""Infrastructure-only command-line interface for the project."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import ConfigurationError, Settings, load_settings
from .database.connection import check_database_connection
from .database.exceptions import (
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseDriverError,
    SchemaContractError,
    UnexpectedDatabaseError,
)
from .database.metadata import collect_schema_metadata
from .database.reporting import write_validation_reports
from .database.schema_contract import load_schema_contract
from .database.schema_validator import validate_schema
from .logging_config import configure_logging
from .paths import discover_repository_root, resolve_project_paths


def build_parser() -> argparse.ArgumentParser:
    """Build the supported infrastructure CLI parser."""

    parser = argparse.ArgumentParser(
        prog="warranty-model",
        description="Infrastructure, Phase 2 schema, and Phase 3 data diagnostics for the warranty analytics project.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="Validate Phase 1 infrastructure only.")
    subparsers.add_parser("show-config", help="Show effective non-secret configuration.")
    subparsers.add_parser("version", help="Show the package version.")
    contract_check = subparsers.add_parser(
        "schema-contract-check",
        help="Validate the version-controlled schema contract without a database.",
    )
    contract_check.add_argument("--contract", type=Path, help="Optional contract YAML path.")
    subparsers.add_parser("db-check", help="Run safe live SQL Server connectivity checks.")
    schema_validate = subparsers.add_parser(
        "schema-validate",
        help="Validate live SQL Server catalog metadata against the schema contract.",
    )
    schema_validate.add_argument(
        "--strict", action="store_true", help="Promote warnings to blocking errors."
    )
    schema_validate.add_argument("--contract", type=Path, help="Optional contract YAML path.")
    schema_validate.add_argument("--output-dir", type=Path, help="Report directory override.")
    schema_validate.add_argument(
        "--no-report", action="store_true", help="Do not write JSON or Markdown reports."
    )
    schema_validate.add_argument(
        "--formats",
        nargs="+",
        choices=("json", "markdown"),
        default=("json", "markdown"),
        help="Report formats when reporting is enabled.",
    )
    for command, help_text in (
        ("data-profile", "Profile approved business tables and the warranty target."),
        ("synthetic-audit", "Run target, identifier, duplicate, group, and text audits."),
        ("data-quality-check", "Run referential, temporal, telemetry, and logical quality checks."),
        ("phase3-run", "Run the complete Phase 3 profiling and audit workflow."),
    ):
        phase3 = subparsers.add_parser(command, help=help_text)
        phase3.add_argument("--output-dir", type=Path, help="Report root override.")
        phase3.add_argument(
            "--no-charts", action="store_true", help="Do not generate optional charts."
        )
        phase3.add_argument(
            "--format",
            choices=("json", "markdown", "both"),
            default="both",
            help="Report format to write.",
        )
        phase3.add_argument(
            "--fail-on-error",
            action="store_true",
            help="Return a non-zero status when Phase 3 records ERROR findings.",
        )
    return parser


def run_doctor(project_root: Path | None = None) -> tuple[bool, list[str]]:
    """Run non-data infrastructure checks and return success plus messages."""

    messages: list[str] = []
    try:
        package = importlib.import_module("warranty_analytics_model")
        package_version = getattr(package, "__version__", None)
        if not isinstance(package_version, str) or not package_version:
            raise RuntimeError("Package version is unavailable.")
        messages.append(f"PASS: package import and version {package_version}")

        root = discover_repository_root(project_root)
        paths = resolve_project_paths(root)
        required_files = (
            paths.config_dir / "base.yaml",
            paths.config_dir / "development.yaml",
            paths.config_dir / "test.yaml",
        )
        missing_files = [str(path) for path in required_files if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(f"Missing configuration files: {', '.join(missing_files)}")
        messages.append("PASS: configuration files exist")

        settings = load_settings(root)
        messages.append(f"PASS: configuration loaded for environment {settings.environment}")

        resolved_paths = resolve_project_paths(
            root,
            data_dir=settings.data_dir,
            artifact_dir=settings.artifact_dir,
            report_dir=settings.report_dir,
            log_dir=settings.log_dir,
        )
        resolved_path_values = (
            resolved_paths.root,
            resolved_paths.config_dir,
            resolved_paths.data_dir,
            resolved_paths.artifact_dir,
            resolved_paths.report_dir,
            resolved_paths.log_dir,
        )
        if not all(path.is_absolute() for path in resolved_path_values):
            raise RuntimeError("One or more project paths did not resolve to absolute paths.")
        messages.append("PASS: required project paths resolve")

        configure_logging(settings.log_level)
        messages.append("PASS: logging initialized")
    except (ConfigurationError, FileNotFoundError, RuntimeError, ValueError) as exc:
        messages.append(f"FAIL: {exc}")
        return False, messages
    except Exception as exc:
        messages.append(f"FAIL: unexpected infrastructure error: {exc}")
        return False, messages

    return True, messages


def _run_doctor() -> int:
    """Render doctor messages and return a shell-friendly exit code."""

    success, messages = run_doctor()
    for message in messages:
        print(message)
    return 0 if success else 1


def _run_show_config() -> int:
    """Render effective redacted configuration and return a shell-friendly exit code."""

    try:
        settings = load_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(settings.redacted_dict(), indent=2, sort_keys=True))
    return 0


def _run_schema_contract_check(contract_path: Path | None) -> int:
    """Validate the YAML contract without importing or contacting a database."""

    try:
        contract, checksum = load_schema_contract(path=contract_path)
    except SchemaContractError as exc:
        print(f"Schema contract error: {exc}", file=sys.stderr)
        return 1
    print(
        "PASS: schema contract "
        f"{contract.contract_version} ({checksum}) reconciles "
        f"{len(contract.tables)} tables, {contract.included_column_count} columns, "
        f"{contract.foreign_key_count} foreign keys, "
        f"{contract.summary.estimated_rows} estimated rows."
    )
    return 0


def _load_live_settings() -> tuple[Settings | None, int]:
    """Load settings for a live command with the documented exit policy."""

    try:
        return load_settings(), 0
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return None, 2


def _run_db_check() -> int:
    """Run connectivity checks without reading business rows or writing reports."""

    settings, status = _load_live_settings()
    if settings is None:
        return status
    try:
        result = check_database_connection(settings.database)
    except DatabaseConfigurationError as exc:
        print(f"Database configuration error: {exc}", file=sys.stderr)
        return 2
    except DatabaseDriverError as exc:
        print(f"Database driver error: {exc}", file=sys.stderr)
        return 3
    except DatabaseConnectionError as exc:
        print(f"Database connection error: {exc}", file=sys.stderr)
        return 3
    except UnexpectedDatabaseError as exc:
        print(f"Unexpected database error: {exc}", file=sys.stderr)
        return 4
    print(
        "PASS: database check "
        f"database={result.actual_database} catalog_readable={result.catalog_readable} "
        f"duration_seconds={result.duration_seconds:.3f}"
    )
    return 0


def _run_schema_validate(arguments: argparse.Namespace) -> int:
    """Validate catalog metadata and optionally write explicit reports."""

    settings, status = _load_live_settings()
    if settings is None:
        return status
    try:
        contract, checksum = load_schema_contract(path=arguments.contract)
    except SchemaContractError as exc:
        print(f"Schema contract error: {exc}", file=sys.stderr)
        return 1
    try:
        check_database_connection(settings.database)
        live = collect_schema_metadata(settings.database, contract)
        result = validate_schema(
            contract,
            live,
            checksum,
            environment=settings.environment,
            strict=arguments.strict,
            server=settings.database.server,
        )
        report_paths: list[Path] = []
        if not arguments.no_report:
            root = discover_repository_root()
            project_paths = resolve_project_paths(root, report_dir=settings.report_dir)
            output_dir = arguments.output_dir or project_paths.report_dir / "schema_validation"
            report_paths = write_validation_reports(result, output_dir, arguments.formats)
    except DatabaseConfigurationError as exc:
        print(f"Database configuration error: {exc}", file=sys.stderr)
        return 2
    except DatabaseDriverError as exc:
        print(f"Database driver error: {exc}", file=sys.stderr)
        return 3
    except DatabaseConnectionError as exc:
        print(f"Database connection error: {exc}", file=sys.stderr)
        return 3
    except UnexpectedDatabaseError as exc:
        print(f"Unexpected database error: {exc}", file=sys.stderr)
        return 4
    except Exception as exc:
        print(f"Unexpected schema validation error: {exc}", file=sys.stderr)
        return 4
    print(
        f"Schema validation {result.status.upper()}: errors={result.error_count} "
        f"warnings={result.warning_count} info={result.info_count} "
        f"duration_seconds={result.duration_seconds:.3f}"
    )
    for report_path in report_paths:
        print(f"Report: {report_path}")
    return 0 if result.status == "passed" else 1


def _run_phase3(arguments: argparse.Namespace) -> int:
    """Run a live, read-only Phase 3 command with secret-safe console output."""

    settings, status = _load_live_settings()
    if settings is None:
        return status
    try:
        from .profiling.config import load_profiling_settings
        from .profiling.runner import run_live_phase3

        profiling = load_profiling_settings()
        formats = ("json", "markdown") if arguments.format == "both" else (arguments.format,)
        result = run_live_phase3(
            settings,
            profiling_settings=profiling,
            output_dir=arguments.output_dir,
            report_formats=formats,
            no_charts=arguments.no_charts,
        )
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except DatabaseConfigurationError as exc:
        print(f"Database configuration error: {exc}", file=sys.stderr)
        return 2
    except DatabaseDriverError as exc:
        print(f"Database driver error: {exc}", file=sys.stderr)
        return 3
    except DatabaseConnectionError as exc:
        print(f"Database connection error: {exc}", file=sys.stderr)
        return 3
    except (UnexpectedDatabaseError, RuntimeError, ValueError) as exc:
        print(f"Phase 3 execution error: {exc}", file=sys.stderr)
        return 4
    except Exception as exc:
        print(f"Unexpected Phase 3 error: {exc}", file=sys.stderr)
        return 4

    counts = result.get("finding_counts", {})
    target = result.get("target_profile", {})
    report = result.get("report_directory", "-")
    print("Phase 3 profiling completed")
    print(f"Tables profiled: {result.get('included_table_count', 0)}")
    print(f"Claims analyzed: {target.get('claims', 0) if isinstance(target, dict) else 0}")
    print(f"Data quality errors: {counts.get('ERROR', 0) if isinstance(counts, dict) else 0}")
    print(f"Warnings: {counts.get('WARNING', 0) if isinstance(counts, dict) else 0}")
    if isinstance(target, dict):
        print(f"Target positive rate: {target.get('positive_percentage', 0.0)}%")
    print(f"Overall status: {result.get('status', 'UNKNOWN')}")
    print(f"Report: {report}")
    if arguments.fail_on_error and isinstance(counts, dict) and counts.get("ERROR", 0):
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run one supported CLI command."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help()
        return 0
    if arguments.command == "doctor":
        return _run_doctor()
    if arguments.command == "show-config":
        return _run_show_config()
    if arguments.command == "version":
        print(__version__)
        return 0
    if arguments.command == "schema-contract-check":
        return _run_schema_contract_check(arguments.contract)
    if arguments.command == "db-check":
        return _run_db_check()
    if arguments.command == "schema-validate":
        return _run_schema_validate(arguments)
    if arguments.command in {"data-profile", "synthetic-audit", "data-quality-check", "phase3-run"}:
        return _run_phase3(arguments)
    parser.error(f"Unsupported command: {arguments.command}")
    return 2
