"""Infrastructure-only command-line interface for the project."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

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
from .feature_mart.models import FeatureMartError
from .logging_config import configure_logging
from .paths import discover_repository_root, resolve_project_paths

_PHASE3_TASK_GROUPS = {
    "data-profile": ("data_profile",),
    "synthetic-audit": ("synthetic_audit",),
    "data-quality-check": ("data_quality",),
    "phase3-run": ("data_profile", "synthetic_audit", "data_quality"),
}


def build_parser() -> argparse.ArgumentParser:
    """Build the supported infrastructure CLI parser."""

    parser = argparse.ArgumentParser(
        prog="warranty-model",
        description="Infrastructure, schema, profiling, policy enforcement, and claim-mart commands for the warranty analytics project.",
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
        ("data-profile", "Run table, column, target, category, and missingness profiling."),
        (
            "synthetic-audit",
            "Run target-generation, leakage, identifier, duplicate, group, and text audits.",
        ),
        (
            "data-quality-check",
            "Run referential, temporal, telemetry, maintenance, service/repair, and component/supplier checks.",
        ),
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
    subparsers.add_parser(
        "phase4-contract-check",
        help="Validate Phase 4 target, feature-availability, and leakage contracts offline.",
    )
    phase4 = subparsers.add_parser(
        "phase4-validate",
        help="Run the live read-only Phase 4 target and policy audit.",
    )
    phase4.add_argument(
        "--strict", action="store_true", help="Treat documented warnings as blocking."
    )
    phase4.add_argument("--output-dir", type=Path, help="Report root override.")
    phase4.add_argument("--no-report", action="store_true", help="Do not write Phase 4 reports.")
    phase4.add_argument(
        "--format",
        choices=("json", "markdown", "both"),
        default="both",
        help="Report format to write.",
    )
    subparsers.add_parser(
        "phase5-plan-check",
        help="Validate the Phase 5 mart contract and extraction plan offline.",
    )
    phase5_build = subparsers.add_parser(
        "phase5-build",
        help="Build the Phase 5 claim snapshot and history bundle from read-only SQL Server data.",
    )
    phase5_build.add_argument("--output-dir", type=Path, help="Feature-mart output root override.")
    phase5_build.add_argument("--report-dir", type=Path, help="Phase 5 report root override.")
    phase5_build.add_argument("--no-report", action="store_true", help="Do not write reports.")
    phase5_build.add_argument(
        "--overwrite", action="store_true", help="Allow replacement of a matching completed run."
    )
    phase5_build.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing mart directory instead of reading SQL Server.",
    )
    phase5_build.add_argument(
        "--mart-dir", type=Path, help="Existing mart directory used with --validate-only."
    )
    phase5_validate = subparsers.add_parser(
        "phase5-validate",
        help="Validate an existing Phase 5 mart bundle without database access.",
    )
    phase5_validate.add_argument("--mart-dir", type=Path, required=True, help="Mart run directory.")
    subparsers.add_parser(
        "phase6-contract-check",
        help="Validate the Phase 6 split contract and configuration offline.",
    )
    phase6_plan = subparsers.add_parser(
        "phase6-plan-check",
        help="Validate Phase 5 input compatibility and the Phase 6 split plan offline.",
    )
    phase6_plan.add_argument(
        "--mart-dir", type=Path, required=True, help="Phase 5 mart run directory."
    )
    phase6_build = subparsers.add_parser(
        "phase6-build",
        help="Build a deterministic chronological Phase 6 split from a Phase 5 mart.",
    )
    phase6_build.add_argument(
        "--mart-dir", type=Path, required=True, help="Phase 5 mart run directory."
    )
    phase6_build.add_argument("--output-dir", type=Path, help="Split artifact root override.")
    phase6_build.add_argument("--report-dir", type=Path, help="Phase 6 report root override.")
    phase6_build.add_argument("--run-id", help="Explicit immutable Phase 6 run identifier.")
    phase6_build.add_argument(
        "--overwrite", action="store_true", help="Replace only an incomplete output directory."
    )
    phase6_build.add_argument("--no-report", action="store_true", help="Do not write reports.")
    phase6_validate = subparsers.add_parser(
        "phase6-validate",
        help="Validate an existing Phase 6 split run without database access.",
    )
    phase6_validate.add_argument(
        "--split-dir", type=Path, required=True, help="Split run directory."
    )
    subparsers.add_parser(
        "phase7-contract-check",
        help="Validate the Phase 7 structured feature contract offline.",
    )
    phase7_plan = subparsers.add_parser(
        "phase7-plan-check",
        help="Validate Phase 5/6 inputs and the Phase 7 feature plan offline.",
    )
    phase7_plan.add_argument(
        "--mart-dir", type=Path, required=True, help="Phase 5 mart run directory."
    )
    phase7_plan.add_argument(
        "--split-dir", type=Path, required=True, help="Corrected Phase 6 split run directory."
    )
    phase7_build = subparsers.add_parser(
        "phase7-build",
        help="Build deterministic structured features from an existing Phase 5/6 artifact bundle.",
    )
    phase7_build.add_argument(
        "--mart-dir", type=Path, required=True, help="Phase 5 mart run directory."
    )
    phase7_build.add_argument(
        "--split-dir", type=Path, required=True, help="Corrected Phase 6 split run directory."
    )
    phase7_build.add_argument(
        "--output-dir", type=Path, help="Structured-feature artifact root override."
    )
    phase7_build.add_argument("--report-dir", type=Path, help="Phase 7 report root override.")
    phase7_build.add_argument("--run-id", help="Explicit immutable Phase 7 run identifier.")
    phase7_build.add_argument(
        "--overwrite", action="store_true", help="Replace an existing local Phase 7 run."
    )
    phase7_build.add_argument(
        "--no-report", action="store_true", help="Do not write aggregate reports."
    )
    phase7_validate = subparsers.add_parser(
        "phase7-validate",
        help="Validate a completed Phase 7 structured-feature run offline.",
    )
    phase7_validate.add_argument(
        "--feature-dir", type=Path, required=True, help="Structured-feature run directory."
    )
    return parser


def phase3_task_groups(command: str) -> tuple[str, ...]:
    """Map a Phase 3 CLI command to shared execution task groups."""

    try:
        return _PHASE3_TASK_GROUPS[command]
    except KeyError as exc:
        raise ValueError(f"Unsupported Phase 3 command: {command}") from exc


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
            paths.root / "configs" / "feature_mart.yaml",
            paths.root / "configs" / "splits.yaml",
            paths.root / "contracts" / "claim_split_v1.yaml",
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
            task_groups=phase3_task_groups(arguments.command),
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
    print(f"Phase 3 command completed: {arguments.command}")
    print(f"Task groups: {', '.join(phase3_task_groups(arguments.command))}")
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


def _run_phase4_contract_check() -> int:
    """Validate all Phase 4 contracts without database access."""

    try:
        from .policy.loader import load_phase4_contracts
        from .policy.validator import validate_phase4_contracts

        root = discover_repository_root()
        schema_contract, schema_checksum = load_schema_contract(root)
        bundle = load_phase4_contracts(root)
        result = validate_phase4_contracts(
            schema_contract,
            bundle.target,
            bundle.feature_policy,
            bundle.leakage,
            checksums={
                "high_cost_target_v1.yaml": bundle.target_checksum,
                "claim_time_feature_policy_v1.yaml": bundle.feature_policy_checksum,
                "leakage_policy_v1.yaml": bundle.leakage_checksum,
            },
            schema_contract_checksum=schema_checksum,
        )
    except (SchemaContractError, ValueError) as exc:
        print(f"Phase 4 contract error: {exc}", file=sys.stderr)
        return 1
    if not result.valid:
        print("Phase 4 contract check BLOCKED")
        for error in result.errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "PASS: Phase 4 contracts "
        f"cover {result.classified_columns}/{result.schema_columns} schema columns; "
        f"Tier A={len(result.safe_baseline_allowlist)} "
        f"Tier B={len(result.restricted_experimental_list)} "
        f"requires_confirmation={len(result.requires_confirmation_list)}"
    )
    print(f"Target policy checksum: {result.target_checksum}")
    print(f"Feature policy checksum: {result.feature_policy_checksum}")
    print(f"Leakage policy checksum: {result.leakage_policy_checksum}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    return 0


def _run_phase4_validate(arguments: argparse.Namespace) -> int:
    """Run the live read-only Phase 4 target and policy audit."""

    settings, status = _load_live_settings()
    if settings is None:
        return status
    try:
        from .policy.live import run_live_phase4
        from .policy.loader import load_phase4_contracts
        from .policy.reporting import write_phase4_reports

        root = discover_repository_root()
        schema_contract, schema_checksum = load_schema_contract(root)
        bundle = load_phase4_contracts(root)
        check_database_connection(settings.database)
        live = collect_schema_metadata(settings.database, schema_contract)
        schema_result = validate_schema(
            schema_contract,
            live,
            schema_checksum,
            environment=settings.environment,
            strict=arguments.strict,
            server=settings.database.server,
        )
        result = run_live_phase4(
            settings,
            schema_contract,
            bundle,
            schema_validation=schema_result.model_dump(mode="json"),
            schema_contract_checksum=schema_checksum,
        )
        report_paths: list[Path] = []
        if not arguments.no_report:
            project_paths = resolve_project_paths(root, report_dir=settings.report_dir)
            output_root = arguments.output_dir or project_paths.report_dir / "phase4_validation"
            formats = ("json", "markdown") if arguments.format == "both" else (arguments.format,)
            report_paths = write_phase4_reports(
                result,
                bundle.feature_policy,
                bundle.leakage,
                output_root,
                formats,
            )
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except SchemaContractError as exc:
        print(f"Schema contract error: {exc}", file=sys.stderr)
        return 1
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
        print(f"Phase 4 execution error: {exc}", file=sys.stderr)
        return 4
    except Exception as exc:
        print(f"Unexpected Phase 4 error: {exc}", file=sys.stderr)
        return 4

    target = result.target_validation
    print(f"Phase 4 status: {result.status}")
    print(f"Target: {target.get('target_name', '-')}")
    print(f"Claims: {target.get('total_claims', 0)}; eligible: {target.get('eligible_claims', 0)}")
    print(f"Positive prevalence: {target.get('positive_percentage', 0.0)}%")
    print(
        f"Policy coverage: {result.contract_validation.classified_columns}/{result.contract_validation.schema_columns}"
    )
    print(f"Errors: {len(result.errors)}; warnings: {len(result.warnings)}")
    for report_path in report_paths:
        print(f"Report: {report_path}")
    if result.errors or (arguments.strict and result.warnings):
        return 1
    return 0


def _render_phase5_validation(validation: dict[str, object]) -> None:
    """Print aggregate Phase 5 validation output without record-level values."""

    snapshot = validation.get("snapshot", {})
    if isinstance(snapshot, dict):
        print(
            f"Snapshot rows: {snapshot.get('rows', 0)}; "
            f"unique claims: {snapshot.get('unique_claims', 0)}; "
            f"columns: {snapshot.get('columns', 0)}"
        )
        print(
            f"Target positive/negative: {snapshot.get('positive_claims', 0)}/"
            f"{snapshot.get('negative_claims', 0)}"
        )
    errors = cast(list[object], validation.get("errors", []))
    warnings = cast(list[object], validation.get("warnings", []))
    print(f"Errors: {len(errors)}; warnings: {len(warnings)}")
    for warning in warnings:
        print(f"WARNING: {warning}")


def _run_phase5_plan_check() -> int:
    """Validate Phase 5 contracts and mappings without a database."""

    try:
        from .feature_mart.extraction_plan import build_extraction_plan
        from .feature_mart.mart_contract import load_mart_contract, validate_mart_contract
        from .policy.loader import load_phase4_contracts

        root = discover_repository_root()
        schema_contract, schema_checksum = load_schema_contract(root)
        phase4_bundle = load_phase4_contracts(root)
        mart_contract, mart_checksum = load_mart_contract(root)
        result = validate_mart_contract(
            schema_contract,
            phase4_bundle,
            schema_contract_checksum=schema_checksum,
            contract=mart_contract,
            contract_checksum=mart_checksum,
        )
        if result.valid:
            build_extraction_plan(schema_contract, phase4_bundle, mart_contract)
    except (FeatureMartError, SchemaContractError, ValueError) as exc:
        print(f"Phase 5 plan check BLOCKED: {exc}", file=sys.stderr)
        return 1
    if not result.valid:
        print("Phase 5 plan check BLOCKED")
        for error in result.errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "PASS: Phase 5 plan "
        f"direct={result.direct_materialized}/{result.direct_expected} "
        f"historical={result.historical_mapped}/{result.historical_expected} "
        f"mart_contract_checksum={result.mart_contract_checksum}"
    )
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    return 0


def _run_phase5_build(arguments: argparse.Namespace) -> int:
    """Run live read-only Phase 5 construction or artifact-only validation."""

    if arguments.validate_only:
        if arguments.mart_dir is None:
            print("--mart-dir is required with --validate-only.", file=sys.stderr)
            return 2
        try:
            from .feature_mart.runner import validate_existing_mart

            validation = validate_existing_mart(arguments.mart_dir)
        except (FeatureMartError, FileNotFoundError, ValueError) as exc:
            print(f"Phase 5 validation error: {exc}", file=sys.stderr)
            return 1
        print(f"Phase 5 validation: {validation.get('status', 'BLOCKED')}")
        _render_phase5_validation(validation)
        return 0 if not validation.get("errors") else 1

    settings, status = _load_live_settings()
    if settings is None:
        return status
    try:
        from .feature_mart.runner import run_live_phase5

        result = run_live_phase5(
            settings,
            output_dir=arguments.output_dir,
            report_dir=arguments.report_dir,
            no_report=arguments.no_report,
            overwrite=arguments.overwrite,
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
    except (FeatureMartError, UnexpectedDatabaseError, RuntimeError, ValueError, KeyError) as exc:
        print(f"Phase 5 build error: {exc}", file=sys.stderr)
        return 4
    except Exception as exc:
        print(f"Unexpected Phase 5 error: {exc}", file=sys.stderr)
        return 4
    print(f"Phase 5 status: {result.status}")
    print(f"Mart run directory: {result.run_directory}")
    _render_phase5_validation(result.validation)
    if result.report_directory:
        print(f"Report directory: {result.report_directory}")
    return 0 if result.status in {"PASS", "PASS WITH WARNINGS"} else 1


def _run_phase5_validate(arguments: argparse.Namespace) -> int:
    """Run offline validation for an existing Phase 5 bundle."""

    try:
        from .feature_mart.runner import validate_existing_mart

        validation = validate_existing_mart(arguments.mart_dir)
    except (FeatureMartError, FileNotFoundError, ValueError) as exc:
        print(f"Phase 5 validation error: {exc}", file=sys.stderr)
        return 1
    print(f"Phase 5 validation: {validation.get('status', 'BLOCKED')}")
    _render_phase5_validation(validation)
    return 0 if not validation.get("errors") else 1


def _render_phase6_validation(validation: dict[str, object]) -> None:
    """Print aggregate Phase 6 validation output without record-level values."""

    print(f"Errors: {len(cast(list[object], validation.get('errors', [])))}")
    print(f"Warnings: {len(cast(list[object], validation.get('warnings', [])))}")
    counts = validation.get("split_counts", {})
    if isinstance(counts, dict):
        for split in ("TRAIN", "VALIDATION", "TEST"):
            item = counts.get(split, {})
            if isinstance(item, dict):
                print(
                    f"{split}: rows={item.get('row_count', 0)} "
                    f"positive/negative={item.get('positive_count', 0)}/"
                    f"{item.get('negative_count', 0)}"
                )
    for warning in cast(list[object], validation.get("warnings", [])):
        print(f"WARNING: {warning}")


def _run_phase6_contract_check() -> int:
    """Validate the Phase 6 contract without reading a mart or database."""

    try:
        from .splits.split_contract import validate_current_split_contract

        result = validate_current_split_contract()
    except (FeatureMartError, ValueError, OSError) as exc:
        print(f"Phase 6 contract check BLOCKED: {exc}", file=sys.stderr)
        return 1
    if not result.valid:
        print("Phase 6 contract check BLOCKED")
        for error in result.errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "PASS: Phase 6 split contract "
        f"fractions={result.requested_fractions} "
        f"split_contract_checksum={result.split_contract_checksum}"
    )
    return 0


def _run_phase6_plan_check(arguments: argparse.Namespace) -> int:
    """Validate the Phase 5 input and Phase 6 plan without creating assignments."""

    try:
        from .splits.runner import phase6_plan_check

        result = phase6_plan_check(arguments.mart_dir)
    except (FeatureMartError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"Phase 6 plan check BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(f"Phase 6 plan check: {result.get('status', 'BLOCKED')}")
    print(f"Phase 5 mart: {result.get('mart_run', '-')}")
    print(f"Errors: {len(result.get('errors', []))}; warnings: {len(result.get('warnings', []))}")
    for error in result.get("errors", []):
        print(f"ERROR: {error}")
    for warning in result.get("warnings", []):
        print(f"WARNING: {warning}")
    return 0 if result.get("valid") else 1


def _run_phase6_build(arguments: argparse.Namespace) -> int:
    """Build a local Phase 6 split from a completed Phase 5 mart."""

    try:
        from .splits.runner import build_phase6

        result = build_phase6(
            arguments.mart_dir,
            output_dir=arguments.output_dir,
            report_dir=arguments.report_dir,
            run_id=arguments.run_id,
            overwrite=arguments.overwrite,
            no_report=arguments.no_report,
        )
    except (FeatureMartError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"Phase 6 build BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(f"Phase 6 status: {result.status}")
    print(f"Split run directory: {result.run_directory}")
    _render_phase6_validation(result.validation)
    if result.report_directory:
        print(f"Report directory: {result.report_directory}")
    return 0 if result.status in {"PASS", "PASS WITH WARNINGS"} else 1


def _run_phase6_validate(arguments: argparse.Namespace) -> int:
    """Validate a completed Phase 6 run entirely offline."""

    try:
        from .splits.runner import validate_existing_split

        validation = validate_existing_split(arguments.split_dir)
    except (FeatureMartError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"Phase 6 validation BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(f"Phase 6 validation: {validation.get('status', 'BLOCKED')}")
    _render_phase6_validation(validation)
    return 0 if not validation.get("errors") else 1


def _run_phase7_contract_check() -> int:
    """Validate the Phase 7 contract without reading a mart or database."""

    try:
        from .structured_features.runner import phase7_contract_check

        result = phase7_contract_check()
    except Exception as exc:
        print(f"Phase 7 contract check BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(f"Phase 7 contract check: {result.get('status', 'BLOCKED')}")
    print(f"Contract version: {result.get('contract_version', '-')}")
    print(f"Contract SHA-256: {result.get('contract_checksum', '-')}")
    for error in result.get("errors", []):
        print(f"ERROR: {error}")
    for warning in result.get("warnings", []):
        print(f"WARNING: {warning}")
    return 0 if result.get("valid") else 1


def _run_phase7_plan_check(arguments: argparse.Namespace) -> int:
    """Validate Phase 5/6 inputs and the Phase 7 source plan."""

    try:
        from .structured_features.input import phase7_plan_check

        result = phase7_plan_check(arguments.mart_dir, arguments.split_dir)
    except Exception as exc:
        print(f"Phase 7 plan check BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(f"Phase 7 plan check: {result.get('status', 'BLOCKED')}")
    for error in result.get("errors", []):
        print(f"ERROR: {error}")
    for warning in result.get("warnings", []):
        print(f"WARNING: {warning}")
    return 0 if result.get("valid") else 1


def _run_phase7_build(arguments: argparse.Namespace) -> int:
    """Build the offline Phase 7 structured feature artifact."""

    try:
        from .structured_features.runner import build_phase7

        result = build_phase7(
            arguments.mart_dir,
            arguments.split_dir,
            output_dir=arguments.output_dir,
            report_dir=arguments.report_dir,
            run_id=arguments.run_id,
            overwrite=arguments.overwrite,
            no_report=arguments.no_report,
        )
    except Exception as exc:
        print(f"Phase 7 build BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(f"Phase 7 status: {result.get('status', 'BLOCKED')}")
    print(f"Feature run directory: {result.get('run_directory', '-')}")
    if result.get("report_directory"):
        print(f"Report directory: {result['report_directory']}")
    validation = result.get("validation", {})
    print(
        f"Rows: {validation.get('row_count', 0)}; model features: {validation.get('model_feature_count', 0)}"
    )
    for warning in result.get("warnings", []):
        print(f"WARNING: {warning}")
    return 0 if result.get("status") in {"PASS", "PASS WITH WARNINGS"} else 1


def _run_phase7_validate(arguments: argparse.Namespace) -> int:
    """Validate a completed Phase 7 run without database access."""

    try:
        from .structured_features.runner import validate_existing_feature_run

        validation = validate_existing_feature_run(arguments.feature_dir)
    except Exception as exc:
        print(f"Phase 7 validation BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(f"Phase 7 validation: {validation.get('status', 'BLOCKED')}")
    print(f"Rows: {validation.get('row_count', 0)}; columns: {validation.get('column_count', 0)}")
    print(
        f"Errors: {len(validation.get('errors', []))}; warnings: {len(validation.get('warnings', []))}"
    )
    for error in validation.get("errors", []):
        print(f"ERROR: {error}")
    for warning in validation.get("warnings", []):
        print(f"WARNING: {warning}")
    return 0 if not validation.get("errors") else 1


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
    if arguments.command == "phase4-contract-check":
        return _run_phase4_contract_check()
    if arguments.command == "phase4-validate":
        return _run_phase4_validate(arguments)
    if arguments.command == "phase5-plan-check":
        return _run_phase5_plan_check()
    if arguments.command == "phase5-build":
        return _run_phase5_build(arguments)
    if arguments.command == "phase5-validate":
        return _run_phase5_validate(arguments)
    if arguments.command == "phase6-contract-check":
        return _run_phase6_contract_check()
    if arguments.command == "phase6-plan-check":
        return _run_phase6_plan_check(arguments)
    if arguments.command == "phase6-build":
        return _run_phase6_build(arguments)
    if arguments.command == "phase6-validate":
        return _run_phase6_validate(arguments)
    if arguments.command == "phase7-contract-check":
        return _run_phase7_contract_check()
    if arguments.command == "phase7-plan-check":
        return _run_phase7_plan_check(arguments)
    if arguments.command == "phase7-build":
        return _run_phase7_build(arguments)
    if arguments.command == "phase7-validate":
        return _run_phase7_validate(arguments)
    if arguments.command in {"data-profile", "synthetic-audit", "data-quality-check", "phase3-run"}:
        return _run_phase3(arguments)
    parser.error(f"Unsupported command: {arguments.command}")
    return 2
