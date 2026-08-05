"""Infrastructure-only command-line interface for the project."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import ConfigurationError, load_settings
from .logging_config import configure_logging
from .paths import discover_repository_root, resolve_project_paths


def build_parser() -> argparse.ArgumentParser:
    """Build the supported infrastructure CLI parser."""

    parser = argparse.ArgumentParser(
        prog="warranty-model",
        description="Phase 1 infrastructure diagnostics for the warranty analytics project.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="Validate Phase 1 infrastructure only.")
    subparsers.add_parser("show-config", help="Show effective non-secret configuration.")
    subparsers.add_parser("version", help="Show the package version.")
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
    parser.error(f"Unsupported command: {arguments.command}")
    return 2
