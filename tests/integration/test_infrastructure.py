"""Integration tests for the Phase 1 infrastructure components."""

from __future__ import annotations

from warranty_analytics_model.cli import run_doctor
from warranty_analytics_model.config import load_settings
from warranty_analytics_model.paths import resolve_project_paths


def test_doctor_checks_package_configuration_and_logging() -> None:
    """The doctor command joins package, config, path, and logging checks."""

    success, messages = run_doctor()

    assert success is True
    assert all(message.startswith("PASS:") for message in messages)


def test_settings_and_paths_interoperate() -> None:
    """Effective configured directories resolve from the discovered repository."""

    settings = load_settings()
    paths = resolve_project_paths(
        data_dir=settings.data_dir,
        artifact_dir=settings.artifact_dir,
        report_dir=settings.report_dir,
        log_dir=settings.log_dir,
    )

    assert paths.data_dir.is_absolute()
    assert paths.artifact_dir.is_absolute()
    assert paths.report_dir.is_absolute()
    assert paths.log_dir.is_absolute()
