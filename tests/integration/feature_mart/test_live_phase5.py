"""Opt-in live Phase 5 build coverage; ordinary CI remains database-independent."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from warranty_analytics_model.config import ConfigurationError, load_settings
from warranty_analytics_model.database.exceptions import DatabaseConfigurationError
from warranty_analytics_model.feature_mart.runner import run_live_phase5, validate_existing_mart

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
# tests/conftest.py deliberately clears configuration variables during fixtures. Capture the
# explicit operator opt-in before fixtures run while still requiring it for this test module.
_LIVE_REQUESTED = os.environ.get("WARRANTY_RUN_DB_TESTS", "false").casefold() == "true"
pytestmark = pytest.mark.database


def test_live_phase5_build_is_read_only_and_validated(tmp_path: Path) -> None:
    """Build and validate a temporary local mart without writing to SQL Server."""

    if not _LIVE_REQUESTED:
        pytest.skip("Set WARRANTY_RUN_DB_TESTS=true for an explicit local live run.")
    try:
        settings = load_settings(REPOSITORY_ROOT)
        settings.database.validate_for_connection()
    except (ConfigurationError, DatabaseConfigurationError) as exc:
        pytest.skip(f"Live database configuration is unavailable: {exc}")

    result = run_live_phase5(
        settings,
        output_dir=tmp_path / "artifacts" / "feature_mart",
        report_dir=tmp_path / "reports" / "phase5_feature_mart",
    )
    assert result.status in {"PASS", "PASS WITH WARNINGS"}
    assert not result.errors
    mart_dir = Path(result.run_directory)
    assert (mart_dir / "claim_snapshot.parquet").is_file()
    assert result.manifest["source_database"] == settings.database.database

    offline_validation = validate_existing_mart(mart_dir)
    assert offline_validation["status"] in {"PASS", "PASS WITH WARNINGS"}
    assert not offline_validation["errors"]
