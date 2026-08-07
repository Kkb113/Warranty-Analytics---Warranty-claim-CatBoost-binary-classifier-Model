"""Opt-in live SQL Server checks; ordinary CI always skips these tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from warranty_analytics_model.config import ConfigurationError, load_settings
from warranty_analytics_model.database.connection import check_database_connection
from warranty_analytics_model.database.exceptions import DatabaseConfigurationError
from warranty_analytics_model.database.metadata import collect_schema_metadata
from warranty_analytics_model.database.schema_contract import load_schema_contract

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.database


def _live_settings():
    if os.environ.get("WARRANTY_RUN_DB_TESTS", "false").casefold() != "true":
        pytest.skip("Set WARRANTY_RUN_DB_TESTS=true to enable live SQL Server checks.")
    try:
        settings = load_settings(REPOSITORY_ROOT)
        settings.database.validate_for_connection()
    except (ConfigurationError, DatabaseConfigurationError) as exc:
        pytest.skip(f"Live database configuration is unavailable: {exc}")
    return settings


def test_live_db_check_is_read_only() -> None:
    """Live connectivity checks do not read business rows."""

    settings = _live_settings()
    result = check_database_connection(settings.database)
    assert result.actual_database == "warranty_analytics"
    assert result.catalog_readable is True


def test_live_schema_matches_approved_scope() -> None:
    """Live metadata validation inspects only the approved catalog scope."""

    settings = _live_settings()
    contract = load_schema_contract(REPOSITORY_ROOT)[0]
    live = collect_schema_metadata(settings.database, contract)
    assert live.database_name == contract.database.name
    assert len(live.tables) <= len(contract.tables)
    assert not set(live.excluded_objects) - set(contract.excluded_tables)
