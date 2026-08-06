"""Phase 2 database configuration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from warranty_analytics_model.config import ConfigurationError, DatabaseSettings, load_settings
from warranty_analytics_model.database.exceptions import DatabaseConfigurationError

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_database_defaults_preserve_phase1_load_without_server() -> None:
    """Normal configuration remains valid when live database settings are absent."""

    settings = load_settings(REPOSITORY_ROOT)

    assert settings.database.server is None
    assert settings.database.database == "warranty_analytics"
    assert settings.database.port == 1433
    assert settings.database.auth_mode == "trusted"
    assert settings.database.encrypt is True
    assert settings.database.trust_server_certificate is False
    assert settings.database.application_intent == "ReadOnly"
    assert settings.database.connection_timeout_seconds == 15
    assert settings.database.query_timeout_seconds == 30
    assert settings.database.run_db_tests is False


def test_database_environment_overrides_are_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    """All documented database environment variables flow through the existing settings layer."""

    values = {
        "WARRANTY_DB_SERVER": "sql.example.test",
        "WARRANTY_DB_PORT": "1444",
        "WARRANTY_DB_DATABASE": "warranty_analytics",
        "WARRANTY_DB_AUTH_MODE": "sql_password",
        "WARRANTY_DB_USERNAME": "analyst",
        "WARRANTY_DB_PASSWORD": "fictional-secret",
        "WARRANTY_DB_DRIVER": "ODBC Driver 18 for SQL Server",
        "WARRANTY_DB_ENCRYPT": "true",
        "WARRANTY_DB_TRUST_SERVER_CERTIFICATE": "true",
        "WARRANTY_DB_APPLICATION_INTENT": "ReadOnly",
        "WARRANTY_DB_CONNECTION_TIMEOUT_SECONDS": "20",
        "WARRANTY_DB_QUERY_TIMEOUT_SECONDS": "40",
        "WARRANTY_DB_APPLICATION_NAME": "phase2-test",
        "WARRANTY_RUN_DB_TESTS": "true",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    settings = load_settings(REPOSITORY_ROOT)

    assert settings.database.server == "sql.example.test"
    assert settings.database.port == 1444
    assert settings.database.auth_mode == "sql_password"
    assert settings.database.username == "analyst"
    assert settings.database.password == SecretStr("fictional-secret")
    assert settings.database.trust_server_certificate is True
    assert settings.database.query_timeout_seconds == 40
    assert settings.database.application_name == "phase2-test"
    assert settings.database.run_db_tests is True


def test_database_auth_requirements_are_live_only() -> None:
    """Loading remains safe while connection validation enforces auth requirements."""

    trusted = DatabaseSettings(auth_mode="trusted")
    with pytest.raises(DatabaseConfigurationError, match="SERVER"):
        trusted.validate_for_connection()

    sql_password = DatabaseSettings(
        server="sql.example.test",
        auth_mode="sql_password",
    )
    with pytest.raises(DatabaseConfigurationError, match="USERNAME"):
        sql_password.validate_for_connection()
    sql_password = sql_password.model_copy(update={"username": "analyst"})
    with pytest.raises(DatabaseConfigurationError, match="PASSWORD"):
        sql_password.validate_for_connection()


def test_database_safe_display_redacts_secret_and_enforces_readonly() -> None:
    """Safe diagnostics contain connection metadata but never passwords."""

    settings = DatabaseSettings(
        server="sql.example.test",
        auth_mode="sql_password",
        username="analyst",
        password=SecretStr("fictional-secret"),
        trust_server_certificate=False,
    )
    settings.validate_for_connection()
    safe = settings.safe_display()

    assert safe["server"] == "sql.example.test"
    assert safe["application_intent"] == "ReadOnly"
    assert "password" not in safe
    assert "fictional-secret" not in str(safe)


def test_invalid_database_values_are_configuration_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid auth mode and invalid timeout values fail during typed loading."""

    monkeypatch.setenv("WARRANTY_DB_AUTH_MODE", "integrated")
    with pytest.raises(ConfigurationError, match="Invalid configuration"):
        load_settings(REPOSITORY_ROOT)

    monkeypatch.delenv("WARRANTY_DB_AUTH_MODE")
    monkeypatch.setenv("WARRANTY_DB_QUERY_TIMEOUT_SECONDS", "0")
    with pytest.raises(ConfigurationError, match="Invalid configuration"):
        load_settings(REPOSITORY_ROOT)
