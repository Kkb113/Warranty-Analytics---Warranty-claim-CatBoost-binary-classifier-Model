"""Typed, layered configuration for the project infrastructure."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal, cast

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from .paths import discover_repository_root

EnvironmentName = Literal["development", "test"]

SUPPORTED_ENVIRONMENTS: Final[frozenset[str]] = frozenset({"development", "test"})

_DEFAULTS: Final[dict[str, Any]] = {
    "project_name": "Truck-Warranty High-Cost Claim Prediction",
    "environment": "development",
    "random_seed": 42,
    "log_level": "INFO",
    "data_dir": "data",
    "artifact_dir": "artifacts",
    "report_dir": "reports",
    "log_dir": "logs",
}

_MODEL_ENV_FIELDS: Final[dict[str, str]] = {
    "WARRANTY_MODEL_ENV": "environment",
    "WARRANTY_MODEL_PROJECT_NAME": "project_name",
    "WARRANTY_MODEL_RANDOM_SEED": "random_seed",
    "WARRANTY_MODEL_LOG_LEVEL": "log_level",
    "WARRANTY_MODEL_DATA_DIR": "data_dir",
    "WARRANTY_MODEL_ARTIFACT_DIR": "artifact_dir",
    "WARRANTY_MODEL_REPORT_DIR": "report_dir",
    "WARRANTY_MODEL_LOG_DIR": "log_dir",
}

_DATABASE_ENV_FIELDS: Final[dict[str, str]] = {
    "WARRANTY_DB_SERVER": "server",
    "WARRANTY_DB_PORT": "port",
    "WARRANTY_DB_DATABASE": "database",
    "WARRANTY_DB_AUTH_MODE": "auth_mode",
    "WARRANTY_DB_USERNAME": "username",
    "WARRANTY_DB_PASSWORD": "password",
    "WARRANTY_DB_DRIVER": "driver",
    "WARRANTY_DB_ENCRYPT": "encrypt",
    "WARRANTY_DB_TRUST_SERVER_CERTIFICATE": "trust_server_certificate",
    "WARRANTY_DB_APPLICATION_INTENT": "application_intent",
    "WARRANTY_DB_CONNECTION_TIMEOUT_SECONDS": "connection_timeout_seconds",
    "WARRANTY_DB_QUERY_TIMEOUT_SECONDS": "query_timeout_seconds",
    "WARRANTY_DB_APPLICATION_NAME": "application_name",
    "WARRANTY_RUN_DB_TESTS": "run_db_tests",
}

_SECRET_KEY_PARTS: Final[tuple[str, ...]] = (
    "password",
    "secret",
    "token",
    "credential",
    "api_key",
)


class ConfigurationError(RuntimeError):
    """Raised when project configuration cannot be loaded safely."""


class DatabaseSettings(BaseModel):
    """Typed SQL Server settings; connection validation is explicit and opt-in."""

    model_config = ConfigDict(extra="forbid")

    server: str | None = None
    port: int = Field(default=1433, ge=1, le=65535)
    database: str = "warranty_analytics"
    auth_mode: Literal["trusted", "sql_password"] = "trusted"
    username: str | None = None
    password: SecretStr | None = None
    driver: str = "ODBC Driver 18 for SQL Server"
    encrypt: bool = True
    trust_server_certificate: bool = False
    application_intent: Literal["ReadOnly"] = "ReadOnly"
    connection_timeout_seconds: int = Field(default=15, ge=1, le=300)
    query_timeout_seconds: int = Field(default=30, ge=1, le=3600)
    application_name: str = "warranty_analytics_model"
    run_db_tests: bool = False

    def validate_for_connection(self) -> None:
        """Validate values that are required only by live database commands."""

        from .database.exceptions import DatabaseConfigurationError

        if not self.server or not self.server.strip():
            raise DatabaseConfigurationError(
                "WARRANTY_DB_SERVER is required for a live database command."
            )
        if not self.database.strip():
            raise DatabaseConfigurationError("The database name must not be empty.")
        if not self.driver.strip():
            raise DatabaseConfigurationError("The ODBC driver name must not be empty.")
        if self.auth_mode == "sql_password":
            if not self.username or not self.username.strip():
                raise DatabaseConfigurationError(
                    "WARRANTY_DB_USERNAME is required when auth_mode is sql_password."
                )
            if self.password is None or not self.password.get_secret_value():
                raise DatabaseConfigurationError(
                    "WARRANTY_DB_PASSWORD is required when auth_mode is sql_password."
                )

    def safe_display(self) -> dict[str, Any]:
        """Return non-secret connection diagnostics suitable for logs and CLI output."""

        return {
            "server": self.server,
            "port": self.port,
            "database": self.database,
            "driver": self.driver,
            "auth_mode": self.auth_mode,
            "encrypt": self.encrypt,
            "trust_server_certificate": self.trust_server_certificate,
            "application_intent": self.application_intent,
            "connection_timeout_seconds": self.connection_timeout_seconds,
            "query_timeout_seconds": self.query_timeout_seconds,
            "application_name": self.application_name,
        }


class Settings(BaseSettings):
    """Validated effective settings after all configuration layers are applied."""

    model_config = SettingsConfigDict(extra="forbid", env_prefix="WARRANTY_MODEL_")

    project_name: str = "Truck-Warranty High-Cost Claim Prediction"
    environment: EnvironmentName = "development"
    random_seed: int = 42
    log_level: str = "INFO"
    data_dir: str = "data"
    artifact_dir: str = "artifacts"
    report_dir: str = "reports"
    log_dir: str = "logs"
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)

    def redacted_dict(self) -> dict[str, Any]:
        """Return a serializable configuration with sensitive values redacted."""

        raw = self.model_dump(mode="python")
        return cast(dict[str, Any], _redact_value(raw))


def _is_secret_key(key: object) -> bool:
    """Return whether a mapping key indicates a secret-bearing setting."""

    normalized = str(key).lower()
    return any(part in normalized for part in _SECRET_KEY_PARTS)


def _redact_value(value: Any, key: str | None = None) -> Any:
    """Recursively redact secret values for display."""

    if key is not None and _is_secret_key(key):
        return "***REDACTED***"
    if isinstance(value, SecretStr):
        return "***REDACTED***"
    if isinstance(value, Mapping):
        return {
            str(item_key): _redact_value(item_value, str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_value(item) for item in value]
    return value


def _assert_no_secret_keys(payload: Mapping[str, Any], source: Path) -> None:
    """Reject secret-bearing keys in YAML configuration files."""

    for key, value in payload.items():
        if _is_secret_key(key):
            raise ConfigurationError(
                f"Secret-bearing setting {key!s} is not allowed in YAML file {source}."
            )
        if isinstance(value, Mapping):
            _assert_no_secret_keys(value, source)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load one non-secret YAML mapping."""

    if not path.is_file():
        raise ConfigurationError(f"Required configuration file is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except OSError as exc:
        raise ConfigurationError(f"Could not read configuration file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in configuration file {path}: {exc}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigurationError(f"Configuration file {path} must contain a YAML mapping.")
    payload = {str(key): value for key, value in loaded.items()}
    _assert_no_secret_keys(payload, path)
    return payload


def _merge_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested mappings while allowing later values to override earlier values."""

    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[str(key)] = _merge_mappings(current, value)
        else:
            merged[str(key)] = value
    return merged


def _load_dotenv(path: Path) -> dict[str, str]:
    """Load optional local dotenv values without requiring the file."""

    if not path.is_file():
        return {}
    values = dotenv_values(path)
    return {
        str(key): value for key, value in values.items() if key is not None and value is not None
    }


def _select_environment(base: Mapping[str, Any], dotenv: Mapping[str, str]) -> EnvironmentName:
    """Select and validate the environment before loading its YAML overrides."""

    requested = (
        os.environ.get("WARRANTY_MODEL_ENV")
        or dotenv.get("WARRANTY_MODEL_ENV")
        or str(base.get("environment", _DEFAULTS["environment"]))
    )
    if requested not in SUPPORTED_ENVIRONMENTS:
        supported = ", ".join(sorted(SUPPORTED_ENVIRONMENTS))
        raise ConfigurationError(
            f"Invalid WARRANTY_MODEL_ENV value {requested!r}; expected one of: {supported}."
        )
    return cast(EnvironmentName, requested)


def _non_empty_overrides(values: Mapping[str, str], field_map: Mapping[str, str]) -> dict[str, str]:
    """Translate non-empty environment values to typed-setting field names."""

    return {
        field_name: values[environment_name]
        for environment_name, field_name in field_map.items()
        if values.get(environment_name)
    }


def _apply_environment_overrides(
    payload: dict[str, Any],
    dotenv: Mapping[str, str],
    operating_system: Mapping[str, str],
) -> None:
    """Apply dotenv values followed by operating-system values."""

    dotenv_model = _non_empty_overrides(dotenv, _MODEL_ENV_FIELDS)
    operating_system_model = _non_empty_overrides(operating_system, _MODEL_ENV_FIELDS)
    payload.update(dotenv_model)
    payload.update(operating_system_model)

    configured_database = payload.get("database", {})
    if not isinstance(configured_database, Mapping):
        raise ConfigurationError("The database configuration must be a mapping.")
    database = dict(configured_database)
    dotenv_database = _non_empty_overrides(dotenv, _DATABASE_ENV_FIELDS)
    operating_system_database = _non_empty_overrides(operating_system, _DATABASE_ENV_FIELDS)
    database.update(dotenv_database)
    database.update(operating_system_database)
    payload["database"] = database


def _validation_error_message(error: ValidationError) -> str:
    """Format validation errors without echoing potentially sensitive input values."""

    details = []
    for item in error.errors():
        location = ".".join(str(part) for part in item.get("loc", ())) or "settings"
        details.append(f"{location}: {item.get('msg', 'invalid value')}")
    return "; ".join(details)


def load_settings(project_root: Path | None = None) -> Settings:
    """Load typed settings with the documented precedence order."""

    root = discover_repository_root(project_root)
    config_dir = root / "configs"
    base = _merge_mappings(_DEFAULTS, _load_yaml(config_dir / "base.yaml"))
    dotenv = _load_dotenv(root / ".env")
    environment = _select_environment(base, dotenv)
    environment_config = _load_yaml(config_dir / f"{environment}.yaml")

    payload = _merge_mappings(base, environment_config)
    payload["environment"] = environment
    _apply_environment_overrides(payload, dotenv, os.environ)

    try:
        return Settings.model_validate(payload)
    except ValidationError as exc:
        raise ConfigurationError(
            f"Invalid configuration: {_validation_error_message(exc)}"
        ) from exc
