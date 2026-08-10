"""Shared test isolation for the Phase 1 infrastructure suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from warranty_analytics_model import config as configuration_module

_LIVE_DB_TESTS_REQUESTED = os.environ.get("WARRANTY_RUN_DB_TESTS", "false").casefold() == "true"

_CONFIGURATION_ENVIRONMENT_VARIABLES = (
    "WARRANTY_MODEL_ENV",
    "WARRANTY_MODEL_PROJECT_NAME",
    "WARRANTY_MODEL_RANDOM_SEED",
    "WARRANTY_MODEL_LOG_LEVEL",
    "WARRANTY_MODEL_DATA_DIR",
    "WARRANTY_MODEL_ARTIFACT_DIR",
    "WARRANTY_MODEL_REPORT_DIR",
    "WARRANTY_MODEL_LOG_DIR",
    "WARRANTY_DB_SERVER",
    "WARRANTY_DB_PORT",
    "WARRANTY_DB_DATABASE",
    "WARRANTY_DB_AUTH_MODE",
    "WARRANTY_DB_USERNAME",
    "WARRANTY_DB_PASSWORD",
    "WARRANTY_DB_DRIVER",
    "WARRANTY_DB_ENCRYPT",
    "WARRANTY_DB_TRUST_SERVER_CERTIFICATE",
    "WARRANTY_DB_APPLICATION_INTENT",
    "WARRANTY_DB_CONNECTION_TIMEOUT_SECONDS",
    "WARRANTY_DB_QUERY_TIMEOUT_SECONDS",
    "WARRANTY_DB_APPLICATION_NAME",
    "WARRANTY_RUN_DB_TESTS",
)


@pytest.fixture(autouse=True)
def clear_configuration_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent the developer environment from changing test expectations."""

    for variable in _CONFIGURATION_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    if not _LIVE_DB_TESTS_REQUESTED:
        repository_dotenv = (Path(__file__).resolve().parents[1] / ".env").resolve()
        original_loader = configuration_module._load_dotenv

        def isolated_dotenv(path: Path) -> dict[str, str]:
            if path.resolve() == repository_dotenv:
                return {}
            return original_loader(path)

        monkeypatch.setattr(configuration_module, "_load_dotenv", isolated_dotenv)
