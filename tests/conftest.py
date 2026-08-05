"""Shared test isolation for the Phase 1 infrastructure suite."""

from __future__ import annotations

import pytest

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
    "WARRANTY_DB_DATABASE",
    "WARRANTY_DB_USERNAME",
    "WARRANTY_DB_PASSWORD",
    "WARRANTY_DB_DRIVER",
)


@pytest.fixture(autouse=True)
def clear_configuration_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent the developer environment from changing test expectations."""

    for variable in _CONFIGURATION_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
