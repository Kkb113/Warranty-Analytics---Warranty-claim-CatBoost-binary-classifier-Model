"""Tests for layered typed configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from warranty_analytics_model.config import ConfigurationError, load_settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_minimal_config(root: Path, base: str | None = None) -> None:
    """Create an isolated configuration tree for configuration tests."""

    (root / "configs").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    base_text = base or (
        "project_name: Temporary Project\n"
        "environment: development\n"
        "random_seed: 7\n"
        "log_level: INFO\n"
        "data_dir: data\n"
        "artifact_dir: artifacts\n"
        "report_dir: reports\n"
        "log_dir: logs\n"
    )
    (root / "configs" / "base.yaml").write_text(base_text, encoding="utf-8")
    (root / "configs" / "development.yaml").write_text(
        "environment: development\nlog_level: DEBUG\n",
        encoding="utf-8",
    )
    (root / "configs" / "test.yaml").write_text(
        "environment: test\nlog_level: WARNING\n",
        encoding="utf-8",
    )


def test_base_configuration_loads() -> None:
    """The repository base configuration loads with typed values."""

    settings = load_settings(REPOSITORY_ROOT)

    assert settings.project_name == "Truck-Warranty High-Cost Claim Prediction"
    assert settings.environment == "development"
    assert settings.random_seed == 42


def test_development_configuration_loads() -> None:
    """The development overlay is selected by default."""

    settings = load_settings(REPOSITORY_ROOT)

    assert settings.environment == "development"
    assert settings.log_level == "DEBUG"


def test_test_configuration_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The test overlay changes the environment-specific values."""

    monkeypatch.setenv("WARRANTY_MODEL_ENV", "test")

    settings = load_settings(REPOSITORY_ROOT)

    assert settings.environment == "test"
    assert settings.log_level == "WARNING"
    assert settings.data_dir == "data/test"


def test_environment_variables_override_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operating-system values take precedence over YAML values."""

    monkeypatch.setenv("WARRANTY_MODEL_ENV", "test")
    monkeypatch.setenv("WARRANTY_MODEL_RANDOM_SEED", "123")
    monkeypatch.setenv("WARRANTY_MODEL_LOG_LEVEL", "ERROR")
    monkeypatch.setenv("WARRANTY_MODEL_DATA_DIR", "temporary-data")

    settings = load_settings(REPOSITORY_ROOT)

    assert settings.environment == "test"
    assert settings.random_seed == 123
    assert settings.log_level == "ERROR"
    assert settings.data_dir == "temporary-data"


def test_optional_dotenv_is_loaded(tmp_path: Path) -> None:
    """A local dotenv file is optional and participates below OS overrides."""

    _write_minimal_config(tmp_path)
    (tmp_path / ".env").write_text(
        "WARRANTY_MODEL_ENV=test\nWARRANTY_MODEL_RANDOM_SEED=99\n",
        encoding="utf-8",
    )

    settings = load_settings(tmp_path)

    assert settings.environment == "test"
    assert settings.random_seed == 99


def test_invalid_environment_value_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsupported environments fail before an invalid YAML overlay is selected."""

    monkeypatch.setenv("WARRANTY_MODEL_ENV", "production")

    with pytest.raises(ConfigurationError, match="expected one of"):
        load_settings(REPOSITORY_ROOT)


def test_invalid_typed_setting_fails_without_echoing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid typed values produce an actionable but safe configuration error."""

    monkeypatch.setenv("WARRANTY_MODEL_RANDOM_SEED", "not-a-number")

    with pytest.raises(ConfigurationError, match="Invalid configuration") as error:
        load_settings(REPOSITORY_ROOT)

    assert "not-a-number" not in str(error.value)


def test_secret_values_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Database passwords are accepted from the environment but never displayed."""

    monkeypatch.setenv("WARRANTY_DB_PASSWORD", "fictional-test-secret")
    monkeypatch.setenv("WARRANTY_DB_USERNAME", "fictional-user")

    settings = load_settings(REPOSITORY_ROOT)
    redacted = settings.redacted_dict()

    assert redacted["database"]["password"] == "***REDACTED***"
    assert "fictional-test-secret" not in str(redacted)


def test_secret_keys_are_rejected_from_yaml(tmp_path: Path) -> None:
    """Secret-bearing YAML keys are rejected instead of being silently persisted."""

    _write_minimal_config(
        tmp_path,
        base=(
            "project_name: Temporary Project\n"
            "environment: development\n"
            "password: never-store-this\n"
        ),
    )

    with pytest.raises(ConfigurationError, match="not allowed in YAML"):
        load_settings(tmp_path)


def test_missing_configuration_file_fails_clearly(tmp_path: Path) -> None:
    """A missing base configuration is reported as a configuration error."""

    (tmp_path / "configs").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Required configuration file is missing"):
        load_settings(tmp_path)
