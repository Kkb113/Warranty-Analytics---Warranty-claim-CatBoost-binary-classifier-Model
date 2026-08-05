"""Tests for the infrastructure-only CLI."""

from __future__ import annotations

import pytest

from warranty_analytics_model.cli import main


def test_cli_version_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    """The version command returns the package version."""

    assert main(["version"]) == 0

    assert capsys.readouterr().out.strip() == "0.1.0"


def test_cli_help_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    """The module help path is available through the parser."""

    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    assert "warranty-model" in capsys.readouterr().out


def test_cli_show_config_redacts_secrets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """show-config does not expose an environment-provided password."""

    monkeypatch.setenv("WARRANTY_DB_PASSWORD", "fictional-cli-secret")

    assert main(["show-config"]) == 0
    output = capsys.readouterr().out

    assert "***REDACTED***" in output
    assert "fictional-cli-secret" not in output


def test_cli_doctor_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    """doctor validates the local Phase 1 setup."""

    assert main(["doctor"]) == 0

    output = capsys.readouterr().out
    assert "PASS: package import" in output
    assert "PASS: logging initialized" in output


def test_cli_doctor_returns_nonzero_for_invalid_environment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """doctor reports invalid configuration through a non-zero return code."""

    monkeypatch.setenv("WARRANTY_MODEL_ENV", "production")

    assert main(["doctor"]) == 1
    assert "FAIL: Invalid WARRANTY_MODEL_ENV" in capsys.readouterr().out


def test_cli_show_config_returns_nonzero_for_invalid_environment(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """show-config reports configuration errors on stderr."""

    monkeypatch.setenv("WARRANTY_MODEL_ENV", "production")

    assert main(["show-config"]) == 2
    assert "Configuration error" in capsys.readouterr().err


def test_cli_without_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Invoking the CLI without a subcommand is informative and harmless."""

    assert main([]) == 0
    assert "usage:" in capsys.readouterr().out
