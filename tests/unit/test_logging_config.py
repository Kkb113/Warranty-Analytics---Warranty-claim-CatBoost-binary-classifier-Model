"""Tests for standard-library logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from warranty_analytics_model.logging_config import configure_logging, get_logger


def test_logging_initialization_does_not_duplicate_handlers(tmp_path: Path) -> None:
    """Repeated setup reuses console handlers and optional file handlers."""

    logger_name = "warranty_analytics_model.tests.logging"
    logger = configure_logging("INFO", logger_name=logger_name)
    first_handler_count = len(logger.handlers)

    same_logger = configure_logging("DEBUG", logger_name=logger_name)

    assert same_logger is logger
    assert len(logger.handlers) == first_handler_count == 1
    assert logger.level == logging.DEBUG

    configure_logging("INFO", logger_name=logger_name, log_dir=tmp_path, enable_file=True)
    logger.warning("fictional infrastructure message")
    log_file = tmp_path / "warranty_model.log"
    assert log_file.is_file()
    assert "fictional infrastructure message" in log_file.read_text(encoding="utf-8")

    configure_logging("INFO", logger_name=logger_name, enable_file=False)
    assert len(logger.handlers) == 1


def test_logging_rejects_invalid_level() -> None:
    """Invalid log levels fail instead of silently choosing a default."""

    with pytest.raises(ValueError, match="Invalid logging level"):
        configure_logging("NOT_A_LEVEL", logger_name="warranty_analytics_model.tests.invalid")


def test_file_logging_requires_a_directory() -> None:
    """File logging requires an explicit destination."""

    with pytest.raises(ValueError, match="log_dir is required"):
        configure_logging(
            "INFO",
            logger_name="warranty_analytics_model.tests.no_directory",
            enable_file=True,
        )


def test_get_logger_returns_named_logger() -> None:
    """Modules can obtain stable named loggers."""

    assert get_logger("warranty_analytics_model.tests.named").name == (
        "warranty_analytics_model.tests.named"
    )
