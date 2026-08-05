"""Small standard-library logging setup for project modules and the CLI."""

from __future__ import annotations

import logging
from pathlib import Path

_LOGGER_NAME = "warranty_analytics_model"
_HANDLER_MARKER = "_warranty_analytics_handler"
_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a named project logger."""

    return logging.getLogger(name or _LOGGER_NAME)


def _marked_handler(logger: logging.Logger, kind: str) -> logging.Handler | None:
    """Find an existing infrastructure handler without assuming its concrete type."""

    for handler in logger.handlers:
        if getattr(handler, _HANDLER_MARKER, None) == kind:
            return handler
    return None


def _mark_handler(handler: logging.Handler, kind: str) -> None:
    """Mark a handler so repeated setup can reuse it."""

    setattr(handler, _HANDLER_MARKER, kind)


def configure_logging(
    level: str = "INFO",
    *,
    log_dir: Path | None = None,
    enable_file: bool = False,
    logger_name: str = _LOGGER_NAME,
) -> logging.Logger:
    """Configure console logging once, with optional explicit file logging."""

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {level!r}")

    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT)

    console_handler = _marked_handler(logger, "console")
    if console_handler is None:
        console_handler = logging.StreamHandler()
        _mark_handler(console_handler, "console")
        logger.addHandler(console_handler)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    file_handler = _marked_handler(logger, "file")
    if enable_file:
        if log_dir is None:
            raise ValueError("log_dir is required when file logging is enabled.")
        log_dir.mkdir(parents=True, exist_ok=True)
        if file_handler is None:
            file_handler = logging.FileHandler(log_dir / "warranty_model.log", encoding="utf-8")
            _mark_handler(file_handler, "file")
            logger.addHandler(file_handler)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
    elif file_handler is not None:
        logger.removeHandler(file_handler)
        file_handler.close()

    return logger
