"""Database-specific accessors over the existing project settings layer."""

from __future__ import annotations

from pathlib import Path

from ..config import DatabaseSettings, Settings, load_settings


def load_database_settings(project_root: Path | None = None) -> DatabaseSettings:
    """Return typed database settings without opening a connection."""

    return load_settings(project_root).database


def database_settings_from(settings: Settings) -> DatabaseSettings:
    """Extract the database section from an already loaded project settings object."""

    return settings.database


__all__ = ["DatabaseSettings", "database_settings_from", "load_database_settings"]
