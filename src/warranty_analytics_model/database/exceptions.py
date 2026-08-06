"""Public exceptions for safe Phase 2 database operations."""

from __future__ import annotations


class DatabaseConfigurationError(RuntimeError):
    """Raised when live database settings are missing or unsafe."""


class DatabaseDriverError(RuntimeError):
    """Raised when the configured ODBC driver cannot be used."""


class DatabaseConnectionError(RuntimeError):
    """Raised when a read-only database connection cannot be opened."""


class UnexpectedDatabaseError(RuntimeError):
    """Raised for unexpected database failures without exposing connection secrets."""


class SchemaContractError(RuntimeError):
    """Raised when the version-controlled schema contract is invalid or unreadable."""


class SchemaValidationError(RuntimeError):
    """Raised when schema validation cannot produce a trustworthy result."""
