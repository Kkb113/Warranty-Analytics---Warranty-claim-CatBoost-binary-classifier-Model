"""Lazy, read-only SQL Server connectivity built from typed settings."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.resources import files
from time import monotonic

from sqlalchemy import Connection, Engine, create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from ..config import DatabaseSettings
from .exceptions import (
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseDriverError,
    UnexpectedDatabaseError,
)
from .models import ConnectivityResult

_SQL_RESOURCES = frozenset(
    {
        "catalog_access.sql",
        "column_metadata.sql",
        "connection_info.sql",
        "foreign_key_metadata.sql",
        "index_metadata.sql",
        "primary_key_metadata.sql",
        "row_count_estimate.sql",
        "schema_names.sql",
        "sequence_names.sql",
        "table_metadata.sql",
        "table_names.sql",
        "view_names.sql",
    }
)


def load_sql_resource(name: str) -> str:
    """Load one reviewed packaged catalog query by allow-listed filename."""

    if name not in _SQL_RESOURCES:
        raise ValueError(f"Unknown catalog query resource: {name}")
    return (
        files("warranty_analytics_model.database").joinpath("sql", name).read_text(encoding="utf-8")
    )


def available_odbc_drivers() -> tuple[str, ...]:
    """Return installed ODBC driver names or an actionable driver error."""

    try:
        import pyodbc  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DatabaseDriverError(
            "pyodbc is not installed; install the project database extra with "
            'python -m pip install -e ".[database]".'
        ) from exc
    try:
        return tuple(str(driver) for driver in pyodbc.drivers())
    except Exception as exc:
        raise DatabaseDriverError("Could not enumerate installed ODBC drivers.") from exc


def validate_driver(settings: DatabaseSettings) -> None:
    """Ensure the configured ODBC driver is installed before opening a connection."""

    drivers = available_odbc_drivers()
    if settings.driver not in drivers:
        raise DatabaseDriverError(
            f"Configured ODBC driver is unavailable: {settings.driver}. "
            "Install Microsoft ODBC Driver 18 for SQL Server."
        )


def build_connection_url(settings: DatabaseSettings) -> URL:
    """Build a structured SQLAlchemy URL without naïve string concatenation."""

    settings.validate_for_connection()
    query = {
        "driver": settings.driver,
        "Encrypt": "yes" if settings.encrypt else "no",
        "TrustServerCertificate": "yes" if settings.trust_server_certificate else "no",
        "ApplicationIntent": settings.application_intent,
        "APP": settings.application_name,
        "Connection Timeout": str(settings.connection_timeout_seconds),
    }
    if settings.auth_mode == "trusted":
        query["Trusted_Connection"] = "yes"
        username = None
        password = None
    else:
        username = settings.username
        password = settings.password.get_secret_value() if settings.password else None
    return URL.create(
        "mssql+pyodbc",
        username=username,
        password=password,
        host=settings.server,
        port=settings.port,
        database=settings.database,
        query=query,
    )


def safe_connection_display(settings: DatabaseSettings) -> dict[str, object]:
    """Return connection diagnostics that never include a password or full URL."""

    return settings.safe_display()


class DatabaseConnection:
    """Own a lazy SQLAlchemy engine and expose only read-only connections."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self.settings = settings
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        """Create the SQLAlchemy engine only when a live operation needs it."""

        if self._engine is None:
            try:
                self._engine = create_engine(
                    build_connection_url(self.settings),
                    connect_args={"timeout": self.settings.query_timeout_seconds},
                    pool_pre_ping=True,
                )
            except DatabaseConfigurationError:
                raise
            except (ImportError, ModuleNotFoundError) as exc:
                raise DatabaseDriverError(
                    "The configured SQL Server dialect is unavailable; install the database extra."
                ) from exc
            except SQLAlchemyError as exc:
                raise DatabaseConnectionError(
                    "Could not construct the SQL Server connection engine."
                ) from exc
        return self._engine

    @contextmanager
    def connect(self) -> Iterator[Connection]:
        """Open one connection and translate failures without exposing secrets."""

        try:
            with self.engine.connect() as connection:
                yield connection
        except DatabaseConfigurationError:
            raise
        except (DatabaseDriverError, DatabaseConnectionError, UnexpectedDatabaseError):
            raise
        except (DBAPIError, SQLAlchemyError) as exc:
            raise DatabaseConnectionError("Could not open the SQL Server connection.") from exc
        except Exception as exc:
            raise UnexpectedDatabaseError(
                "Unexpected failure while using the SQL Server connection."
            ) from exc

    def dispose(self) -> None:
        """Dispose the engine if it was created."""

        if self._engine is not None:
            self._engine.dispose()
            self._engine = None


def check_database_connection(settings: DatabaseSettings) -> ConnectivityResult:
    """Run the bounded read-only connectivity and catalog-access checks."""

    started = monotonic()
    settings.validate_for_connection()
    validate_driver(settings)
    connection = DatabaseConnection(settings)
    try:
        with connection.connect() as db_connection:
            db_connection.execute(text("SELECT 1 AS check_value")).scalar_one()
            info = (
                db_connection.execute(text(load_sql_resource("connection_info.sql")))
                .mappings()
                .one()
            )
            actual_database = str(info["database_name"])
            sql_version = str(info["sql_version"])
            if actual_database.casefold() != settings.database.casefold():
                raise DatabaseConnectionError(
                    "Connected database does not match the configured warranty_analytics database."
                )
            if not info["product_version"] or not sql_version:
                raise DatabaseConnectionError("Connected server did not identify as SQL Server.")
            catalog_row = (
                db_connection.execute(
                    text(load_sql_resource("catalog_access.sql")), {"schema_name": "dbo"}
                )
                .mappings()
                .first()
            )
            if catalog_row is None:
                raise DatabaseConnectionError("SQL Server catalog metadata is not readable.")
            return ConnectivityResult(
                checked_at=datetime.now(UTC),
                server=settings.server,
                port=settings.port,
                expected_database=settings.database,
                actual_database=actual_database,
                sql_version=sql_version,
                catalog_readable=True,
                duration_seconds=monotonic() - started,
            )
    finally:
        connection.dispose()
