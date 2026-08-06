"""Connection construction and safety tests without live SQL Server access."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

from warranty_analytics_model.config import DatabaseSettings
from warranty_analytics_model.database import connection
from warranty_analytics_model.database.exceptions import (
    DatabaseConnectionError,
    DatabaseDriverError,
)


def test_trusted_url_uses_readonly_encryption_and_driver_query() -> None:
    """Trusted Windows authentication is represented with structured URL query fields."""

    settings = DatabaseSettings(
        server="sql.example.test",
        port=1444,
        driver="ODBC Driver 18 for SQL Server",
        trust_server_certificate=False,
    )
    url = connection.build_connection_url(settings)

    assert url.drivername == "mssql+pyodbc"
    assert url.host == "sql.example.test"
    assert url.port == 1444
    assert url.database == "warranty_analytics"
    assert url.username is None
    assert url.password is None
    assert url.query["driver"] == "ODBC Driver 18 for SQL Server"
    assert url.query["Trusted_Connection"] == "yes"
    assert url.query["ApplicationIntent"] == "ReadOnly"
    assert url.query["Encrypt"] == "yes"
    assert url.query["TrustServerCertificate"] == "no"
    assert "password" not in url.render_as_string(hide_password=True).casefold()


def test_sql_password_url_handles_special_password_without_logging_it() -> None:
    """SQL authentication preserves special characters through URL encoding."""

    secret = "p@ss:word/with spaces?"
    settings = DatabaseSettings(
        server="sql.example.test",
        auth_mode="sql_password",
        username="user.name",
        password=secret,
    )
    url = connection.build_connection_url(settings)

    assert url.username == "user.name"
    assert url.password == secret
    safe_url = url.render_as_string(hide_password=True)
    assert secret not in safe_url
    assert "ODBC+Driver+18+for+SQL+Server" in safe_url


def test_engine_is_lazy_and_disposable() -> None:
    """Constructing the wrapper does not open a database connection."""

    database = connection.DatabaseConnection(DatabaseSettings(server="sql.example.test"))
    assert database._engine is None
    database.dispose()
    engine = database.engine
    assert engine is database.engine
    database.dispose()
    assert database._engine is None


def test_driver_validation_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing or unavailable driver names fail before connection attempts."""

    settings = DatabaseSettings(server="sql.example.test")
    monkeypatch.setattr(connection, "available_odbc_drivers", lambda: ("Other Driver",))
    with pytest.raises(DatabaseDriverError, match="unavailable"):
        connection.validate_driver(settings)

    monkeypatch.setattr(
        connection,
        "available_odbc_drivers",
        lambda: ("ODBC Driver 18 for SQL Server",),
    )
    connection.validate_driver(settings)


class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]], scalar: int | None = None) -> None:
        self.rows = rows
        self.scalar = scalar

    def scalar_one(self) -> int:
        assert self.scalar is not None
        return self.scalar

    def mappings(self) -> _FakeResult:
        return self

    def one(self) -> dict[str, Any]:
        return self.rows[0]

    def first(self) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None


class _FakeConnection:
    def execute(self, statement: object, parameters: object | None = None) -> _FakeResult:
        query = str(statement)
        if "SELECT 1 AS check_value" in query:
            return _FakeResult([], scalar=1)
        if "connection_info" in query:
            return _FakeResult([])
        if "SERVERPROPERTY" in query:
            return _FakeResult(
                [
                    {
                        "database_name": "warranty_analytics",
                        "server_name": "sql.example.test",
                        "product_version": "16.0",
                        "sql_version": "Microsoft SQL Server 2022",
                    }
                ]
            )
        return _FakeResult([{"schema_name": "dbo"}])


class _FakeDatabaseConnection:
    def __init__(self, settings: DatabaseSettings) -> None:
        self.settings = settings
        self.closed = False

    @contextmanager
    def connect(self):
        yield _FakeConnection()

    def dispose(self) -> None:
        self.closed = True


def test_database_check_uses_bounded_queries_and_disposes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The connectivity command succeeds through mocked safe probes only."""

    monkeypatch.setattr(connection, "validate_driver", lambda settings: None)
    monkeypatch.setattr(connection, "DatabaseConnection", _FakeDatabaseConnection)
    result = connection.check_database_connection(DatabaseSettings(server="sql.example.test"))

    assert result.actual_database == "warranty_analytics"
    assert result.catalog_readable is True
    assert result.duration_seconds >= 0


def test_database_check_rejects_wrong_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrong live database cannot pass the expected-database guard."""

    class WrongDatabaseConnection(_FakeDatabaseConnection):
        @contextmanager
        def connect(self):
            class WrongConnection(_FakeConnection):
                def execute(self, statement: object, parameters: object | None = None) -> _FakeResult:
                    result = super().execute(statement, parameters)
                    if "SERVERPROPERTY" in str(statement):
                        result.rows[0]["database_name"] = "other_database"
                    return result

            yield WrongConnection()

    monkeypatch.setattr(connection, "validate_driver", lambda settings: None)
    monkeypatch.setattr(connection, "DatabaseConnection", WrongDatabaseConnection)
    with pytest.raises(DatabaseConnectionError, match="does not match"):
        connection.check_database_connection(DatabaseSettings(server="sql.example.test"))


def test_sql_resource_loader_allow_lists_queries() -> None:
    """No arbitrary SQL resource name can be requested through the loader."""

    assert "sys.columns" in connection.load_sql_resource("column_metadata.sql")
    with pytest.raises(ValueError, match="Unknown catalog query"):
        connection.load_sql_resource("business_rows.sql")
