"""Static safety checks for packaged catalog SQL resources."""

from __future__ import annotations

from pathlib import Path

from warranty_analytics_model.database.connection import _SQL_RESOURCES, load_sql_resource

SQL_DIRECTORY = Path(__file__).resolve().parents[3] / "src" / "warranty_analytics_model" / "database" / "sql"


def test_all_allowlisted_sql_resources_are_packaged_and_explicit() -> None:
    """Every query resource is readable, explicit-column, and read-only."""

    forbidden = ("insert ", "update ", "delete ", "merge ", "truncate ", "create ", "alter ", "drop ", "exec ")
    assert _SQL_RESOURCES
    for name in _SQL_RESOURCES:
        query = load_sql_resource(name)
        lowered = query.casefold()
        assert "select *" not in lowered
        assert not any(token in lowered for token in forbidden)
        assert "from sys." in lowered or "serverproperty" in lowered or "@@version" in lowered
        assert "warranty_analytics" not in lowered
        assert "password" not in lowered


def test_repository_sql_readme_documents_packaging_deviation() -> None:
    """The required repository-level SQL directory explains the package-resource layout."""

    readme = SQL_DIRECTORY.parents[3] / "sql" / "source_validation" / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert "packaged" in text
    assert "sys.partitions" in text
    assert "excluded ML" in text
