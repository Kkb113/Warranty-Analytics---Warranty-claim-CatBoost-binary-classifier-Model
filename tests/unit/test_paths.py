"""Tests for repository-relative path handling."""

from __future__ import annotations

from pathlib import Path

from warranty_analytics_model.paths import (
    discover_repository_root,
    repository_root,
    resolve_project_paths,
)


def test_paths_resolve_without_creating_directories(tmp_path: Path) -> None:
    """Resolution is pure and uses the configured repository root."""

    paths = resolve_project_paths(
        tmp_path,
        data_dir="local-data",
        artifact_dir="local-artifacts",
        report_dir="local-reports",
        log_dir="local-logs",
    )

    assert paths.root == tmp_path.resolve()
    assert paths.data_dir == (tmp_path / "local-data").resolve()
    assert not paths.data_dir.exists()


def test_absolute_path_override_is_supported(tmp_path: Path) -> None:
    """An explicitly configured absolute path is not joined twice."""

    absolute_data = (tmp_path / "external-data").resolve()

    paths = resolve_project_paths(tmp_path, data_dir=str(absolute_data))

    assert paths.data_dir == absolute_data


def test_output_directories_are_created_only_explicitly(tmp_path: Path) -> None:
    """The explicit creation method creates all generated-output directories."""

    paths = resolve_project_paths(tmp_path)
    paths.ensure_output_directories()

    assert paths.data_dir.is_dir()
    assert paths.artifact_dir.is_dir()
    assert paths.report_dir.is_dir()
    assert paths.log_dir.is_dir()


def test_repository_root_is_found_from_nested_directory(tmp_path: Path) -> None:
    """Root discovery walks upward from a nested path."""

    (tmp_path / "configs").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    nested = tmp_path / "nested" / "child"
    nested.mkdir(parents=True)

    assert discover_repository_root(nested) == tmp_path.resolve()


def test_repository_root_function_finds_this_checkout() -> None:
    """The convenience function finds the current checkout."""

    root = repository_root()

    assert (root / "pyproject.toml").is_file()
    assert (root / "configs").is_dir()
