"""Repository-relative path helpers for the project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

_REPOSITORY_MARKERS: Final[tuple[str, ...]] = ("pyproject.toml", "configs")


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolved project paths without creating any directories."""

    root: Path
    config_dir: Path
    data_dir: Path
    artifact_dir: Path
    report_dir: Path
    log_dir: Path

    def as_dict(self) -> dict[str, str]:
        """Return paths in a serializable representation."""

        return {
            "repository_root": str(self.root),
            "configuration_directory": str(self.config_dir),
            "data_directory": str(self.data_dir),
            "artifact_directory": str(self.artifact_dir),
            "report_directory": str(self.report_dir),
            "log_directory": str(self.log_dir),
        }

    def ensure_output_directories(self) -> None:
        """Create configured output directories explicitly when requested."""

        for directory in (self.data_dir, self.artifact_dir, self.report_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)


def _is_repository_root(candidate: Path) -> bool:
    """Return whether a path contains the minimum project markers."""

    return all((candidate / marker).exists() for marker in _REPOSITORY_MARKERS)


def discover_repository_root(start: Path | None = None) -> Path:
    """Find the project root from a starting path or the current working directory."""

    starting_point = (start or Path.cwd()).expanduser().resolve()
    if starting_point.is_file():
        starting_point = starting_point.parent

    candidates = (starting_point, *starting_point.parents)
    for candidate in candidates:
        if _is_repository_root(candidate):
            return candidate

    package_candidate = Path(__file__).resolve().parents[2]
    if _is_repository_root(package_candidate):
        return package_candidate

    raise FileNotFoundError(
        "Could not discover the project root. Run from the repository or pass an explicit root."
    )


def repository_root() -> Path:
    """Return the discovered repository root."""

    return discover_repository_root()


def _resolve_configured_path(root: Path, configured_path: str) -> Path:
    """Resolve a configured path relative to the project root when it is not absolute."""

    path = Path(configured_path).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def resolve_project_paths(
    project_root: Path | None = None,
    *,
    data_dir: str = "data",
    artifact_dir: str = "artifacts",
    report_dir: str = "reports",
    log_dir: str = "logs",
) -> ProjectPaths:
    """Resolve named project paths from a repository or configured project root."""

    root = (project_root or discover_repository_root()).expanduser().resolve()
    return ProjectPaths(
        root=root,
        config_dir=root / "configs",
        data_dir=_resolve_configured_path(root, data_dir),
        artifact_dir=_resolve_configured_path(root, artifact_dir),
        report_dir=_resolve_configured_path(root, report_dir),
        log_dir=_resolve_configured_path(root, log_dir),
    )
