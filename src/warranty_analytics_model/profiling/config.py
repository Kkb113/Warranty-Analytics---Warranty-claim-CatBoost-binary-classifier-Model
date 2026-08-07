"""Typed Phase 3 profiling configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..config import ConfigurationError
from ..paths import discover_repository_root


class ProfilingSettings(BaseModel):
    """Settings for diagnostics, not acceptance thresholds or model behavior."""

    model_config = ConfigDict(extra="forbid")

    output_directory: str = "reports/data_profiling"
    chunk_size: int = Field(default=10_000, ge=100, le=1_000_000)
    top_categories: int = Field(default=20, ge=1, le=100)
    percentiles: list[float] = Field(
        default_factory=lambda: [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
        min_length=1,
    )
    rare_category_thresholds: list[int] = Field(
        default_factory=lambda: [1, 5, 10, 20], min_length=1
    )
    enable_text_audit: bool = True
    enable_identifier_audit: bool = True
    enable_temporal_audit: bool = True
    enable_target_audit: bool = True
    generate_charts: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> ProfilingSettings:
        """Load the documented non-secret YAML configuration."""

        if not path.is_file():
            raise ConfigurationError(f"Required profiling configuration is missing: {path}")
        try:
            payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigurationError(f"Could not read profiling configuration: {path}") from exc
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ConfigurationError(f"Profiling configuration must be a mapping: {path}")
        section = payload.get("profiling", payload)
        if not isinstance(section, dict):
            raise ConfigurationError("The profiling configuration section must be a mapping.")
        try:
            return cls.model_validate(section)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(part) for part in item.get('loc', ())) or 'profiling'}: "
                f"{item.get('msg', 'invalid value')}"
                for item in exc.errors()
            )
            raise ConfigurationError(f"Invalid profiling configuration: {details}") from exc

    def resolved_output_directory(self, project_root: Path | None = None) -> Path:
        """Resolve the configured report root without creating it."""

        root = discover_repository_root(project_root)
        configured = Path(self.output_directory).expanduser()
        return configured.resolve() if configured.is_absolute() else (root / configured).resolve()


def load_profiling_settings(
    project_root: Path | None = None,
    path: Path | None = None,
) -> ProfilingSettings:
    """Load ``configs/profiling.yaml`` using the project-root convention."""

    root = discover_repository_root(project_root)
    return ProfilingSettings.from_yaml(path or root / "configs" / "profiling.yaml")
