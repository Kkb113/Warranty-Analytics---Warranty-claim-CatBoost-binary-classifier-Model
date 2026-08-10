"""Technical Phase 5 configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..paths import discover_repository_root
from .models import FeatureMartError, FeatureMartSettings


def load_feature_mart_settings(
    project_root: Path | None = None,
    *,
    path: Path | None = None,
) -> FeatureMartSettings:
    """Load non-secret mart settings without duplicating business policy."""

    root = discover_repository_root(project_root)
    config_path = path or root / "configs" / "feature_mart.yaml"
    if not config_path.is_file():
        raise FeatureMartError(f"Feature-mart configuration is missing: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FeatureMartError(f"Could not read feature-mart configuration: {config_path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("feature_mart"), dict):
        raise FeatureMartError("configs/feature_mart.yaml must contain a feature_mart mapping.")
    try:
        return FeatureMartSettings.model_validate(dict(payload["feature_mart"]))
    except Exception as exc:
        raise FeatureMartError("Invalid feature-mart technical configuration.") from exc


def resolve_mart_output_root(
    root: Path, settings: FeatureMartSettings, override: Path | None
) -> Path:
    """Resolve an artifact root from configuration or an explicit CLI override."""

    configured = override if override is not None else Path(settings.output_directory)
    return configured.resolve() if configured.is_absolute() else (root / configured).resolve()


def resolve_mart_report_root(
    root: Path, settings: FeatureMartSettings, override: Path | None
) -> Path:
    """Resolve a report root from configuration or an explicit CLI override."""

    configured = override if override is not None else Path(settings.report_directory)
    return configured.resolve() if configured.is_absolute() else (root / configured).resolve()


def technical_settings_dict(settings: FeatureMartSettings) -> dict[str, Any]:
    """Return stable, non-secret settings for manifests."""

    return settings.model_dump(mode="json")
