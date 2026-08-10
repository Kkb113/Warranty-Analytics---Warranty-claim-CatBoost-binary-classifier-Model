"""Configuration loading and validation for Phase 6 split design."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ..paths import discover_repository_root
from .models import SplitError, SplitSettings

SPLIT_CONFIG_NAME = "splits.yaml"


def load_split_settings(
    project_root: Path | None = None,
    *,
    path: Path | None = None,
) -> SplitSettings:
    """Load the non-secret Phase 6 technical configuration."""

    root = discover_repository_root(project_root)
    config_path = path or root / "configs" / SPLIT_CONFIG_NAME
    if not config_path.is_file():
        raise SplitError(f"Phase 6 split configuration is missing: {config_path}")
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SplitError(f"Could not read Phase 6 split configuration: {config_path}") from exc
    if not isinstance(loaded, Mapping) or not isinstance(loaded.get("split"), Mapping):
        raise SplitError("configs/splits.yaml must contain a top-level split mapping.")
    try:
        settings = SplitSettings.model_validate(dict(loaded["split"]))
    except ValidationError as exc:
        raise SplitError(f"Invalid Phase 6 split configuration: {exc}") from exc
    errors = validate_split_settings(settings)
    if errors:
        raise SplitError("; ".join(errors))
    return settings


def validate_split_settings(settings: SplitSettings) -> list[str]:
    """Return fail-closed configuration diagnostics."""

    errors: list[str] = []
    fractions = settings.requested_fractions
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        errors.append("Train, validation, and test fractions must sum to 1.0.")
    if settings.strategy != "chronological":
        errors.append("Phase 6 primary split strategy must be chronological.")
    if not settings.preserve_same_date:
        errors.append("Phase 6 must preserve same-date claim grouping.")
    if settings.tie_break != "earlier_date":
        errors.append("Phase 6 boundary tie-breaking must select the earlier date.")
    if settings.min_positive_warning_validation < settings.min_positive_block_validation:
        errors.append("Validation warning threshold cannot be below its blocking threshold.")
    if settings.min_positive_warning_test < settings.min_positive_block_test:
        errors.append("Test warning threshold cannot be below its blocking threshold.")
    return errors


def settings_as_dict(settings: SplitSettings) -> dict[str, Any]:
    """Return a stable JSON-compatible settings payload."""

    return settings.model_dump(mode="json")
