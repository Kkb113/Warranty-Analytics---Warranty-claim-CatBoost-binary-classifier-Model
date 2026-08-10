"""Load Phase 7 technical settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..paths import discover_repository_root
from .models import StructuredFeatureError, StructuredFeatureSettings


def load_structured_feature_settings(
    project_root: Path | None = None, *, path: Path | None = None
) -> StructuredFeatureSettings:
    """Load and validate non-secret Phase 7 settings."""

    root = discover_repository_root(project_root)
    config_path = path or root / "configs" / "structured_features.yaml"
    if not config_path.is_file():
        raise StructuredFeatureError(f"Structured-feature configuration is missing: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StructuredFeatureError(
            f"Could not read Phase 7 configuration: {config_path}"
        ) from exc
    values = payload.get("structured_features") if isinstance(payload, dict) else None
    if not isinstance(values, dict):
        raise StructuredFeatureError(
            "configs/structured_features.yaml must contain a structured_features mapping."
        )
    windows = tuple(int(item) for item in values.get("windows_months", (3, 6, 12, 24)))
    if windows != (3, 6, 12, 24):
        raise StructuredFeatureError("Phase 7 windows must be exactly 3, 6, 12, and 24 months.")
    output_directory = str(values.get("output_directory", "artifacts/structured_features"))
    report_directory = str(values.get("report_directory", "reports/phase7_structured_features"))
    compression = str(values.get("compression", "snappy"))
    if compression not in {"snappy", "gzip", "brotli", "none"}:
        raise StructuredFeatureError(f"Unsupported Phase 7 Parquet compression: {compression}")
    return StructuredFeatureSettings(
        windows_months=windows,
        include_all_history=bool(values.get("include_all_history", True)),
        std_min_observations=int(values.get("std_min_observations", 2)),
        slope_min_observations=int(values.get("slope_min_observations", 3)),
        output_directory=output_directory,
        report_directory=report_directory,
        compression=compression,
        write_manifest=bool(values.get("write_manifest", True)),
        validate_after_build=bool(values.get("validate_after_build", True)),
    )


def technical_settings_dict(settings: StructuredFeatureSettings) -> dict[str, Any]:
    """Return stable settings for a run manifest."""

    return {
        "windows_months": list(settings.windows_months),
        "include_all_history": settings.include_all_history,
        "std_min_observations": settings.std_min_observations,
        "slope_min_observations": settings.slope_min_observations,
        "output_directory": settings.output_directory,
        "report_directory": settings.report_directory,
        "compression": settings.compression,
        "write_manifest": settings.write_manifest,
        "validate_after_build": settings.validate_after_build,
    }
