"""Configuration loading for deterministic Phase 8 text candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..paths import discover_repository_root
from .models import TextFeatureError, TextFeatureSettings


def load_text_feature_settings(project_root: Path | None = None) -> TextFeatureSettings:
    """Load and validate the non-secret Phase 8 YAML configuration."""

    root = discover_repository_root(project_root)
    path = root / "configs" / "text_features.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TextFeatureError(f"Could not read Phase 8 configuration: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("text_features"), dict):
        raise TextFeatureError("Phase 8 configuration must contain a text_features mapping.")
    settings = payload["text_features"]
    normalization = settings.get("normalization", {})
    if not isinstance(normalization, dict):
        raise TextFeatureError("Phase 8 normalization configuration must be a mapping.")
    windows = tuple(int(value) for value in settings.get("windows_months", (6, 12, 24)))
    if windows != (6, 12, 24):
        raise TextFeatureError("Phase 8 windows must be exactly 6m, 12m, and 24m.")
    if not bool(settings.get("include_all_history", True)):
        raise TextFeatureError("Phase 8 must include the all-history document.")
    return TextFeatureSettings(
        windows_months=windows,
        include_all_history=True,
        unicode_form=str(normalization.get("unicode_form", "NFKC")),
        lowercase=bool(normalization.get("lowercase", True)),
        collapse_whitespace=bool(normalization.get("collapse_whitespace", True)),
        trim=bool(normalization.get("trim", True)),
        preserve_punctuation=bool(normalization.get("preserve_punctuation", True)),
        preserve_numbers=bool(normalization.get("preserve_numbers", True)),
        document_separator=str(settings.get("document_separator", " [SEP] ")),
        output_directory=str(settings.get("output_directory", "artifacts/text_features")),
        report_directory=str(settings.get("report_directory", "reports/phase8_text_features")),
        compression=str(settings.get("compression", "snappy")),
        validate_after_build=bool(settings.get("validate_after_build", True)),
    )


def settings_payload(settings: TextFeatureSettings) -> dict[str, Any]:
    """Return the stable settings representation used in manifests."""

    return {
        "windows_months": list(settings.windows_months),
        "include_all_history": settings.include_all_history,
        "normalization": {
            "unicode_form": settings.unicode_form,
            "lowercase": settings.lowercase,
            "collapse_whitespace": settings.collapse_whitespace,
            "trim": settings.trim,
            "preserve_punctuation": settings.preserve_punctuation,
            "preserve_numbers": settings.preserve_numbers,
        },
        "document_separator": settings.document_separator,
        "output_directory": settings.output_directory,
        "report_directory": settings.report_directory,
        "compression": settings.compression,
        "validate_after_build": settings.validate_after_build,
    }
