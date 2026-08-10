"""Typed models and metadata for Phase 8 text candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd


class TextFeatureError(ValueError):
    """Raised when a Phase 8 input, contract, or artifact is unsafe."""


TextTier = Literal["TEXT_CORE", "TEXT_EXTENDED", "CONTROL"]
TextFeatureType = Literal["text", "numeric", "boolean", "categorical", "date_control"]


@dataclass(frozen=True, slots=True)
class TextFeatureSettings:
    """Deterministic Phase 8 settings; policy remains versioned in YAML."""

    windows_months: tuple[int, ...] = (6, 12, 24)
    include_all_history: bool = True
    unicode_form: str = "NFKC"
    lowercase: bool = True
    collapse_whitespace: bool = True
    trim: bool = True
    preserve_punctuation: bool = True
    preserve_numbers: bool = True
    document_separator: str = " [SEP] "
    output_directory: str = "artifacts/text_features"
    report_directory: str = "reports/phase8_text_features"
    compression: str = "snappy"
    validate_after_build: bool = True


@dataclass(frozen=True, slots=True)
class TextFeatureDefinition:
    """Complete lineage metadata for one Phase 8 output column."""

    feature_name: str
    tier: TextTier
    feature_type: TextFeatureType
    source_artifacts: tuple[str, ...] = ()
    source_columns: tuple[str, ...] = ()
    value_sources: tuple[str, ...] = ()
    control_sources: tuple[str, ...] = ()
    window: str | None = None
    aggregation: str | None = None
    transform: tuple[str, ...] = ()
    separator: str | None = None
    target_dependent: bool = False
    fitted_transformation: str | None = None
    is_model_feature: bool = True
    is_control: bool = False
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return stable JSON metadata."""

        return {
            "feature_name": self.feature_name,
            "tier": self.tier,
            "feature_type": self.feature_type,
            "source_artifacts": list(self.source_artifacts),
            "source_columns": list(self.source_columns),
            "value_sources": list(self.value_sources),
            "control_sources": list(self.control_sources),
            "window": self.window,
            "aggregation": self.aggregation,
            "transform": list(self.transform),
            "separator": self.separator,
            "target_dependent": self.target_dependent,
            "fitted_transformation": self.fitted_transformation,
            "is_model_feature": self.is_model_feature,
            "is_control": self.is_control,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class Phase8Inputs:
    """Validated local Phase 5/6/7 artifacts consumed by Phase 8."""

    root: Path
    mart_dir: Path
    split_dir: Path
    structured_dir: Path
    prior_claim_history: pd.DataFrame
    assignments: pd.DataFrame
    phase5_manifest: dict[str, Any]
    phase6_manifest: dict[str, Any]
    test_lock: dict[str, Any]
    phase7_manifest: dict[str, Any]
    phase7_lineage: dict[str, Any]
    phase5_validation: dict[str, Any]
    phase6_validation: dict[str, Any]
    phase7_validation: dict[str, Any]
    phase5_contract_checksum: str
    phase6_contract_checksum: str
    phase5_manifest_checksum: str
    phase7_contract_checksum: str
    phase7_content_sha256: str


@dataclass(slots=True)
class TextBuildResult:
    """In-memory text artifact and metadata before publication."""

    frame: pd.DataFrame
    definitions: list[TextFeatureDefinition]
    quality: dict[str, Any]
    source_coverage: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
