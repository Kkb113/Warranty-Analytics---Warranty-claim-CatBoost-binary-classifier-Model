"""Typed models and feature metadata for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd


class StructuredFeatureError(ValueError):
    """Raised when a Phase 7 contract, input, or artifact is unsafe."""


FeatureTier = Literal["CORE", "EXTENDED", "CONTROL"]
FeatureType = Literal["numeric", "categorical", "boolean", "date_control"]


@dataclass(frozen=True, slots=True)
class StructuredFeatureSettings:
    """Technical feature-engineering settings; policy remains versioned in YAML."""

    windows_months: tuple[int, ...] = (3, 6, 12, 24)
    include_all_history: bool = True
    std_min_observations: int = 2
    slope_min_observations: int = 3
    output_directory: str = "artifacts/structured_features"
    report_directory: str = "reports/phase7_structured_features"
    compression: str = "snappy"
    write_manifest: bool = True
    validate_after_build: bool = True


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """Complete metadata for one output column."""

    feature_name: str
    family: str
    tier: FeatureTier
    feature_type: FeatureType
    source_artifacts: tuple[str, ...] = ()
    source_columns: tuple[str, ...] = ()
    value_sources: tuple[str, ...] = ()
    control_sources: tuple[str, ...] = ()
    window: str | None = None
    aggregation: str | None = None
    null_behavior: str = "preserve_null"
    minimum_observations: int | None = None
    target_dependent: bool = False
    fitted_transformation: str | None = None
    is_model_feature: bool = True
    is_control: bool = False
    is_lineage: bool = False
    phase4_source_policy: str = "ALLOW_BASELINE_POC"
    formula: str = ""
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return stable JSON metadata."""

        return {
            "feature_name": self.feature_name,
            "family": self.family,
            "tier": self.tier,
            "feature_type": self.feature_type,
            "source_artifacts": list(self.source_artifacts),
            "source_columns": list(self.source_columns),
            "value_sources": list(self.value_sources),
            "control_sources": list(self.control_sources),
            "window": self.window,
            "aggregation": self.aggregation,
            "null_behavior": self.null_behavior,
            "minimum_observations": self.minimum_observations,
            "target_dependent": self.target_dependent,
            "fitted_transformation": self.fitted_transformation,
            "is_model_feature": self.is_model_feature,
            "is_control": self.is_control,
            "is_lineage": self.is_lineage,
            "phase4_source_policy": self.phase4_source_policy,
            "formula": self.formula,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class Phase7Inputs:
    """Validated, offline Phase 5/6 inputs consumed by Phase 7."""

    root: Any
    mart_dir: Any
    split_dir: Any
    mart_manifest: dict[str, Any]
    split_manifest: dict[str, Any]
    test_lock: dict[str, Any]
    frames: dict[str, pd.DataFrame]
    phase5_validation: dict[str, Any]
    phase6_validation: dict[str, Any]
    phase5_contract_checksum: str
    phase6_contract_checksum: str
    phase5_manifest_checksum: str


@dataclass(slots=True)
class FeatureBuildResult:
    """In-memory feature matrix and metadata before artifact publication."""

    frame: pd.DataFrame
    definitions: list[FeatureDefinition]
    source_coverage: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
