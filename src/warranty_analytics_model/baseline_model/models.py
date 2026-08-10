"""Typed Phase 9 settings, inputs, feature sets, and experiment results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


class BaselineModelError(ValueError):
    """Raised when a Phase 9 safety, contract, or reproducibility gate fails."""


@dataclass(frozen=True, slots=True)
class BaselineSettings:
    """Fixed Phase 9 model and adapter settings."""

    model_family: str
    random_seed: int
    catboost_parameters: dict[str, Any]
    text_processing: dict[str, Any]
    categorical_missing_value: str
    text_missing_value: str
    fixed_threshold: float
    primary_metric: str
    output_directory: str
    report_directory: str
    compression: str


@dataclass(frozen=True, slots=True)
class FeatureSetSpec:
    """One deterministic experiment feature set resolved from lineage."""

    experiment_id: str
    feature_names: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    boolean_features: tuple[str, ...]
    text_features: tuple[str, ...]
    phase7_core_count: int
    phase7_extended_count: int
    phase8_lexical_count: int
    phase8_text_count: int
    feature_set_sha256: str

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "feature_count": self.feature_count,
            "numeric_feature_count": len(self.numeric_features),
            "categorical_feature_count": len(self.categorical_features),
            "boolean_feature_count": len(self.boolean_features),
            "text_feature_count": len(self.text_features),
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "boolean_features": list(self.boolean_features),
            "text_features": list(self.text_features),
            "phase7_core_count": self.phase7_core_count,
            "phase7_extended_count": self.phase7_extended_count,
            "phase8_lexical_count": self.phase8_lexical_count,
            "phase8_text_count": self.phase8_text_count,
            "feature_names": list(self.feature_names),
            "feature_set_sha256": self.feature_set_sha256,
        }


@dataclass(frozen=True, slots=True)
class Phase9Inputs:
    """Validated locked Phase 5–8 inputs consumed by Phase 9."""

    root: Path
    mart_dir: Path
    split_dir: Path
    structured_dir: Path
    text_dir: Path
    assignments: pd.DataFrame
    structured_features: pd.DataFrame
    text_features: pd.DataFrame
    phase7_lineage: dict[str, dict[str, Any]]
    phase8_lineage: dict[str, dict[str, Any]]
    phase5_manifest: dict[str, Any]
    phase6_manifest: dict[str, Any]
    phase7_manifest: dict[str, Any]
    phase8_manifest: dict[str, Any]
    test_lock: dict[str, Any]
    upstream_validations: dict[str, dict[str, Any]]
    frozen_membership: dict[str, Any]
    source_audit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DevelopmentTargets:
    """Authorized TRAIN and VALIDATION labels only."""

    train: pd.DataFrame
    validation: pd.DataFrame
    train_target_content_sha256: str
    validation_target_content_sha256: str
    audit: dict[str, Any]


@dataclass(slots=True)
class ExperimentResult:
    """One completed or explicitly unavailable Phase 9 experiment."""

    experiment_id: str
    model_type: str
    status: str
    feature_set: FeatureSetSpec | None
    metrics: dict[str, Any]
    probabilities: pd.Series | None
    model: Any | None = None
    model_file: str | None = None
    model_sha256: str | None = None
    training_seconds: float = 0.0
    prediction_seconds: float = 0.0
    warning: str | None = None
    effective_parameters: dict[str, Any] = field(default_factory=dict)
    validation_probability_sha256: str | None = None
