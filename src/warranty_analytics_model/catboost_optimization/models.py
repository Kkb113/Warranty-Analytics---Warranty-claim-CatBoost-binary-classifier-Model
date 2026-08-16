"""Typed Phase 10 settings, fold plans, study results, and input bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..baseline_model.models import FeatureSetSpec, Phase9Inputs


class OptimizationError(ValueError):
    """Raised when a Phase 10 contract, safety, or reproducibility gate fails."""


@dataclass(frozen=True, slots=True)
class OptimizationSettings:
    """Fail-closed Phase 10 configuration."""

    tracks: tuple[str, ...]
    trials_per_track: int
    parallel_jobs: int
    pruning: bool
    threshold: float
    random_seed: int
    sampler: str
    n_startup_trials: int
    fixed_parameters: dict[str, Any]
    search_space: dict[str, Any]
    inner_fold_fractions: tuple[float, ...]
    minimum_train_positive: int
    minimum_validation_positive: int
    instability_std_ap_warning: float
    failure_warning_fraction: float
    minimum_completed_fraction: float
    output_directory: str
    report_directory: str
    compression: str


@dataclass(frozen=True, slots=True)
class Phase10Inputs:
    """Locked Phase 9 inputs plus the exact Phase 10 E1/E3 feature tracks."""

    root: Path
    phase9_dir: Path
    phase9_manifest: dict[str, Any]
    phase9_inputs: Phase9Inputs
    feature_sets: dict[str, FeatureSetSpec]
    development: pd.DataFrame
    claim_snapshot_path: Path


@dataclass(frozen=True, slots=True)
class InnerFold:
    """One chronological expanding inner fold."""

    fold_id: int
    train_keys: tuple[int, ...]
    validation_keys: tuple[int, ...]
    train_max_date: str
    validation_min_date: str
    validation_max_date: str
    train_rows: int
    validation_rows: int
    train_positive_count: int
    validation_positive_count: int
    train_membership_sha256: str
    validation_membership_sha256: str


@dataclass(frozen=True, slots=True)
class InnerFoldPlan:
    """Persistable fold membership and aggregate manifest."""

    assignments: pd.DataFrame
    folds: tuple[InnerFold, ...]
    manifest: dict[str, Any]
    content_sha256: str


@dataclass(frozen=True, slots=True)
class TrialEvaluation:
    """One successful trial's aggregate and per-fold metrics."""

    params: dict[str, Any]
    fold_metrics: tuple[dict[str, Any], ...]
    aggregate: dict[str, Any]
    training_seconds: float


@dataclass(slots=True)
class StudyResult:
    """One independent Optuna study and its canonical history."""

    track: str
    phase9_experiment_id: str
    study_name: str
    trial_history: pd.DataFrame
    fold_metrics: pd.DataFrame
    baseline_inner_cv_metrics: dict[str, Any]
    best_trial_number: int
    best_params: dict[str, Any]
    best_inner_metrics: dict[str, Any]
    best_param_sha256: str
    warnings: list[str] = field(default_factory=list)
