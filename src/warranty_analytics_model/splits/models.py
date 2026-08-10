"""Typed models for the Phase 6 split contract and artifacts."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SplitName = Literal["TRAIN", "VALIDATION", "TEST"]


class SplitError(ValueError):
    """Raised when a Phase 6 split is unsafe or internally inconsistent."""


class SplitSettings(BaseModel):
    """Technical evaluation-design settings loaded from ``configs/splits.yaml``."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["chronological"] = "chronological"
    train_fraction: float = Field(default=0.70, gt=0.0, lt=1.0)
    validation_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_fraction: float = Field(default=0.15, gt=0.0, lt=1.0)
    preserve_same_date: bool = True
    tie_break: Literal["earlier_date"] = "earlier_date"
    fraction_warning_tolerance: float = Field(default=0.03, ge=0.0, lt=1.0)
    minimum_split_fraction: float = Field(default=0.10, gt=0.0, lt=1.0)
    min_positive_block_validation: int = Field(default=10, ge=0)
    min_positive_block_test: int = Field(default=10, ge=0)
    min_positive_warning_validation: int = Field(default=20, ge=0)
    min_positive_warning_test: int = Field(default=20, ge=0)
    min_positive_warning_train: int = Field(default=100, ge=0)
    group_exposure_enabled: bool = True
    fingerprint_clean_cohort_enabled: bool = True

    @property
    def requested_fractions(self) -> dict[str, float]:
        """Return fractions in the stable contract ordering."""

        return {
            "TRAIN": self.train_fraction,
            "VALIDATION": self.validation_fraction,
            "TEST": self.test_fraction,
        }


class SplitContract(BaseModel):
    """Versioned machine-readable Phase 6 split contract."""

    model_config = ConfigDict(extra="forbid")

    contract_version: str
    created_at: str
    input_mart_contract_version: str
    input_mart_contract_checksum: str
    input_schema_contract_checksum: str
    input_target_contract_checksum: str
    input_feature_policy_checksum: str
    input_leakage_policy_checksum: str
    split_strategy: str
    prediction_reference: str
    requested_fractions: dict[str, float]
    boundary_algorithm: dict[str, Any]
    tie_breaking_rule: str
    same_day_policy: dict[str, Any]
    class_sufficiency_policy: dict[str, Any]
    test_access_policy: dict[str, Any]
    group_exposure_policy: dict[str, Any]
    scenario_fingerprint_policy: dict[str, Any]
    artifact_layout: dict[str, Any]
    validation_policy: dict[str, Any]
    development_status: dict[str, Any]


class BoundaryResult(BaseModel):
    """Date boundaries selected from claim dates and row counts only."""

    model_config = ConfigDict(extra="forbid")

    total_claims: int
    unique_dates: int
    train_target_count: float
    validation_end_target_count: float
    train_end_date: date
    validation_end_date: date
    train_date_count: int
    validation_date_count: int
    test_date_count: int
    train_count: int
    validation_count: int
    test_count: int


class ContractValidationResult(BaseModel):
    """Result of the offline Phase 6 contract gate."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    split_contract_checksum: str
    mart_contract_checksum: str
    requested_fractions: dict[str, float]


class Phase6BuildResult(BaseModel):
    """Aggregate result returned by a Phase 6 build."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["PASS", "PASS WITH WARNINGS", "BLOCKED", "INCOMPLETE"]
    run_directory: str
    report_directory: str | None = None
    manifest_path: str | None = None
    validation_path: str | None = None
    manifest: dict[str, Any]
    validation: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
