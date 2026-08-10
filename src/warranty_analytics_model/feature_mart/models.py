"""Typed models used by the Phase 5 claim feature-mart workflow."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MartTransform = Literal[
    "direct",
    "rename",
    "dimension_join",
    "temporal_filter",
    "history_bridge",
    "hash_lineage",
]
MappingStatus = Literal["MATERIALIZED", "DEFERRED_WITH_REASON"]


class FeatureMartError(ValueError):
    """Raised when a Phase 5 contract or artifact is unsafe."""


class FieldMapping(BaseModel):
    """Lineage and safety metadata for one materialized output column."""

    model_config = ConfigDict(extra="forbid")

    artifact: str = "claim_snapshot"
    output_column: str
    source_table: str
    source_column: str
    policy: str
    is_model_feature: bool = False
    is_target: bool = False
    is_lineage: bool = False
    is_control: bool = False
    transform_type: MartTransform
    join_path: list[str] = Field(default_factory=list)
    as_of_rule: str | None = None
    mapping_status: MappingStatus = "MATERIALIZED"
    defer_reason: str | None = None


class HistoricalBridgeDefinition(BaseModel):
    """Contract for one one-to-many historical source bridge."""

    model_config = ConfigDict(extra="forbid")

    name: str
    artifact: str
    source_table: str
    grain: str
    pair_key: list[str] = Field(min_length=2)
    as_of_rule: str
    same_day_policy: Literal["exclude"]
    current_record_exclusion: str | None = None
    field_mappings: list[FieldMapping] = Field(default_factory=list)
    control_mappings: list[FieldMapping] = Field(default_factory=list)


class MartContract(BaseModel):
    """Versioned, machine-readable Phase 5 mart contract."""

    model_config = ConfigDict(extra="forbid")

    contract_version: str
    version: str
    created_at: str
    schema_contract_version: str
    schema_contract_checksum: str
    target_contract_version: str
    target_contract_checksum: str
    feature_policy_version: str
    feature_policy_checksum: str
    leakage_policy_version: str
    leakage_policy_checksum: str
    mart_grain: str
    target: dict[str, Any]
    prediction_reference: dict[str, Any]
    direct_feature_mappings: list[FieldMapping] = Field(min_length=1)
    lineage_mappings: list[FieldMapping] = Field(default_factory=list)
    historical_bridge_definitions: list[HistoricalBridgeDefinition] = Field(min_length=1)
    restricted_artifact_definitions: list[dict[str, Any]] = Field(default_factory=list)
    artifact_layout: dict[str, Any]
    column_naming_policy: dict[str, Any]
    serialization_policy: dict[str, Any]
    deferred_fields: list[dict[str, Any]] = Field(default_factory=list)
    source_join_paths: dict[str, Any]
    safety_rules: dict[str, Any]
    artifact_column_mappings: list[FieldMapping] = Field(default_factory=list)


class FeatureMartSettings(BaseModel):
    """Technical build settings; business policy remains in contracts."""

    model_config = ConfigDict(extra="forbid")

    output_directory: str = "artifacts/feature_mart"
    report_directory: str = "reports/phase5_feature_mart"
    serialization_format: Literal["parquet"] = "parquet"
    row_sort_key: str = "warranty_claim_key"
    history_chunk_size: int = Field(default=10_000, ge=1)
    compression: Literal["snappy", "gzip", "brotli", "none"] = "snappy"
    write_manifest: bool = True
    write_field_lineage: bool = True
    validate_after_build: bool = True


class MartPlanValidationResult(BaseModel):
    """Serializable result of the database-independent Phase 5 plan gate."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    errors: list[str]
    warnings: list[str]
    mart_contract_checksum: str
    direct_expected: int
    direct_materialized: int
    direct_deferred: int
    direct_fields: list[str]
    historical_expected: int
    historical_mapped: int
    historical_deferred: int
    historical_fields: list[str]
    excluded_tables: list[str]


class Phase5BuildResult(BaseModel):
    """Aggregate result returned by a Phase 5 build and used for reporting."""

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
