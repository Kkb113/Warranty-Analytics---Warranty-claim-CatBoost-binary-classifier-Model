"""Typed models for versioned Phase 4 policy contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

FeaturePolicyName = Literal[
    "TARGET_ONLY",
    "CONTROL_ONLY",
    "ALLOW_BASELINE_POC",
    "ALLOW_HISTORICAL_POC",
    "RESTRICTED_EXPERIMENTAL",
    "REQUIRES_CONFIRMATION",
    "PROHIBITED",
]

FEATURE_POLICY_NAMES = frozenset(
    {
        "TARGET_ONLY",
        "CONTROL_ONLY",
        "ALLOW_BASELINE_POC",
        "ALLOW_HISTORICAL_POC",
        "RESTRICTED_EXPERIMENTAL",
        "REQUIRES_CONFIRMATION",
        "PROHIBITED",
    }
)


class Phase4ContractError(ValueError):
    """Raised when a Phase 4 contract is malformed or internally unsafe."""


class DevelopmentStatus(BaseModel):
    """Explicit distinction between synthetic development and production use."""

    model_config = ConfigDict(extra="forbid")

    development_mode: Literal["synthetic_poc"]
    production_approved: bool
    real_data_reapproval_required: bool
    business_target_definition_confirmed: bool
    generator_source_confirmed: bool = False
    exact_submission_timestamp_available: bool


class TargetGenerationEvidence(BaseModel):
    """Descriptive Phase 3 evidence that must not become a target rule."""

    model_config = ConfigDict(extra="forbid")

    empirically_deterministic_from_total_claim_cost: bool
    candidate_separator: float | None
    exceptions_observed: int = Field(ge=0)
    maximum_negative_total_claim_cost: float | None = None
    minimum_positive_total_claim_cost: float | None = None
    business_rule_approved: bool
    generator_source_confirmed: bool
    interpretation: str


class EligibilityCategory(BaseModel):
    """One mutually exclusive claim-eligibility result category."""

    model_config = ConfigDict(extra="forbid")

    code: str
    description: str
    priority: int = Field(ge=1)


class EligibilityPolicy(BaseModel):
    """Claim-level supervised-learning eligibility rules."""

    model_config = ConfigDict(extra="forbid")

    row_grain: str
    required_conditions: list[str] = Field(min_length=1)
    excluded_conditions: list[str] = Field(min_length=1)
    categories: list[EligibilityCategory] = Field(min_length=1)
    no_silent_drop: bool


class TargetContract(BaseModel):
    """The stored target and its non-business-approved development status."""

    model_config = ConfigDict(extra="forbid")

    contract_version: str
    version: str
    created_at: str
    schema_contract_version: str
    schema_contract_checksum: str
    source_documents: list[str] = Field(min_length=1)
    development_status: DevelopmentStatus
    target_name: str
    source_table: str
    source_column: str
    target_type: Literal["binary"]
    positive_value: int
    negative_value: int
    prediction_grain: str
    prediction_point: str
    prediction_reference: str
    prediction_reference_status: str
    business_definition_status: str
    synthetic_development_status: str
    target_generation_evidence: TargetGenerationEvidence
    eligibility_policy: EligibilityPolicy
    prohibited_derivation_fields: list[str] = Field(min_length=1)
    known_limitations: list[str] = Field(min_length=1)
    source_schema_contract_version: str
    source_schema_contract_checksum: str


class HistoricalSourcePolicy(BaseModel):
    """As-of and same-day rule for one historical source."""

    model_config = ConfigDict(extra="forbid")

    source_table: str
    event_date_column: str | None
    qualification_rule: str
    same_day_policy: Literal["exclude"]
    current_record_exclusion: str | None = None
    completion_gate: str | None = None
    notes: str


class FeaturePolicyEntry(BaseModel):
    """Policy metadata for exactly one included schema column."""

    model_config = ConfigDict(extra="forbid")

    table: str
    column: str
    policy: FeaturePolicyName
    role: str
    availability_basis: str
    as_of_rule: str
    leakage_risks: list[str]
    synthetic_poc_allowed: bool
    production_approved: bool
    reason: str
    evidence_source: str
    notes: str
    is_model_feature: bool = False
    is_lineage: bool = False
    current_claim_use: str | None = None

    @property
    def field_name(self) -> str:
        """Return a stable fully-qualified field identity."""

        return f"{self.table}.{self.column}"


class FeatureTierPolicy(BaseModel):
    """Machine-readable tier rule consumed by future feature-mart code."""

    model_config = ConfigDict(extra="forbid")

    description: str
    included_policies: list[FeaturePolicyName]
    requires_as_of_enforcement: bool
    restricted_evaluation_only: bool


class FeaturePolicyContract(BaseModel):
    """Versioned policy covering every included schema column."""

    model_config = ConfigDict(extra="forbid")

    contract_version: str
    version: str
    created_at: str
    schema_contract_version: str
    schema_contract_checksum: str
    source_documents: list[str] = Field(min_length=1)
    development_status: DevelopmentStatus
    policy_defaults: dict[str, dict[str, Any]] = Field(default_factory=dict)
    feature_policy_enum: list[FeaturePolicyName] = Field(min_length=1)
    prediction_time_policy: dict[str, Any]
    same_day_policy: dict[str, Any]
    historical_sources: dict[str, HistoricalSourcePolicy]
    feature_tiers: dict[str, FeatureTierPolicy]
    lineage_fields: list[str] = Field(min_length=1)
    excluded_tables: list[str] = Field(min_length=1)
    field_policies: list[FeaturePolicyEntry] = Field(min_length=1)


class LeakageRule(BaseModel):
    """One hard leakage rule or wildcard exclusion."""

    model_config = ConfigDict(extra="forbid")

    field: str
    leakage_types: list[str] = Field(min_length=1)
    reason: str
    applies_to: str


class LeakagePolicyContract(BaseModel):
    """Versioned hard blacklist and leakage-classification policy."""

    model_config = ConfigDict(extra="forbid")

    contract_version: str
    version: str
    created_at: str
    schema_contract_version: str
    schema_contract_checksum: str
    source_documents: list[str] = Field(min_length=1)
    development_status: DevelopmentStatus
    leakage_categories: list[str] = Field(min_length=1)
    hard_blacklist: list[LeakageRule] = Field(min_length=1)
    identifier_fields: list[str] = Field(min_length=1)
    high_cardinality_group_fields: list[str] = Field(min_length=1)
    excluded_tables: list[str] = Field(min_length=1)
    current_claim_exclusions: list[str] = Field(min_length=1)
    historical_rules: dict[str, str] = Field(min_length=1)


class Phase4ContractBundle(BaseModel):
    """Loaded Phase 4 contracts and exact source-file checksums."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    target: TargetContract
    feature_policy: FeaturePolicyContract
    leakage: LeakagePolicyContract
    target_checksum: str
    feature_policy_checksum: str
    leakage_checksum: str


class PolicyValidationResult(BaseModel):
    """Serializable offline contract-validation result."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    errors: list[str]
    warnings: list[str]
    schema_columns: int
    classified_columns: int
    unclassified_columns: int
    policy_counts: dict[str, int]
    safe_baseline_allowlist: list[str]
    historical_allowlist: list[str]
    restricted_experimental_list: list[str]
    requires_confirmation_list: list[str]
    lineage_fields: list[str]
    target_checksum: str | None = None
    feature_policy_checksum: str | None = None
    leakage_policy_checksum: str | None = None


class EligibilityValidationResult(BaseModel):
    """Serializable claim eligibility and target audit result."""

    model_config = ConfigDict(extra="forbid")

    total_claims: int = Field(ge=0)
    eligible_claims: int = Field(ge=0)
    excluded_claims: int = Field(ge=0)
    category_counts: dict[str, int]
    invalid_target_claims: int = Field(ge=0)
    null_target_claims: int = Field(ge=0)
    missing_claim_date_claims: int = Field(ge=0)
    duplicate_claim_key_claims: int = Field(ge=0)
    unresolved_truck_link_claims: int = Field(ge=0)
    positive_claims: int = Field(ge=0)
    negative_claims: int = Field(ge=0)
    positive_percentage: float = Field(ge=0)
    target_valid: bool
    target_generation_audit: dict[str, Any]


class Phase4ValidationResult(BaseModel):
    """Combined offline and live Phase 4 validation result."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["READY", "READY WITH WARNINGS", "BLOCKED"]
    errors: list[str]
    warnings: list[str]
    contract_validation: PolicyValidationResult
    target_validation: dict[str, Any]
    schema_validation: dict[str, Any] | None = None
    source_policy_validation: dict[str, Any]
    leakage_policy_validation: dict[str, Any]
    checksums: dict[str, str]
    execution_timestamp: str
    report_directory: str | None = None
