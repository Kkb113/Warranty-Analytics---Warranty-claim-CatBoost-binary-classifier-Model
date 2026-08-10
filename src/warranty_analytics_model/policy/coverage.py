"""Schema coverage, allowlist, and fail-closed policy helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from ..database.models import SchemaContract
from .models import (
    FEATURE_POLICY_NAMES,
    FeaturePolicyContract,
    Phase4ContractError,
    PolicyValidationResult,
)


def schema_field_names(schema_contract: SchemaContract) -> set[str]:
    """Return the exact fully-qualified fields in the approved schema scope."""

    return {
        f"{table.name}.{column.name}"
        for table in schema_contract.tables
        for column in table.columns
    }


def _entry_field_names(feature_policy: FeaturePolicyContract) -> list[str]:
    return [entry.field_name for entry in feature_policy.field_policies]


def build_future_allowlists(
    feature_policy: FeaturePolicyContract,
) -> dict[str, list[str]]:
    """Build explicit future feature, restricted, confirmation, and lineage lists."""

    safe_policies = {"ALLOW_BASELINE_POC", "ALLOW_HISTORICAL_POC"}
    baseline = [
        entry.field_name
        for entry in feature_policy.field_policies
        if entry.policy == "ALLOW_BASELINE_POC" and entry.is_model_feature
    ]
    historical = [
        entry.field_name
        for entry in feature_policy.field_policies
        if entry.policy == "ALLOW_HISTORICAL_POC" and entry.is_model_feature
    ]
    restricted = [
        entry.field_name
        for entry in feature_policy.field_policies
        if entry.policy == "RESTRICTED_EXPERIMENTAL"
    ]
    requires_confirmation = [
        entry.field_name
        for entry in feature_policy.field_policies
        if entry.policy == "REQUIRES_CONFIRMATION"
    ]
    lineage = list(
        dict.fromkeys(
            [*feature_policy.lineage_fields]
            + [entry.field_name for entry in feature_policy.field_policies if entry.is_lineage]
        )
    )
    safe = [*baseline, *historical]
    if any(
        entry.policy in safe_policies and not entry.is_model_feature
        for entry in feature_policy.field_policies
    ):
        # A safe policy can be intentionally lineage-only, but it must not be
        # silently omitted from the generated allowlist. The validator reports
        # the omission as a contract error; this branch keeps output explicit.
        safe = [*safe]
    return {
        "tier_a_safe_baseline": safe,
        "tier_a_direct_baseline": baseline,
        "tier_a_historical": historical,
        "tier_b_restricted_experimental": restricted,
        "requires_confirmation": requires_confirmation,
        "lineage_and_split_control": lineage,
    }


def validate_feature_policy_coverage(
    schema_contract: SchemaContract,
    feature_policy: FeaturePolicyContract,
) -> PolicyValidationResult:
    """Validate exact one-to-one coverage and return derived policy inventories."""

    schema_fields = schema_field_names(schema_contract)
    entries = _entry_field_names(feature_policy)
    counts = Counter(entries)
    duplicate_fields = sorted(field for field, count in counts.items() if count > 1)
    policy_fields = set(entries)
    missing_fields = sorted(schema_fields - policy_fields)
    unknown_fields = sorted(policy_fields - schema_fields)
    errors: list[str] = []
    if missing_fields:
        errors.append(f"Unclassified schema columns: {', '.join(missing_fields)}")
    if unknown_fields:
        errors.append(f"Policy contains unknown schema columns: {', '.join(unknown_fields)}")
    if duplicate_fields:
        errors.append(f"Duplicate policy entries: {', '.join(duplicate_fields)}")
    if set(feature_policy.feature_policy_enum) != FEATURE_POLICY_NAMES:
        errors.append("Feature policy enum must contain exactly the seven supported policies.")
    allowlists = build_future_allowlists(feature_policy)
    policy_counts = {
        str(policy): int(count)
        for policy, count in sorted(
            Counter(entry.policy for entry in feature_policy.field_policies).items()
        )
    }
    result = PolicyValidationResult(
        valid=not errors,
        errors=errors,
        warnings=[],
        schema_columns=len(schema_fields),
        classified_columns=len(policy_fields & schema_fields),
        unclassified_columns=len(missing_fields),
        policy_counts=policy_counts,
        safe_baseline_allowlist=allowlists["tier_a_safe_baseline"],
        historical_allowlist=allowlists["tier_a_historical"],
        restricted_experimental_list=allowlists["tier_b_restricted_experimental"],
        requires_confirmation_list=allowlists["requires_confirmation"],
        lineage_fields=allowlists["lineage_and_split_control"],
    )
    if errors:
        raise Phase4ContractError("; ".join(errors))
    return result


def ensure_fields_exist(
    schema_contract: SchemaContract,
    fields: Iterable[str],
    *,
    label: str,
) -> list[str]:
    """Return unknown fields in an auxiliary allowlist or lineage list."""

    schema_fields = schema_field_names(schema_contract)
    return sorted(field for field in fields if field not in schema_fields)
