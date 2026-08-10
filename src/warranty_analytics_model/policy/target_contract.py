"""Stored-target validation and claim-level eligibility checks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from ..profiling.target_profile import audit_target_generation
from .models import EligibilityValidationResult, Phase4ContractError, TargetContract


def validate_target_contract(
    schema_contract: Any,
    target_contract: TargetContract,
) -> list[str]:
    """Return blocking target-contract errors against the approved schema."""

    table_map = schema_contract.table_map
    errors: list[str] = []
    if target_contract.target_name != target_contract.source_column:
        errors.append("Target name must equal the stored source column.")
    if target_contract.source_table not in table_map:
        errors.append(
            f"Target source table is not in the schema contract: {target_contract.source_table}"
        )
    elif target_contract.source_column not in table_map[target_contract.source_table].column_map:
        errors.append(
            f"Target source column is not in the schema contract: "
            f"{target_contract.source_table}.{target_contract.source_column}"
        )
    if (target_contract.positive_value, target_contract.negative_value) != (1, 0):
        errors.append("Target positive and negative values must be exactly 1 and 0.")
    if target_contract.prediction_reference != "claim_date":
        errors.append("The provisional prediction reference must remain claim_date.")
    if target_contract.prediction_reference_status != "provisional_date_level":
        errors.append("claim_date must be labeled as a provisional date-level reference.")
    if target_contract.development_status.production_approved:
        errors.append("Phase 4 synthetic policy cannot be marked production approved.")
    if target_contract.development_status.business_target_definition_confirmed:
        errors.append("The business target definition remains unconfirmed.")
    if "total_claim_cost" not in target_contract.prohibited_derivation_fields:
        errors.append("total_claim_cost must be prohibited as a target derivation field.")
    evidence = target_contract.target_generation_evidence
    if evidence.business_rule_approved:
        errors.append("Empirical cost-separation evidence cannot be business approval.")
    if target_contract.source_column == "total_claim_cost":
        errors.append("The target cannot be total_claim_cost.")
    return errors


def _duplicate_claim_keys(claims: pd.DataFrame, key_column: str) -> pd.Series:
    values = claims[key_column]
    return values.notna() & values.duplicated(keep=False)


def _safe_truck_keys(trucks: pd.DataFrame, key_column: str) -> set[object]:
    if key_column not in trucks:
        return set()
    return set(trucks[key_column].dropna().tolist())


def claim_eligibility_mask(
    claims: pd.DataFrame,
    trucks: pd.DataFrame,
    *,
    target_column: str = "high_cost_claim_flag",
    claim_key_column: str = "warranty_claim_key",
    claim_date_column: str = "claim_date",
    truck_key_column: str = "truck_key",
) -> pd.Series:
    """Return the exact Phase 4 eligible-row mask for downstream mart construction."""

    required = {
        claim_key_column,
        claim_date_column,
        target_column,
        truck_key_column,
    }
    missing = sorted(required - set(claims.columns))
    if missing:
        raise Phase4ContractError(f"Eligibility input is missing columns: {', '.join(missing)}")
    target = pd.to_numeric(claims[target_column], errors="coerce")
    claim_dates = pd.to_datetime(claims[claim_date_column], errors="coerce")
    duplicate_keys = _duplicate_claim_keys(claims, claim_key_column)
    missing_keys = claims[claim_key_column].isna()
    null_target = target.isna()
    invalid_target = target.notna() & ~target.isin([0, 1])
    missing_dates = claim_dates.isna()
    truck_keys = _safe_truck_keys(trucks, truck_key_column)
    unresolved_trucks = ~claims[truck_key_column].isin(truck_keys)
    eligible = ~(missing_keys | duplicate_keys | null_target | invalid_target | missing_dates)
    return eligible & ~unresolved_trucks


def validate_claim_eligibility(
    claims: pd.DataFrame,
    trucks: pd.DataFrame,
    *,
    target_column: str = "high_cost_claim_flag",
    claim_key_column: str = "warranty_claim_key",
    claim_date_column: str = "claim_date",
    truck_key_column: str = "truck_key",
    audit_cost_columns: Iterable[str] = ("total_claim_cost",),
) -> EligibilityValidationResult:
    """Classify every claim row without silently dropping any records."""

    required = {
        claim_key_column,
        claim_date_column,
        target_column,
        truck_key_column,
    }
    missing = sorted(required - set(claims.columns))
    if missing:
        raise Phase4ContractError(f"Eligibility input is missing columns: {', '.join(missing)}")

    target = pd.to_numeric(claims[target_column], errors="coerce")
    claim_dates = pd.to_datetime(claims[claim_date_column], errors="coerce")
    duplicate_keys = _duplicate_claim_keys(claims, claim_key_column)
    missing_keys = claims[claim_key_column].isna()
    null_target = target.isna()
    invalid_target = target.notna() & ~target.isin([0, 1])
    missing_dates = claim_dates.isna()
    truck_keys = _safe_truck_keys(trucks, truck_key_column)
    unresolved_trucks = ~claims[truck_key_column].isin(truck_keys)

    categories = pd.Series("ELIGIBLE", index=claims.index, dtype="string")
    # Categories are mutually exclusive with a documented priority so a row is
    # counted once while independent diagnostics below retain all issue counts.
    categories[missing_keys | duplicate_keys] = "INELIGIBLE_INVALID_CLAIM_KEY"
    remaining = categories == "ELIGIBLE"
    categories[remaining & null_target] = "INELIGIBLE_NULL_TARGET"
    remaining = categories == "ELIGIBLE"
    categories[remaining & invalid_target] = "INELIGIBLE_INVALID_TARGET"
    remaining = categories == "ELIGIBLE"
    categories[remaining & missing_dates] = "INELIGIBLE_MISSING_CLAIM_DATE"
    remaining = categories == "ELIGIBLE"
    categories[remaining & unresolved_trucks] = "INELIGIBLE_MISSING_TRUCK_LINK"

    category_counts = {
        str(category): int(count) for category, count in categories.value_counts(sort=False).items()
    }
    eligible = categories == "ELIGIBLE"
    valid_target = target.isin([0, 1])
    positive_claims = int((eligible & (target == 1)).sum())
    negative_claims = int((eligible & (target == 0)).sum())
    eligible_count = int(eligible.sum())
    target_audit = audit_target_generation(
        claims,
        target_column=target_column,
        cost_columns=audit_cost_columns,
    )
    return EligibilityValidationResult(
        total_claims=int(len(claims)),
        eligible_claims=eligible_count,
        excluded_claims=int((~eligible).sum()),
        category_counts=category_counts,
        invalid_target_claims=int(invalid_target.sum()),
        null_target_claims=int(null_target.sum()),
        missing_claim_date_claims=int(missing_dates.sum()),
        duplicate_claim_key_claims=int(duplicate_keys.sum()),
        unresolved_truck_link_claims=int(unresolved_trucks.sum()),
        positive_claims=positive_claims,
        negative_claims=negative_claims,
        positive_percentage=(
            round(positive_claims / eligible_count * 100, 6) if eligible_count else 0.0
        ),
        target_valid=bool(valid_target.all()),
        target_generation_audit=target_audit,
    )
