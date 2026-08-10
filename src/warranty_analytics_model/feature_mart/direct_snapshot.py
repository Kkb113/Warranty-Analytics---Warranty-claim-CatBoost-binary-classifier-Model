"""Build the one-row-per-eligible-claim snapshot with cardinality gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..policy.target_contract import claim_eligibility_mask, validate_claim_eligibility
from .common import as_datetime, assert_unique_key, deterministic_sort, merge_many_to_one
from .mart_contract import _target_mapping
from .models import FeatureMartError, MartContract


@dataclass(frozen=True, slots=True)
class DirectSnapshotResult:
    """Snapshot frame plus the eligible raw claims and join diagnostics."""

    snapshot: pd.DataFrame
    eligible_claims: pd.DataFrame
    eligibility: dict[str, Any]
    join_validation: dict[str, dict[str, int]]


def _mapping_source(mapping: Any) -> str:
    return f"{mapping.source_table}.{mapping.source_column}"


def _copy_mapping_value(frame: pd.DataFrame, mapping: Any) -> None:
    if mapping.source_table == "derived":
        return
    if mapping.source_column not in frame.columns:
        raise FeatureMartError(
            f"Direct snapshot cannot materialize source column: {_mapping_source(mapping)}"
        )
    frame[mapping.output_column] = frame[mapping.source_column]


def _prepare_dimension(frame: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
    result = frame.copy()
    assert_unique_key(result, key, label)
    return result


def build_direct_snapshot(
    frames: dict[str, pd.DataFrame],
    mart_contract: MartContract,
) -> DirectSnapshotResult:
    """Build direct Tier A fields and lineage without flattening history."""

    required_claim_columns = {
        "warranty_claim_key",
        "claim_date",
        "high_cost_claim_flag",
        "truck_key",
        "service_event_key",
        "service_center_key",
        "failure_code_key",
        "repair_end_date",
    }
    claims = frames["dbo.fact_warranty_claim"].copy()
    missing = sorted(required_claim_columns - set(claims.columns))
    if missing:
        raise FeatureMartError(
            f"Claim extraction is missing required control columns: {', '.join(missing)}"
        )
    claims["claim_date"] = as_datetime(claims["claim_date"])
    trucks = _prepare_dimension(frames["dbo.dim_truck"], "truck_key", "dim_truck")
    eligibility_frame = claims[
        ["warranty_claim_key", "claim_date", "high_cost_claim_flag", "truck_key"]
    ].copy()
    eligibility_result = validate_claim_eligibility(
        eligibility_frame,
        trucks[["truck_key"]],
        audit_cost_columns=(),
    )
    eligible_mask = claim_eligibility_mask(
        eligibility_frame,
        trucks[["truck_key"]],
    )
    eligible_claims = claims.loc[eligible_mask].copy()
    if len(eligible_claims) != eligibility_result.eligible_claims:
        raise FeatureMartError("Phase 4 eligibility mask disagrees with its aggregate validation.")

    snapshot = eligible_claims.copy()
    joins: dict[str, dict[str, int]] = {}
    truck_join_columns = [
        "truck_key",
        "truck_model_key",
        "production_batch_id",
        "warranty_policy_key",
        "manufacturing_plant",
        "assembly_line",
        "build_date",
        "delivery_date",
        "in_service_date",
        "axle_configuration",
        "fuel_type",
        "emission_standard",
    ]
    snapshot, joins["claim_to_truck"] = merge_many_to_one(
        snapshot,
        trucks[truck_join_columns],
        on="truck_key",
        label="claim-to-truck",
    )

    truck_models = _prepare_dimension(
        frames["dbo.dim_truck_model"], "truck_model_key", "dim_truck_model"
    )
    model_columns = [
        "truck_model_key",
        "brand",
        "model_name",
        "model_year",
        "segment",
        "application_type",
        "cab_type",
        "engine_platform",
        "gvwr_class",
    ]
    snapshot, joins["truck_to_model"] = merge_many_to_one(
        snapshot,
        truck_models[model_columns],
        on="truck_model_key",
        label="truck-to-truck-model",
    )

    policies = _prepare_dimension(
        frames["dbo.dim_warranty_policy"], "warranty_policy_key", "dim_warranty_policy"
    ).copy()
    policy_columns = [
        "warranty_policy_key",
        "coverage_months",
        "coverage_miles",
        "coverage_engine_hours",
        "deductible_amount",
        "coverage_type",
        "effective_start_date",
        "effective_end_date",
    ]
    snapshot, joins["truck_to_policy"] = merge_many_to_one(
        snapshot,
        policies[policy_columns],
        on="warranty_policy_key",
        label="truck-to-warranty-policy",
    )
    claim_dates = as_datetime(snapshot["claim_date"])
    policy_start = as_datetime(snapshot["effective_start_date"])
    policy_end = as_datetime(snapshot["effective_end_date"])
    policy_applicable = (
        policy_start.notna()
        & (policy_start <= claim_dates)
        & (policy_end.isna() | (policy_end >= claim_dates))
    )
    for column in (
        "coverage_months",
        "coverage_miles",
        "coverage_engine_hours",
        "deductible_amount",
        "coverage_type",
    ):
        snapshot[column] = snapshot[column].where(policy_applicable, pd.NA)

    dates = _prepare_dimension(frames["dbo.dim_date"], "full_date", "dim_date").copy()
    dates["full_date"] = as_datetime(dates["full_date"])
    snapshot["claim_date"] = claim_dates
    date_columns = [
        "full_date",
        "day_number",
        "day_name",
        "week_number",
        "month_number",
        "month_name",
        "quarter_number",
        "year_number",
        "fiscal_month",
        "fiscal_quarter",
        "fiscal_year",
    ]
    snapshot, joins["claim_to_calendar"] = merge_many_to_one(
        snapshot,
        dates[date_columns].rename(columns={"full_date": "claim_date"}),
        on="claim_date",
        label="claim-to-calendar",
    )

    service_centers = _prepare_dimension(
        frames["dbo.dim_service_center"], "service_center_key", "dim_service_center"
    )
    snapshot, joins["claim_to_service_center"] = merge_many_to_one(
        snapshot,
        service_centers[["service_center_key", "location_key"]],
        on="service_center_key",
        label="claim-to-service-center",
    )
    locations = _prepare_dimension(frames["dbo.dim_location"], "location_key", "dim_location")
    snapshot, joins["service_center_to_location"] = merge_many_to_one(
        snapshot,
        locations[["location_key", "country", "region", "climate_zone", "terrain_type"]],
        on="location_key",
        label="service-center-to-location",
    )

    for mapping in mart_contract.direct_feature_mappings:
        _copy_mapping_value(snapshot, mapping)
    target_mapping = _target_mapping(mart_contract)
    snapshot[target_mapping.output_column] = pd.to_numeric(
        snapshot[target_mapping.source_column], errors="coerce"
    ).astype("int64")

    derived_values: dict[str, Any] = {
        "lineage__policy_applicable": policy_applicable.astype("boolean"),
        "lineage__policy_values_available": policy_applicable.astype("boolean"),
    }
    for mapping in mart_contract.lineage_mappings:
        if mapping.output_column in derived_values:
            snapshot[mapping.output_column] = derived_values[mapping.output_column]
        elif mapping.source_table == "derived":
            raise FeatureMartError(
                f"No derived lineage value was implemented for {mapping.output_column}."
            )
        else:
            _copy_mapping_value(snapshot, mapping)

    artifact_mappings = [
        mapping
        for mapping in [
            *mart_contract.direct_feature_mappings,
            *mart_contract.lineage_mappings,
            target_mapping,
        ]
        if mapping.artifact == "claim_snapshot"
    ]
    output_columns = [mapping.output_column for mapping in artifact_mappings]
    missing_outputs = sorted(set(output_columns) - set(snapshot.columns))
    if missing_outputs:
        raise FeatureMartError(
            f"Claim snapshot is missing declared output columns: {', '.join(missing_outputs)}"
        )
    snapshot = snapshot[output_columns]
    snapshot = deterministic_sort(snapshot, ["warranty_claim_key"])
    if snapshot["warranty_claim_key"].duplicated().any():
        raise FeatureMartError("Claim snapshot warranty_claim_key is not unique.")
    if snapshot["warranty_claim_key"].isna().any():
        raise FeatureMartError("Claim snapshot warranty_claim_key contains null values.")
    return DirectSnapshotResult(
        snapshot=snapshot,
        eligible_claims=eligible_claims,
        eligibility=eligibility_result.model_dump(mode="json"),
        join_validation=joins,
    )
