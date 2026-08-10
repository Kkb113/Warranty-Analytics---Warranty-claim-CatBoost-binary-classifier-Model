"""Strictly pre-claim maintenance history bridge."""

from __future__ import annotations

import pandas as pd

from .common import (
    as_datetime,
    assert_pair_unique,
    assert_unique_key,
    deterministic_sort,
    history_diagnostics,
)
from .models import FeatureMartError

MAINTENANCE_FIELDS = (
    "odometer_miles",
    "engine_hours",
    "maintenance_type",
    "scheduled_flag",
    "completed_on_time_flag",
    "overdue_days",
    "maintenance_cost",
)


def build_maintenance_history(
    eligible_claims: pd.DataFrame,
    maintenance: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Preserve maintenance events with maintenance_date strictly before claim_date."""

    required_claim = {"warranty_claim_key", "truck_key", "claim_date"}
    required_source = {
        "maintenance_event_key",
        "truck_key",
        "service_center_key",
        "maintenance_date",
        *MAINTENANCE_FIELDS,
    }
    if missing := sorted(required_claim - set(eligible_claims.columns)):
        raise FeatureMartError(f"Maintenance history claims are missing: {', '.join(missing)}")
    if missing := sorted(required_source - set(maintenance.columns)):
        raise FeatureMartError(f"Maintenance history source is missing: {', '.join(missing)}")
    assert_unique_key(maintenance, "maintenance_event_key", "fact_maintenance_event")
    claims = eligible_claims[["warranty_claim_key", "truck_key", "claim_date"]].copy()
    claims["claim_date"] = as_datetime(claims["claim_date"])
    source = maintenance[
        [
            "maintenance_event_key",
            "truck_key",
            "service_center_key",
            "maintenance_date",
            *MAINTENANCE_FIELDS,
        ]
    ].copy()
    source["maintenance_date"] = as_datetime(source["maintenance_date"])
    merged = claims.merge(source, on="truck_key", how="inner", validate="many_to_many")
    merged = merged.loc[merged["maintenance_date"] < merged["claim_date"]].copy()
    output = pd.DataFrame(
        {
            "current_warranty_claim_key": merged["warranty_claim_key"],
            "maintenance_event_key": merged["maintenance_event_key"],
            "lineage__truck_key": merged["truck_key"],
            "lineage__maintenance_service_center_key": merged["service_center_key"],
            "maintenance__maintenance_date": merged["maintenance_date"],
        }
    )
    for field in MAINTENANCE_FIELDS:
        output[f"maintenance__{field}"] = merged[field]
    output = deterministic_sort(output, ["current_warranty_claim_key", "maintenance_event_key"])
    assert_pair_unique(
        output, ["current_warranty_claim_key", "maintenance_event_key"], "maintenance"
    )
    if not output.empty:
        claim_dates = (
            claims.set_index("warranty_claim_key")
            .loc[output["current_warranty_claim_key"], "claim_date"]
            .reset_index(drop=True)
        )
        if bool((as_datetime(output["maintenance__maintenance_date"]) >= claim_dates).any()):
            raise FeatureMartError("Maintenance bridge contains a same-day or future event.")
    return output, history_diagnostics(eligible_claims, output)
