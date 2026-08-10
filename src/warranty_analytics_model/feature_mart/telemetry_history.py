"""Completed-month telemetry history bridge."""

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

TELEMETRY_FIELDS = (
    "mileage_month",
    "total_odometer_miles",
    "engine_hours_month",
    "idle_hours_month",
    "avg_engine_temp",
    "max_engine_temp",
    "avg_oil_pressure",
    "low_oil_pressure_events",
    "brake_air_pressure_alerts",
    "battery_voltage_alerts",
    "fault_code_count",
    "harsh_braking_events",
    "avg_payload_weight",
    "fuel_efficiency_mpg",
    "route_severity_score",
    "maintenance_compliance_score",
)


def build_telemetry_history(
    eligible_claims: pd.DataFrame,
    telemetry: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Preserve all completed telemetry months before each eligible claim."""

    required_claim = {"warranty_claim_key", "truck_key", "claim_date"}
    required_source = {
        "telemetry_month_key",
        "truck_key",
        "month_start_date",
        *TELEMETRY_FIELDS,
    }
    if missing := sorted(required_claim - set(eligible_claims.columns)):
        raise FeatureMartError(f"Telemetry history claims are missing: {', '.join(missing)}")
    if missing := sorted(required_source - set(telemetry.columns)):
        raise FeatureMartError(f"Telemetry history source is missing: {', '.join(missing)}")
    assert_unique_key(telemetry, "telemetry_month_key", "fact_telemetry_monthly")
    claims = eligible_claims[["warranty_claim_key", "truck_key", "claim_date"]].copy()
    claims["claim_date"] = as_datetime(claims["claim_date"])
    source = telemetry[
        ["telemetry_month_key", "truck_key", "month_start_date", *TELEMETRY_FIELDS]
    ].copy()
    source["month_start_date"] = as_datetime(source["month_start_date"])
    merged = claims.merge(source, on="truck_key", how="inner", validate="many_to_many")
    month_end = merged["month_start_date"] + pd.offsets.MonthEnd(0)
    merged = merged.loc[month_end < merged["claim_date"]].copy()
    output = pd.DataFrame(
        {
            "current_warranty_claim_key": merged["warranty_claim_key"],
            "telemetry_month_key": merged["telemetry_month_key"],
            "lineage__truck_key": merged["truck_key"],
            "telemetry__month_start_date": merged["month_start_date"],
        }
    )
    for field in TELEMETRY_FIELDS:
        output[f"telemetry__{field}"] = merged[field]
    output = deterministic_sort(output, ["current_warranty_claim_key", "telemetry_month_key"])
    assert_pair_unique(output, ["current_warranty_claim_key", "telemetry_month_key"], "telemetry")
    if not output.empty:
        violation = as_datetime(output["telemetry__month_start_date"]) + pd.offsets.MonthEnd(0)
        claim_dates = (
            claims.set_index("warranty_claim_key")
            .loc[output["current_warranty_claim_key"], "claim_date"]
            .reset_index(drop=True)
        )
        if bool((violation >= claim_dates).any()):
            raise FeatureMartError("Telemetry bridge contains a same-day or future month.")
    return output, history_diagnostics(
        eligible_claims,
        output,
        claim_key="warranty_claim_key",
    )
