"""Pre-claim service-event history bridge with current-event exclusion."""

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

SERVICE_FIELDS = ("odometer_miles", "engine_hours", "service_type")


def build_service_history(
    eligible_claims: pd.DataFrame,
    service_events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Preserve strict pre-claim service events except the current claim event."""

    required_claim = {"warranty_claim_key", "truck_key", "claim_date", "service_event_key"}
    required_source = {
        "service_event_key",
        "truck_key",
        "service_center_key",
        "service_date",
        *SERVICE_FIELDS,
    }
    if missing := sorted(required_claim - set(eligible_claims.columns)):
        raise FeatureMartError(f"Service history claims are missing: {', '.join(missing)}")
    if missing := sorted(required_source - set(service_events.columns)):
        raise FeatureMartError(f"Service history source is missing: {', '.join(missing)}")
    assert_unique_key(service_events, "service_event_key", "fact_service_event")
    claims = eligible_claims[
        ["warranty_claim_key", "truck_key", "claim_date", "service_event_key"]
    ].copy()
    claims["claim_date"] = as_datetime(claims["claim_date"])
    source = service_events[
        ["service_event_key", "truck_key", "service_center_key", "service_date", *SERVICE_FIELDS]
    ].rename(columns={"service_event_key": "history_service_event_key"})
    source["service_date"] = as_datetime(source["service_date"])
    merged = claims.merge(source, on="truck_key", how="inner", validate="many_to_many")
    merged = merged.loc[
        (merged["service_date"] < merged["claim_date"])
        & (merged["history_service_event_key"] != merged["service_event_key"])
    ].copy()
    output = pd.DataFrame(
        {
            "current_warranty_claim_key": merged["warranty_claim_key"],
            "service_event_key": merged["history_service_event_key"],
            "lineage__truck_key": merged["truck_key"],
            "lineage__service_center_key": merged["service_center_key"],
            "service__service_date": merged["service_date"],
        }
    )
    for field in SERVICE_FIELDS:
        output[f"service__{field}"] = merged[field]
    output = deterministic_sort(output, ["current_warranty_claim_key", "service_event_key"])
    assert_pair_unique(output, ["current_warranty_claim_key", "service_event_key"], "service")
    if not output.empty:
        claim_dates = (
            claims.set_index("warranty_claim_key")
            .loc[output["current_warranty_claim_key"], "claim_date"]
            .reset_index(drop=True)
        )
        if bool((as_datetime(output["service__service_date"]) >= claim_dates).any()):
            raise FeatureMartError("Service bridge contains a same-day or future event.")
        current_keys = (
            claims.set_index("warranty_claim_key")
            .loc[output["current_warranty_claim_key"], "service_event_key"]
            .reset_index(drop=True)
        )
        if bool((output["service_event_key"].reset_index(drop=True) == current_keys).any()):
            raise FeatureMartError("Service bridge contains current-claim service events.")
    return output, history_diagnostics(eligible_claims, output)
