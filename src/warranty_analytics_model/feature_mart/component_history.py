"""Truck-level pre-claim component-installation history bridge."""

from __future__ import annotations

import pandas as pd

from .common import (
    as_datetime,
    assert_pair_unique,
    assert_unique_key,
    deterministic_sort,
    history_diagnostics,
    merge_many_to_one,
)
from .models import FeatureMartError

INSTALLATION_FIELDS = ("quality_check_status", "rework_flag", "torque_value", "inspection_score")
COMPONENT_FIELDS = (
    "component_system",
    "component_category",
    "standard_life_miles",
    "standard_life_months",
    "is_safety_critical",
    "unit_cost",
)


def build_component_installation_history(
    eligible_claims: pd.DataFrame,
    installations: pd.DataFrame,
    components: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | float], dict[str, int]]:
    """Preserve all pre-claim installations without using current diagnosis."""

    required_claim = {"warranty_claim_key", "truck_key", "claim_date"}
    required_installation = {
        "installation_key",
        "truck_key",
        "component_key",
        "supplier_key",
        "component_lot_no",
        "installed_date",
        "production_batch_id",
        *INSTALLATION_FIELDS,
    }
    required_component = {"component_key", *COMPONENT_FIELDS}
    if missing := sorted(required_claim - set(eligible_claims.columns)):
        raise FeatureMartError(f"Component history claims are missing: {', '.join(missing)}")
    if missing := sorted(required_installation - set(installations.columns)):
        raise FeatureMartError(f"Component installation source is missing: {', '.join(missing)}")
    if missing := sorted(required_component - set(components.columns)):
        raise FeatureMartError(f"Component dimension is missing: {', '.join(missing)}")
    assert_unique_key(installations, "installation_key", "fact_component_installation")
    components = components.copy()
    assert_unique_key(components, "component_key", "dim_component")
    component_columns = ["component_key", *COMPONENT_FIELDS]
    enriched, join_validation = merge_many_to_one(
        installations[
            [
                "installation_key",
                "truck_key",
                "component_key",
                "supplier_key",
                "component_lot_no",
                "installed_date",
                "production_batch_id",
                *INSTALLATION_FIELDS,
            ]
        ].copy(),
        components[component_columns],
        on="component_key",
        label="component-installation-to-component",
    )
    claims = eligible_claims[["warranty_claim_key", "truck_key", "claim_date"]].copy()
    claims["claim_date"] = as_datetime(claims["claim_date"])
    enriched["installed_date"] = as_datetime(enriched["installed_date"])
    merged = claims.merge(enriched, on="truck_key", how="inner", validate="many_to_many")
    merged = merged.loc[merged["installed_date"] < merged["claim_date"]].copy()
    output = pd.DataFrame(
        {
            "current_warranty_claim_key": merged["warranty_claim_key"],
            "installation_key": merged["installation_key"],
            "lineage__truck_key": merged["truck_key"],
            "component_key": merged["component_key"],
            "supplier_key": merged["supplier_key"],
            "component_lot_no": merged["component_lot_no"],
            "production_batch_id": merged["production_batch_id"],
            "component_installation__installed_date": merged["installed_date"],
        }
    )
    for field in INSTALLATION_FIELDS:
        output[f"component_installation__{field}"] = merged[field]
    for field in COMPONENT_FIELDS:
        output[f"component__{field}"] = merged[field]
    output = deterministic_sort(output, ["current_warranty_claim_key", "installation_key"])
    assert_pair_unique(output, ["current_warranty_claim_key", "installation_key"], "component")
    if not output.empty:
        claim_dates = (
            claims.set_index("warranty_claim_key")
            .loc[output["current_warranty_claim_key"], "claim_date"]
            .reset_index(drop=True)
        )
        if bool(
            (as_datetime(output["component_installation__installed_date"]) >= claim_dates).any()
        ):
            raise FeatureMartError("Component bridge contains a same-day or future installation.")
    return output, history_diagnostics(eligible_claims, output), join_validation
