"""Read-only prior-repair eligibility/index bridge."""

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


def build_repair_history_index(
    eligible_claims: pd.DataFrame,
    all_claims: pd.DataFrame,
    repair_lines: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Index only repair lines tied to a completed earlier claim."""

    required_current = {"warranty_claim_key", "truck_key", "claim_date"}
    required_prior = {"warranty_claim_key", "truck_key", "repair_end_date"}
    required_repair = {
        "repair_line_key",
        "warranty_claim_key",
        "service_event_key",
        "component_key",
    }
    if missing := sorted(required_current - set(eligible_claims.columns)):
        raise FeatureMartError(f"Repair history claims are missing: {', '.join(missing)}")
    if missing := sorted(required_prior - set(all_claims.columns)):
        raise FeatureMartError(f"Repair history prior claims are missing: {', '.join(missing)}")
    if missing := sorted(required_repair - set(repair_lines.columns)):
        raise FeatureMartError(f"Repair-line source is missing: {', '.join(missing)}")
    assert_unique_key(all_claims, "warranty_claim_key", "fact_warranty_claim")
    assert_unique_key(repair_lines, "repair_line_key", "fact_repair_line")
    current = eligible_claims[["warranty_claim_key", "truck_key", "claim_date"]].copy()
    current["claim_date"] = as_datetime(current["claim_date"])
    prior = (
        all_claims[["warranty_claim_key", "truck_key", "repair_end_date"]]
        .copy()
        .rename(
            columns={
                "warranty_claim_key": "prior_warranty_claim_key",
                "repair_end_date": "prior_repair_end_date",
            }
        )
    )
    prior["prior_repair_end_date"] = as_datetime(prior["prior_repair_end_date"])
    repairs = repair_lines[
        ["repair_line_key", "warranty_claim_key", "service_event_key", "component_key"]
    ].rename(columns={"warranty_claim_key": "repair_claim_key"})
    candidates = prior.merge(
        repairs,
        left_on="prior_warranty_claim_key",
        right_on="repair_claim_key",
        how="inner",
        validate="one_to_many",
    )
    candidates = candidates.merge(current, on="truck_key", how="inner", validate="many_to_many")
    candidates = candidates.loc[
        (candidates["prior_warranty_claim_key"] != candidates["warranty_claim_key"])
        & candidates["prior_repair_end_date"].notna()
        & (candidates["prior_repair_end_date"] < candidates["claim_date"])
    ].copy()
    output = pd.DataFrame(
        {
            "current_warranty_claim_key": candidates["warranty_claim_key"],
            "prior_warranty_claim_key": candidates["prior_warranty_claim_key"],
            "repair_line_key": candidates["repair_line_key"],
            "prior_repair_end_date": candidates["prior_repair_end_date"],
            "component_key": candidates["component_key"],
            "service_event_key": candidates["service_event_key"],
        }
    )
    output = deterministic_sort(
        output,
        ["current_warranty_claim_key", "prior_warranty_claim_key", "repair_line_key"],
    )
    assert_pair_unique(
        output,
        ["current_warranty_claim_key", "repair_line_key"],
        "repair history",
    )
    if not output.empty:
        current_dates = (
            current.set_index("warranty_claim_key")
            .loc[output["current_warranty_claim_key"], "claim_date"]
            .reset_index(drop=True)
        )
        if bool((as_datetime(output["prior_repair_end_date"]) >= current_dates).any()):
            raise FeatureMartError("Repair history contains a same-day or future completion.")
        if bool(
            (
                output["current_warranty_claim_key"].reset_index(drop=True)
                == output["prior_warranty_claim_key"].reset_index(drop=True)
            ).any()
        ):
            raise FeatureMartError("Repair history contains a current-claim repair line.")
    return output, history_diagnostics(eligible_claims, output)
