"""Pre-claim warranty-claim and historical failure-taxonomy bridge."""

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

FAILURE_FIELDS = (
    "failure_code",
    "failure_description",
    "failure_system",
    "failure_category",
    "severity_level",
    "safety_related_flag",
    "recall_related_flag",
)


def build_prior_claim_history(
    eligible_claims: pd.DataFrame,
    all_claims: pd.DataFrame,
    failure_codes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | float], dict[str, int]]:
    """Join earlier claims by truck and expose only historical taxonomy fields."""

    required_current = {"warranty_claim_key", "truck_key", "claim_date"}
    required_prior = {
        "warranty_claim_key",
        "truck_key",
        "claim_date",
        "failure_code_key",
    }
    required_failure = {"failure_code_key", *FAILURE_FIELDS}
    if missing := sorted(required_current - set(eligible_claims.columns)):
        raise FeatureMartError(f"Prior-claim history claims are missing: {', '.join(missing)}")
    if missing := sorted(required_prior - set(all_claims.columns)):
        raise FeatureMartError(f"Prior-claim source is missing: {', '.join(missing)}")
    if missing := sorted(required_failure - set(failure_codes.columns)):
        raise FeatureMartError(f"Failure-code dimension is missing: {', '.join(missing)}")
    assert_unique_key(all_claims, "warranty_claim_key", "fact_warranty_claim")
    failure_codes = failure_codes.copy()
    assert_unique_key(failure_codes, "failure_code_key", "dim_failure_code")
    current = eligible_claims[["warranty_claim_key", "truck_key", "claim_date"]].copy()
    current["claim_date"] = as_datetime(current["claim_date"])
    prior = all_claims[["warranty_claim_key", "truck_key", "claim_date", "failure_code_key"]].copy()
    prior = prior.rename(
        columns={
            "warranty_claim_key": "prior_warranty_claim_key",
            "claim_date": "prior_claim_date",
            "failure_code_key": "prior_failure_code_key",
        }
    )
    prior["prior_claim_date"] = as_datetime(prior["prior_claim_date"])
    merged = current.merge(prior, on="truck_key", how="inner", validate="many_to_many")
    merged = merged.loc[
        (merged["prior_claim_date"] < merged["claim_date"])
        & (merged["prior_warranty_claim_key"] != merged["warranty_claim_key"])
    ].copy()
    enriched, join_validation = merge_many_to_one(
        merged,
        failure_codes[["failure_code_key", *FAILURE_FIELDS]].rename(
            columns={"failure_code_key": "prior_failure_code_key"}
        ),
        on="prior_failure_code_key",
        label="prior-claim-to-failure-code",
    )
    output = pd.DataFrame(
        {
            "current_warranty_claim_key": enriched["warranty_claim_key"],
            "prior_warranty_claim_key": enriched["prior_warranty_claim_key"],
            "lineage__truck_key": enriched["truck_key"],
            "prior_claim__claim_date": enriched["prior_claim_date"],
            "prior_failure_code_key": enriched["prior_failure_code_key"],
        }
    )
    for field in FAILURE_FIELDS:
        output[f"prior_failure__{field}"] = enriched[field]
    output = deterministic_sort(
        output,
        ["current_warranty_claim_key", "prior_warranty_claim_key"],
    )
    assert_pair_unique(
        output,
        ["current_warranty_claim_key", "prior_warranty_claim_key"],
        "prior claim",
    )
    if not output.empty:
        current_dates = (
            current.set_index("warranty_claim_key")
            .loc[output["current_warranty_claim_key"], "claim_date"]
            .reset_index(drop=True)
        )
        if bool((as_datetime(output["prior_claim__claim_date"]) >= current_dates).any()):
            raise FeatureMartError("Prior-claim bridge contains a same-day or future claim.")
    return output, history_diagnostics(eligible_claims, output), join_validation
