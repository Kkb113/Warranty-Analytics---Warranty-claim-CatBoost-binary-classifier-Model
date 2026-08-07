"""Warranty-claim target distribution and target-generation diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from .column_profile import is_sensitive_column

_DEFAULT_GROUP_COLUMNS = (
    "claim_month",
    "claim_year",
    "model_name",
    "truck_model_key",
    "model_year",
    "manufacturing_plant",
    "assembly_line",
    "supplier_key",
    "component_system",
    "component_category",
    "service_center_key",
    "region",
    "climate_zone",
    "terrain_type",
    "claim_type",
    "failure_system",
    "failure_category",
)


def _safe_group_value(value: object, column: str) -> object:
    return "<redacted>" if is_sensitive_column(column) else value


def group_target_rates(
    frame: pd.DataFrame, column: str, target_column: str
) -> list[dict[str, object]]:
    """Return counts and target rates for one grouping column."""

    if column not in frame or target_column not in frame:
        return []
    subset = frame[[column, target_column]].copy()
    subset[target_column] = pd.to_numeric(subset[target_column], errors="coerce")
    subset = subset[subset[target_column].isin([0, 1])]
    if subset.empty:
        return []
    grouped = subset.groupby(column, dropna=False, observed=True)[target_column].agg(
        records="size", positives="sum", target_rate="mean"
    )
    return [
        {
            "group": _safe_group_value(value, column),
            "records": int(row.records),
            "positive_count": int(row.positives),
            "negative_count": int(row.records - row.positives),
            "positive_rate_percentage": round(float(row.target_rate) * 100, 6),
        }
        for value, row in grouped.sort_values(["records", "target_rate"], ascending=False)
        .head(200)
        .iterrows()
    ]


def audit_target_generation(
    claims: pd.DataFrame,
    target_column: str = "high_cost_claim_flag",
    cost_columns: Iterable[str] = (
        "total_claim_cost",
        "labor_cost",
        "parts_cost",
        "diagnostic_cost",
        "towing_cost",
        "other_cost",
        "approved_amount",
        "rejected_amount",
        "customer_paid_amount",
    ),
) -> dict[str, Any]:
    """Test threshold separability descriptively; never call it a business rule."""

    if target_column not in claims:
        return {"available": False, "reason": "target column not present"}
    target = pd.to_numeric(claims[target_column], errors="coerce")
    valid_target = target.isin([0, 1])
    result: dict[str, Any] = {
        "available": True,
        "target_column": target_column,
        "phrase_if_deterministic": "Empirical synthetic target-generation rule suspected",
        "cost_relationships": {},
    }
    for column in cost_columns:
        if column not in claims:
            continue
        values = pd.to_numeric(claims[column], errors="coerce")
        frame = pd.DataFrame({"value": values, "target": target})
        frame = frame[valid_target & values.notna()]
        positives = frame.loc[frame.target == 1, "value"]
        negatives = frame.loc[frame.target == 0, "value"]
        item: dict[str, Any] = {
            "records": int(len(frame)),
            "minimum_positive": float(positives.min()) if not positives.empty else None,
            "maximum_negative": float(negatives.max()) if not negatives.empty else None,
            "distribution_overlap": None,
            "candidate_threshold": None,
            "exceptions": None,
            "deterministic_separation": False,
        }
        if not positives.empty and not negatives.empty:
            minimum_positive = float(positives.min())
            maximum_negative = float(negatives.max())
            item["distribution_overlap"] = bool(minimum_positive <= maximum_negative)
            if minimum_positive > maximum_negative:
                threshold = (minimum_positive + maximum_negative) / 2
                predicted = (values >= threshold).astype(int)
                valid = valid_target & values.notna()
                exceptions = int((predicted[valid] != target[valid]).sum())
                item.update(
                    {
                        "candidate_threshold": threshold,
                        "exceptions": exceptions,
                        "deterministic_separation": exceptions == 0,
                    }
                )
        result["cost_relationships"][column] = item
    total = result["cost_relationships"].get("total_claim_cost")
    result["total_claim_cost_deterministic"] = bool(
        isinstance(total, dict) and total.get("deterministic_separation")
    )
    result["interpretation"] = (
        "Empirical synthetic target-generation rule suspected"
        if result["total_claim_cost_deterministic"]
        else "No exact single-threshold separation established from available records"
    )
    return result


def grouped_threshold_audit(
    claims: pd.DataFrame,
    *,
    target_column: str = "high_cost_claim_flag",
    group_columns: Iterable[str] = (
        "claim_type",
        "model_name",
        "truck_model_key",
        "warranty_policy_key",
        "component_category",
        "region",
    ),
) -> dict[str, list[dict[str, object]]]:
    """Check whether cost separation appears to vary across descriptive groups."""

    if target_column not in claims or "total_claim_cost" not in claims:
        return {}
    target = pd.to_numeric(claims[target_column], errors="coerce")
    costs = pd.to_numeric(claims["total_claim_cost"], errors="coerce")
    output: dict[str, list[dict[str, object]]] = {}
    for column in group_columns:
        if column not in claims:
            continue
        rows: list[dict[str, object]] = []
        for value, group in claims.groupby(column, dropna=False, observed=True):
            valid = pd.DataFrame(
                {"target": target.loc[group.index], "cost": costs.loc[group.index]}
            ).dropna()
            valid = valid[valid["target"].isin([0, 1])]
            positive = valid.loc[valid.target == 1, "cost"]
            negative = valid.loc[valid.target == 0, "cost"]
            if positive.empty or negative.empty:
                continue
            minimum_positive = float(positive.min())
            maximum_negative = float(negative.max())
            rows.append(
                {
                    "group": _safe_group_value(value, column),
                    "records": int(len(valid)),
                    "minimum_positive": minimum_positive,
                    "maximum_negative": maximum_negative,
                    "distribution_overlap": bool(minimum_positive <= maximum_negative),
                    "separates_exactly": bool(minimum_positive > maximum_negative),
                }
            )
        output[column] = rows
    return output


def profile_target(
    claims: pd.DataFrame,
    target_column: str = "high_cost_claim_flag",
    *,
    group_columns: Iterable[str] = _DEFAULT_GROUP_COLUMNS,
) -> dict[str, Any]:
    """Build target counts, time/group distributions, and target audit."""

    if target_column not in claims:
        return {
            "available": False,
            "target_column": target_column,
            "claims": 0,
            "error": "target column not present",
        }
    target = pd.to_numeric(claims[target_column], errors="coerce")
    valid = target.isin([0, 1])
    positive = int((target == 1).sum())
    negative = int((target == 0).sum())
    null_count = int(target.isna().sum())
    invalid_count = int((~valid & target.notna()).sum())
    output: dict[str, Any] = {
        "available": True,
        "target_column": target_column,
        "claims": int(len(claims)),
        "usable_claims": int(valid.sum()),
        "positive_claims": positive,
        "negative_claims": negative,
        "positive_percentage": round(positive / len(claims) * 100, 6) if len(claims) else 0.0,
        "negative_percentage": round(negative / len(claims) * 100, 6) if len(claims) else 0.0,
        "null_target_count": null_count,
        "invalid_target_count": invalid_count,
        "class_balance": "unknown",
        "by_group": {},
    }
    if len(claims):
        output["class_balance"] = (
            "balanced" if 40 <= positive / len(claims) * 100 <= 60 else "imbalanced"
        )
    if "claim_date" in claims:
        dates = pd.to_datetime(claims["claim_date"], errors="coerce")
        claims_with_time = claims.assign(
            _claim_month=dates.dt.to_period("M").astype("string"), _claim_year=dates.dt.year
        )
        output["by_month"] = group_target_rates(claims_with_time, "_claim_month", target_column)
        output["by_year"] = group_target_rates(claims_with_time, "_claim_year", target_column)
    for column in group_columns:
        if column in claims:
            output["by_group"][column] = group_target_rates(claims, column, target_column)
    output["target_generation_audit"] = audit_target_generation(claims, target_column)
    output["group_threshold_audit"] = grouped_threshold_audit(claims, target_column=target_column)
    return output
