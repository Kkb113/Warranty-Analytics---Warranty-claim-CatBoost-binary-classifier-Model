"""Combined synthetic-data, leakage, duplicate, and purity diagnostics."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .association import association_table, missingness_by_target
from .duplicate_audit import audit_duplicates
from .identifier_audit import audit_identifiers
from .target_profile import audit_target_generation
from .text_audit import audit_text

POST_OUTCOME_COLUMNS = (
    "total_claim_cost",
    "labor_cost",
    "parts_cost",
    "diagnostic_cost",
    "towing_cost",
    "other_cost",
    "approved_amount",
    "rejected_amount",
    "customer_paid_amount",
    "days_to_repair",
    "claim_status",
    "repeat_claim_flag",
    "potential_recall_flag",
    "root_cause_category",
)


def group_purity(
    frame: pd.DataFrame,
    group_columns: Iterable[str],
    target_column: str = "high_cost_claim_flag",
    *,
    minimum_support: int = 5,
) -> list[dict[str, object]]:
    """Measure target purity with support shown for every returned group."""

    if target_column not in frame:
        return []
    target = pd.to_numeric(frame[target_column], errors="coerce")
    output: list[dict[str, object]] = []
    for column in group_columns:
        if column not in frame:
            continue
        data = pd.DataFrame({"group": frame[column], "target": target}).dropna()
        data = data[data.target.isin([0, 1])]
        grouped = data.groupby("group", observed=True)["target"].agg(
            records="size", positives="sum", distinct="nunique"
        )
        for value, row in (
            grouped[grouped.records >= minimum_support]
            .sort_values("records", ascending=False)
            .head(100)
            .iterrows()
        ):
            records = int(row.records)
            positives = int(row.positives)
            rate = positives / records if records else 0.0
            output.append(
                {
                    "field": column,
                    "group_hash": _safe_hash(value),
                    "records": records,
                    "positive_count": positives,
                    "negative_count": records - positives,
                    "positive_rate_percentage": round(rate * 100, 6),
                    "target_pure": rate in (0.0, 1.0),
                    "nearly_pure": rate <= 0.05 or rate >= 0.95,
                    "meaningful_support": records >= minimum_support,
                }
            )
    return output


def leakage_diagnostics(
    claims: pd.DataFrame,
    *,
    target_column: str = "high_cost_claim_flag",
    leakage_columns: Iterable[str] = POST_OUTCOME_COLUMNS,
) -> dict[str, object]:
    """Quantify post-outcome relationships as diagnostic evidence only."""

    present = [column for column in leakage_columns if column in claims]
    associations = association_table(
        claims, target_column, columns=present, leakage_columns=present
    )
    evidence: list[dict[str, object]] = []
    by_field = {str(item["field"]): item for item in associations}
    for column in present:
        item = by_field.get(column, {})
        value = item.get("association_value")
        suspected_type = "post_outcome_field"
        if column in {
            "total_claim_cost",
            "labor_cost",
            "parts_cost",
            "diagnostic_cost",
            "towing_cost",
            "other_cost",
            "approved_amount",
            "rejected_amount",
            "customer_paid_amount",
        }:
            suspected_type = "post_outcome_cost_or_amount"
        evidence.append(
            {
                "field": column,
                "relationship_to_target": item.get(
                    "association_measure", "descriptive relationship unavailable"
                ),
                "suspected_leakage_type": suspected_type,
                "evidence": {
                    "association": value,
                    "target_rate_range": item.get("target_rate_range"),
                },
                "severity": "WARNING",
                "recommended_phase4_action": "Exclude from prediction-time feature set unless availability is explicitly proven.",
            }
        )
    return {
        "fields": evidence,
        "association_summary": associations,
        "missingness_by_target": missingness_by_target(claims, target_column, present),
    }


def run_synthetic_audit(
    frames: dict[str, pd.DataFrame],
    claims: pd.DataFrame,
    *,
    target_column: str = "high_cost_claim_flag",
    enable_text: bool = True,
    enable_identifiers: bool = True,
) -> dict[str, object]:
    """Run all synthetic-data audit components and return aggregate evidence."""

    purity_columns = (
        "production_batch_id",
        "component_lot_no",
        "supplier_key",
        "causal_component_key",
        "service_center_key",
        "manufacturing_plant",
        "assembly_line",
        "truck_model_key",
    )
    return {
        "target_generation": audit_target_generation(claims, target_column),
        "identifier_audit": audit_identifiers(claims, target_column)
        if enable_identifiers
        else {"fields": [], "flags": []},
        "group_purity": group_purity(claims, purity_columns, target_column),
        "duplicate_audit": audit_duplicates(frames, target_column),
        "text_audit": audit_text(frames, target_column)
        if enable_text
        else {"fields": [], "flags": []},
        "leakage_diagnostics": leakage_diagnostics(claims, target_column=target_column),
    }


def _safe_hash(value: object) -> str:
    import hashlib

    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]
