"""Fictional tests for group exposure and evaluation cohorts."""

from __future__ import annotations

import pandas as pd

from warranty_analytics_model.splits.cohorts import (
    build_evaluation_cohorts,
    fingerprint_overlap_summary,
)
from warranty_analytics_model.splits.group_exposure import (
    build_group_exposure,
    summarize_group_overlap,
)


def _assignments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "warranty_claim_key": [1, 2, 3, 4, 5, 6],
            "claim_date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-05",
                    "2025-01-06",
                ]
            ),
            "split": ["TRAIN", "TRAIN", "TRAIN", "VALIDATION", "TEST", "TEST"],
        }
    )


def _groups() -> pd.DataFrame:
    rows = [
        (1, "truck", "truck-a"),
        (2, "truck", "truck-a"),
        (3, "truck", "truck-b"),
        (4, "truck", "truck-b"),
        (5, "truck", "truck-c"),
        (6, "truck", "truck-b"),
        (1, "production_batch", "batch-a"),
        (4, "production_batch", "batch-b"),
        (5, "production_batch", "batch-c"),
        (6, "production_batch", "batch-b"),
        (1, "service_center", "center-a"),
        (4, "service_center", "center-b"),
        (5, "service_center", "center-c"),
        (6, "service_center", "center-b"),
        (1, "historical_supplier", "supplier-a"),
        (4, "historical_supplier", "supplier-a"),
        (4, "historical_supplier", "supplier-b"),
        (5, "historical_supplier", "supplier-b"),
        (5, "historical_supplier", "supplier-c"),
        (6, "historical_supplier", "supplier-d"),
        (1, "historical_component_lot", "lot-a"),
        (4, "historical_component_lot", "lot-b"),
        (5, "historical_component_lot", "lot-c"),
        (6, "historical_component_lot", "lot-d"),
        (1, "safe_scenario_fingerprint", "fingerprint-a"),
        (4, "safe_scenario_fingerprint", "fingerprint-a"),
        (5, "safe_scenario_fingerprint", "fingerprint-b"),
        (6, "safe_scenario_fingerprint", "fingerprint-a"),
    ]
    return pd.DataFrame(
        {
            "warranty_claim_key": [row[0] for row in rows],
            "group_type": [row[1] for row in rows],
            "group_value_hash": [row[2] for row in rows],
            "group_value": [row[2] for row in rows],
            "source": ["fictional" for _ in rows],
            "is_model_feature": [False for _ in rows],
        }
    )


def test_group_exposure_seen_and_unseen_flags() -> None:
    exposure = build_group_exposure(_assignments(), _groups())
    validation_truck = exposure.loc[
        (exposure["warranty_claim_key"] == 4) & (exposure["group_type"] == "truck")
    ].iloc[0]
    test_truck = exposure.loc[
        (exposure["warranty_claim_key"] == 5) & (exposure["group_type"] == "truck")
    ].iloc[0]

    assert bool(validation_truck["seen_in_train"])
    assert not bool(validation_truck["unseen_in_train"])
    assert not bool(test_truck["seen_in_development"])
    assert bool(test_truck["unseen_in_development"])


def test_multi_valued_supplier_any_and_all_unseen_flags() -> None:
    cohorts = build_evaluation_cohorts(_assignments(), _groups())
    validation = cohorts.loc[cohorts["warranty_claim_key"] == 4].iloc[0]
    test = cohorts.loc[cohorts["warranty_claim_key"] == 6].iloc[0]

    assert int(validation["eval__historical_supplier_count"]) == 2
    assert int(validation["eval__historical_supplier_seen_count"]) == 1
    assert int(validation["eval__historical_supplier_unseen_count"]) == 1
    assert bool(validation["eval__any_historical_supplier_unseen"])
    assert not bool(validation["eval__all_historical_suppliers_unseen"])
    assert bool(test["eval__all_historical_suppliers_unseen"])


def test_fingerprint_overlap_is_warning_and_clean_cohort_is_secondary() -> None:
    assignments = _assignments()
    exposure = build_group_exposure(assignments, _groups())
    cohorts = build_evaluation_cohorts(assignments, _groups())
    result = fingerprint_overlap_summary(assignments, exposure)

    assert result["overlap_severity"] == "WARNING"
    assert result["validation_affected_claims"] == 1
    assert result["test_affected_claims_seen_in_development"] == 1
    assert bool(cohorts.loc[cohorts["warranty_claim_key"] == 5, "eval__fingerprint_clean"].iloc[0])
    assert not bool(
        cohorts.loc[cohorts["warranty_claim_key"] == 6, "eval__fingerprint_clean"].iloc[0]
    )


def test_group_summary_is_aggregate_only() -> None:
    exposure = build_group_exposure(_assignments(), _groups())
    summary = summarize_group_overlap(exposure)

    assert "truck" in summary["group_types"]
    assert summary["group_types"]["truck"]["test_groups_unseen_in_development"] == 1
    assert "warranty_claim_key" not in str(summary)
