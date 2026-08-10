"""Database-independent temporal bridge and snapshot tests."""

from __future__ import annotations

import pandas as pd
import pytest

from warranty_analytics_model.feature_mart.component_history import (
    INSTALLATION_FIELDS,
    build_component_installation_history,
)
from warranty_analytics_model.feature_mart.maintenance_history import (
    MAINTENANCE_FIELDS,
    build_maintenance_history,
)
from warranty_analytics_model.feature_mart.models import FeatureMartError
from warranty_analytics_model.feature_mart.prior_claim_history import (
    FAILURE_FIELDS,
    build_prior_claim_history,
)
from warranty_analytics_model.feature_mart.repair_history import build_repair_history_index
from warranty_analytics_model.feature_mart.service_history import build_service_history
from warranty_analytics_model.feature_mart.telemetry_history import (
    TELEMETRY_FIELDS,
    build_telemetry_history,
)


def _claims() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "warranty_claim_key": [1, 2],
            "claim_date": pd.to_datetime(["2026-03-15", "2026-05-15"]),
            "high_cost_claim_flag": [0, 1],
            "truck_key": [10, 10],
            "service_event_key": [100, 101],
            "service_center_key": [500, 500],
            "failure_code_key": [1, 2],
            "repair_end_date": pd.to_datetime(["2026-04-01", "2026-05-20"]),
            "claim_date_source": pd.to_datetime(["2026-03-15", "2026-05-15"]),
            "odometer_miles_at_failure": [1000, 2000],
            "engine_hours_at_failure": [100, 200],
            "months_in_service": [10, 12],
            "warranty_coverage_status": ["covered", "covered"],
            "claim_type": ["mechanical", "electrical"],
        }
    )


def _history_sources() -> dict[str, pd.DataFrame]:
    telemetry = pd.DataFrame(
        {
            "telemetry_month_key": [1, 2, 3],
            "truck_key": [10, 10, 10],
            "month_start_date": pd.to_datetime(["2026-01-01", "2026-03-01", "2026-05-01"]),
            **{field: [1.0, 2.0, 3.0] for field in TELEMETRY_FIELDS},
        }
    )
    maintenance = pd.DataFrame(
        {
            "maintenance_event_key": [10, 11, 12],
            "truck_key": [10, 10, 10],
            "service_center_key": [500, 500, 500],
            "maintenance_date": pd.to_datetime(["2026-01-01", "2026-03-15", "2026-06-01"]),
            **{field: [1.0, 2.0, 3.0] for field in MAINTENANCE_FIELDS},
        }
    )
    service = pd.DataFrame(
        {
            "service_event_key": [90, 100, 101, 102],
            "truck_key": [10, 10, 10, 10],
            "service_center_key": [500, 500, 500, 500],
            "service_date": pd.to_datetime(
                ["2026-01-01", "2026-03-15", "2026-05-01", "2026-01-02"]
            ),
            "odometer_miles": [100, 110, 120, 130],
            "engine_hours": [10, 11, 12, 13],
            "service_type": ["repair", "repair", "repair", "inspection"],
        }
    )
    components = pd.DataFrame(
        {
            "component_key": [900],
            "component_system": ["engine"],
            "component_category": ["fuel"],
            "standard_life_miles": [100000],
            "standard_life_months": [60],
            "is_safety_critical": [True],
            "unit_cost": [100.0],
        }
    )
    installations = pd.DataFrame(
        {
            "installation_key": [20, 21, 22],
            "truck_key": [10, 10, 10],
            "component_key": [900, 900, 900],
            "supplier_key": [700, 700, 700],
            "component_lot_no": ["lot-a", "lot-b", "lot-c"],
            "installed_date": pd.to_datetime(["2026-01-01", "2026-03-15", "2026-06-01"]),
            "production_batch_id": ["batch-a", "batch-b", "batch-c"],
            **{field: [1.0, 2.0, 3.0] for field in INSTALLATION_FIELDS},
        }
    )
    failure_codes = pd.DataFrame(
        {
            "failure_code_key": [1, 2],
            **{field: [f"{field}-1", f"{field}-2"] for field in FAILURE_FIELDS},
        }
    )
    repair_lines = pd.DataFrame(
        {
            "repair_line_key": [30, 31],
            "warranty_claim_key": [1, 2],
            "service_event_key": [90, 101],
            "component_key": [900, 900],
        }
    )
    return {
        "telemetry": telemetry,
        "maintenance": maintenance,
        "service": service,
        "components": components,
        "installations": installations,
        "failure_codes": failure_codes,
        "repair_lines": repair_lines,
    }


def test_completed_month_telemetry_rule_and_zero_history_claim() -> None:
    claims = _claims()
    bridge, coverage = build_telemetry_history(claims, _history_sources()["telemetry"])
    assert set(bridge["telemetry_month_key"]) == {1, 2}
    assert coverage["claims_with_history"] == 2
    assert coverage["claims_without_history"] == 0


def test_maintenance_same_day_and_future_are_excluded() -> None:
    bridge, _ = build_maintenance_history(_claims(), _history_sources()["maintenance"])
    assert set(bridge["maintenance_event_key"]) == {10, 11}


def test_service_same_day_and_current_event_are_excluded() -> None:
    bridge, _ = build_service_history(_claims(), _history_sources()["service"])
    claim_one = bridge.loc[bridge["current_warranty_claim_key"] == 1]
    claim_two = bridge.loc[bridge["current_warranty_claim_key"] == 2]
    assert 100 not in set(claim_one["service_event_key"])
    assert 101 not in set(claim_two["service_event_key"])
    assert set(bridge["service_event_key"]) == {90, 100, 102}


def test_component_history_does_not_require_current_causal_component_key() -> None:
    sources = _history_sources()
    bridge, _, _ = build_component_installation_history(
        _claims(), sources["installations"], sources["components"]
    )
    assert set(bridge["installation_key"]) == {20, 21}
    assert "causal_component_key" not in bridge.columns


def test_prior_claim_taxonomy_excludes_current_and_same_day_claims() -> None:
    sources = _history_sources()
    bridge, _, _ = build_prior_claim_history(_claims(), _claims(), sources["failure_codes"])
    assert set(bridge["prior_warranty_claim_key"]) == {1}
    assert set(bridge["current_warranty_claim_key"]) == {2}
    assert all(
        column.startswith("prior_")
        or column.startswith("current_")
        or column.startswith("lineage__")
        for column in bridge.columns
    )


def test_repair_index_requires_completed_prior_claim_and_excludes_current_line() -> None:
    sources = _history_sources()
    bridge, _ = build_repair_history_index(_claims(), _claims(), sources["repair_lines"])
    assert set(bridge["repair_line_key"]) == {30}
    assert not ((bridge["current_warranty_claim_key"] == bridge["prior_warranty_claim_key"]).any())


def test_duplicate_bridge_pairs_fail() -> None:
    telemetry = _history_sources()["telemetry"]
    telemetry = pd.concat([telemetry, telemetry.iloc[[0]]], ignore_index=True)
    with pytest.raises(FeatureMartError, match="duplicate key"):
        build_telemetry_history(_claims(), telemetry)


def test_duplicate_component_dimension_key_fails() -> None:
    sources = _history_sources()
    components = pd.concat([sources["components"], sources["components"]], ignore_index=True)
    with pytest.raises(FeatureMartError, match="duplicate key"):
        build_component_installation_history(_claims(), sources["installations"], components)
