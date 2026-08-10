"""Offline regression tests for target independence and as-of feature safety."""

from __future__ import annotations

import pandas as pd
import pytest

from warranty_analytics_model.structured_features.builder import build_feature_matrix
from warranty_analytics_model.structured_features.models import StructuredFeatureSettings
from warranty_analytics_model.structured_features.windows import (
    add_claim_dates,
    per_claim_slope,
    safe_divide,
)


def _claims() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "warranty_claim_key": [1, 2, 3, 4],
            "claim__claim_date": pd.to_datetime(
                ["2024-06-15", "2024-07-15", "2024-08-15", "2024-09-15"]
            ),
            "claim__odometer_miles_at_failure": [12000, 20000, 0, 40000],
            "claim__engine_hours_at_failure": [600, 1000, 0, 2000],
            "claim__months_in_service": [12, 20, 0, 40],
            "claim__claim_type": ["A", "B", "A", "C"],
            "truck__build_date": pd.to_datetime(["2023-01-01"] * 4),
            "truck__delivery_date": pd.to_datetime(["2023-02-01"] * 4),
            "truck__in_service_date": pd.to_datetime(["2023-03-01"] * 4),
            "warranty_policy__coverage_months": [36, 36, 0, 48],
            "warranty_policy__coverage_miles": [36000, 36000, 0, 60000],
            "warranty_policy__coverage_engine_hours": [1800, 1800, 0, 3000],
            "warranty_policy__coverage_type": ["full"] * 4,
            "claim_calendar__month_number": [6, 7, 8, 9],
        }
    )


def _telemetry() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, months in {
        1: ["2024-03-01", "2024-04-01", "2024-05-01"],
        2: ["2024-04-01", "2024-05-01", "2024-06-01"],
        3: [],
        4: ["2024-06-01", "2024-07-01", "2024-08-01"],
    }.items():
        for offset, month in enumerate(months, start=1):
            rows.append(
                {
                    "current_warranty_claim_key": key,
                    "telemetry_month_key": key * 100 + offset,
                    "telemetry__month_start_date": pd.Timestamp(month),
                    "telemetry__mileage_month": 100 * offset,
                    "telemetry__total_odometer_miles": 10000 + 100 * offset,
                    "telemetry__engine_hours_month": 10 * offset,
                    "telemetry__idle_hours_month": 2 * offset,
                    "telemetry__avg_engine_temp": 70 + offset,
                    "telemetry__fault_code_count": offset,
                    "telemetry__fuel_efficiency_mpg": 10 + offset,
                    "telemetry__avg_oil_pressure": 40 + offset,
                    "telemetry__route_severity_score": 1 + offset / 10,
                    "telemetry__maintenance_compliance_score": 0.8 + offset / 100,
                }
            )
    return pd.DataFrame(rows)


def _maintenance() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "current_warranty_claim_key": [1, 1, 1, 2],
            "maintenance_event_key": [11, 12, 13, 21],
            "maintenance__maintenance_date": pd.to_datetime(
                ["2024-03-15", "2024-03-14", "2024-05-01", "2024-06-01"]
            ),
            "maintenance__odometer_miles": [9000, 8900, 11000, 18000],
            "maintenance__engine_hours": [450, 440, 550, 900],
            "maintenance__maintenance_type": ["A", "B", "A", "C"],
            "maintenance__scheduled_flag": [True, False, True, True],
            "maintenance__completed_on_time_flag": [True, False, True, False],
            "maintenance__overdue_days": [0, 4, 0, 3],
            "maintenance__maintenance_cost": [100, 200, 150, 400],
        }
    )


def _service() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "current_warranty_claim_key": [1, 1, 2],
            "service_event_key": [101, 102, 201],
            "service__service_date": pd.to_datetime(["2024-04-01", "2024-05-01", "2024-06-01"]),
            "service__odometer_miles": [9500, 11000, 18000],
            "service__engine_hours": [470, 550, 900],
            "service__service_type": ["inspection", "repair", "inspection"],
        }
    )


def _component() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "current_warranty_claim_key": [1, 2],
            "installation_key": [1001, 2001],
            "supplier_key": [99, 98],
            "component_lot_no": ["LOT-A", "LOT-B"],
            "production_batch_id": ["B-A", "B-B"],
            "component_installation__installed_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "component_installation__quality_check_status": ["pass", "pass"],
            "component_installation__rework_flag": [True, False],
            "component_installation__torque_value": [10.0, 12.0],
            "component_installation__inspection_score": [0.9, 0.8],
            "component__component_system": ["brake", "engine"],
            "component__component_category": ["pad", "filter"],
            "component__standard_life_miles": [30000, 40000],
            "component__standard_life_months": [24, 36],
            "component__is_safety_critical": [True, False],
            "component__unit_cost": [100.0, 200.0],
        }
    )


def _prior_claims() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "current_warranty_claim_key": [1, 1, 2],
            "prior_warranty_claim_key": [9001, 9002, 9003],
            "prior_claim__claim_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "prior_failure__failure_code": ["F1", "F2", "F1"],
            "prior_failure__failure_description": ["text", "text", "text"],
            "prior_failure__failure_system": ["brake", "engine", "brake"],
            "prior_failure__failure_category": ["wear", "electrical", "wear"],
            "prior_failure__severity_level": ["high", "low", "medium"],
            "prior_failure__safety_related_flag": [True, False, False],
            "prior_failure__recall_related_flag": [False, False, True],
        }
    )


def _frames() -> dict[str, pd.DataFrame]:
    return {
        "claim_snapshot": _claims().assign(target__high_cost_claim_flag=[0, 1, 0, 1]),
        "telemetry_history": _telemetry(),
        "maintenance_history": _maintenance(),
        "service_history": _service(),
        "component_installation_history": _component(),
        "prior_claim_history": _prior_claims(),
        "repair_history_index": pd.DataFrame(
            {"current_warranty_claim_key": [1], "repair_line_key": [1]}
        ),
    }


def _build(frames: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    assignments = pd.DataFrame(
        {
            "warranty_claim_key": [1, 2, 3, 4],
            "split": ["TRAIN", "VALIDATION", "TEST", "TEST"],
        }
    )
    result = build_feature_matrix(frames or _frames(), assignments, StructuredFeatureSettings())
    return result.frame


def test_target_changes_do_not_change_structured_features() -> None:
    first = _build()
    changed = _frames()
    changed["claim_snapshot"]["target__high_cost_claim_flag"] = [1, 0, 1, 0]
    second = _build(changed)
    pd.testing.assert_frame_equal(first, second)


def test_history_order_is_deterministic() -> None:
    first = _build()
    shuffled = _frames()
    for name in (
        "telemetry_history",
        "maintenance_history",
        "service_history",
        "component_installation_history",
        "prior_claim_history",
    ):
        shuffled[name] = shuffled[name].sample(frac=1, random_state=42).reset_index(drop=True)
    second = _build(shuffled)
    pd.testing.assert_frame_equal(first, second)


def test_no_history_claim_is_retained_with_zero_counts_and_null_measurements() -> None:
    result = _build()
    row = result.loc[result["warranty_claim_key"] == 3].iloc[0]
    assert row["telemetry__3m__months_observed"] == 0
    assert row["maintenance__3m__event_count"] == 0
    assert not bool(row["history__has_telemetry"])
    assert pd.isna(row["telemetry__3m__mileage_month__mean"])


def test_window_boundary_and_claim_month_rules_are_enforced() -> None:
    claims = _claims().iloc[[0]].copy()
    safe = pd.DataFrame(
        {
            "current_warranty_claim_key": [1],
            "telemetry__month_start_date": [pd.Timestamp("2024-05-01")],
        }
    )
    prepared = add_claim_dates(safe, claims, "telemetry__month_start_date", telemetry=True)
    assert len(prepared) == 1
    unsafe = safe.assign(telemetry__month_start_date=pd.Timestamp("2024-06-01"))
    with pytest.raises(ValueError, match="Unsafe same-day/future"):
        add_claim_dates(unsafe, claims, "telemetry__month_start_date", telemetry=True)


def test_ratio_and_trend_safety() -> None:
    ratio = safe_divide(pd.Series([1.0, 2.0]), pd.Series([0.0, 2.0]))
    assert pd.isna(ratio.iloc[0])
    assert ratio.iloc[1] == 1.0
    trend = pd.DataFrame(
        {
            "current_warranty_claim_key": [1, 1, 1],
            "_event_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-04-01"]),
            "value": [1.0, 2.0, 4.0],
        }
    )
    assert per_claim_slope(trend, "value").loc[1] > 0


def test_restricted_identifiers_repair_and_text_never_become_model_features() -> None:
    result = _build()
    model_names = [
        column
        for column in result.columns
        if column
        not in {
            "warranty_claim_key",
            "split",
            "claim__claim_date",
            "truck__build_date",
            "truck__delivery_date",
            "truck__in_service_date",
        }
    ]
    forbidden = (
        "supplier_key",
        "component_lot_no",
        "production_batch_id",
        "repair",
        "failure_description",
    )
    assert not any(any(token in name for token in forbidden) for name in model_names)
    assert "target__high_cost_claim_flag" not in result.columns
