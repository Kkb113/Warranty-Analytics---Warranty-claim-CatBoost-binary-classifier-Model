"""Corrective-hardening tests for Phase 3 joins, as-of matching, and telemetry."""

from __future__ import annotations

import pandas as pd
import pytest

from warranty_analytics_model.profiling.installation_audit import (
    match_component_installations_asof,
)
from warranty_analytics_model.profiling.runner import _claim_context, profile_dataframes
from warranty_analytics_model.profiling.temporal_quality import telemetry_quality


def _claim_context_fixture() -> dict[str, pd.DataFrame]:
    return {
        "dbo.fact_warranty_claim": pd.DataFrame(
            {
                "warranty_claim_key": [1, 2],
                "truck_key": [1, 1],
                "causal_component_key": [10, 99],
                "high_cost_claim_flag": [0, 1],
            }
        ),
        "dbo.dim_component": pd.DataFrame(
            {
                "component_key": [10],
                "component_system": ["Engine"],
                "component_category": ["Mechanical"],
                "supplier_key": [7],
                "is_safety_critical": [True],
            }
        ),
        "dbo.dim_supplier": pd.DataFrame(
            {
                "supplier_key": [7],
                "supplier_region": ["North"],
                "supplier_tier": ["Tier 1"],
                "quality_rating": [94.5],
            }
        ),
    }


def test_claim_component_supplier_join_uses_explicit_left_and_right_keys() -> None:
    """Causal component keys attach component and supplier context once per claim."""

    result = _claim_context(_claim_context_fixture())

    assert len(result) == 2
    assert result.loc[0, "component_system"] == "Engine"
    assert result.loc[0, "component_category"] == "Mechanical"
    assert result.loc[0, "supplier_region"] == "North"
    assert result.loc[0, "supplier_tier"] == "Tier 1"
    assert result.loc[0, "quality_rating"] == pytest.approx(94.5)
    assert pd.isna(result.loc[1, "component_system"])
    assert pd.isna(result.loc[1, "supplier_region"])


def test_component_dimension_duplicate_key_fails_many_to_one_validation() -> None:
    """A duplicate component dimension key cannot be silently collapsed."""

    frames = _claim_context_fixture()
    frames["dbo.dim_component"] = pd.concat(
        [frames["dbo.dim_component"], frames["dbo.dim_component"]], ignore_index=True
    )

    with pytest.raises(ValueError, match="many-to-one"):
        _claim_context(frames)


def test_phase3_task_groups_scope_work() -> None:
    """The shared engine executes only the requested diagnostic task group."""

    frames = _claim_context_fixture()
    frames["dbo.fact_telemetry_monthly"] = pd.DataFrame(
        {
            "truck_key": [1, 1],
            "month_start_date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "total_odometer_miles": [100, 110],
            "engine_hours_month": [180, 145],
            "idle_hours_month": [40, 30],
        }
    )

    data_profile = profile_dataframes(frames, task_groups=("data_profile",))
    synthetic = profile_dataframes(frames, task_groups=("synthetic_audit",))
    quality = profile_dataframes(frames, task_groups=("data_quality",))

    assert data_profile["task_groups"] == ["data_profile"]
    assert data_profile["table_profiles"]
    assert data_profile["synthetic_data_audit"] == {}
    assert synthetic["task_groups"] == ["synthetic_audit"]
    assert synthetic["table_profiles"] == []
    assert synthetic["synthetic_data_audit"]["target_generation"]
    assert quality["task_groups"] == ["data_quality"]
    assert quality["data_quality"]["telemetry"]["records"] == 2
    assert quality["target_profile"] == {}


def _claim(as_of: str = "2026-05-01", *, include_failure: bool = True) -> pd.DataFrame:
    values: dict[str, list[object]] = {
        "warranty_claim_key": [1],
        "truck_key": [1],
        "causal_component_key": [10],
        "claim_date": [pd.Timestamp("2026-04-01")],
        "high_cost_claim_flag": [0],
    }
    if include_failure:
        values["failure_date"] = [pd.Timestamp(as_of)]
    return pd.DataFrame(values)


def _installations(*dates_and_lots: tuple[str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "installation_key": list(range(1, len(dates_and_lots) + 1)),
            "truck_key": [1] * len(dates_and_lots),
            "component_key": [10] * len(dates_and_lots),
            "supplier_key": [7] * len(dates_and_lots),
            "component_lot_no": [item[1] for item in dates_and_lots],
            "production_batch_id": [item[2] for item in dates_and_lots],
            "installed_date": pd.to_datetime([item[0] for item in dates_and_lots]),
        }
    )


def test_asof_matching_selects_one_valid_installation() -> None:
    matched, diagnostics = match_component_installations_asof(
        _claim(), _installations(("2026-01-01", "LOT-1", "B-1"))
    )

    assert diagnostics["matched_as_of_installation"] == 1
    assert matched.loc[0, "installation_component_lot_no"] == "LOT-1"


def test_asof_matching_excludes_future_installation() -> None:
    matched, diagnostics = match_component_installations_asof(
        _claim(), _installations(("2026-06-01", "LOT-FUTURE", "B-FUTURE"))
    )

    assert diagnostics["future_installations_excluded"] == 1
    assert diagnostics["unmatched_as_of_installation"] == 1
    assert pd.isna(matched.loc[0, "installation_component_lot_no"])


def test_asof_matching_selects_latest_historical_replacement() -> None:
    matched, diagnostics = match_component_installations_asof(
        _claim(),
        _installations(
            ("2025-01-01", "LOT-OLD", "B-OLD"),
            ("2026-03-01", "LOT-NEW", "B-NEW"),
        ),
    )

    assert diagnostics["claims_with_multiple_historical_installations"] == 1
    assert matched.loc[0, "installation_component_lot_no"] == "LOT-NEW"


def test_asof_matching_keeps_historical_installation_when_replacement_is_future() -> None:
    matched, diagnostics = match_component_installations_asof(
        _claim(),
        _installations(
            ("2025-01-01", "LOT-OLD", "B-OLD"),
            ("2026-06-01", "LOT-FUTURE", "B-FUTURE"),
        ),
    )

    assert diagnostics["future_installations_excluded"] == 1
    assert matched.loc[0, "installation_component_lot_no"] == "LOT-OLD"


def test_asof_matching_uses_claim_date_fallback() -> None:
    matched, diagnostics = match_component_installations_asof(
        _claim("2026-04-01", include_failure=False),
        _installations(("2026-03-01", "LOT-FALLBACK", "B-FALLBACK")),
    )

    assert diagnostics["matched_as_of_installation"] == 1
    assert matched.loc[0, "installation_component_lot_no"] == "LOT-FALLBACK"


def test_asof_matching_marks_conflicting_same_date_rows_ambiguous_without_multiplying() -> None:
    matched, diagnostics = match_component_installations_asof(
        _claim(),
        _installations(
            ("2026-03-01", "LOT-A", "B-A"),
            ("2026-03-01", "LOT-B", "B-B"),
        ),
    )

    assert len(matched) == 1
    assert diagnostics["ambiguous_as_of_installation"] == 1
    assert diagnostics["matched_as_of_installation"] == 0
    assert matched.loc[0, "installation_match_status"] == "ambiguous"
    assert pd.isna(matched.loc[0, "installation_component_lot_no"])


def test_asof_matching_retains_claim_when_no_historical_installation_exists() -> None:
    claim = _claim().assign(causal_component_key=[99])
    matched, diagnostics = match_component_installations_asof(
        claim, _installations(("2025-01-01", "OTHER-LOT", "OTHER-BATCH"))
    )

    assert len(matched) == 1
    assert diagnostics["unmatched_as_of_installation"] == 1
    assert matched.loc[0, "installation_match_status"] == "unmatched"


def _telemetry(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "truck_key": [1] * len(rows),
            "month_start_date": pd.date_range("2026-01-01", periods=len(rows), freq="MS"),
            "total_odometer_miles": [row[0] for row in rows],
            "engine_hours_month": [row[1] for row in rows],
            "idle_hours_month": [row[2] for row in rows],
        }
    )


def test_monthly_engine_hours_are_not_treated_as_cumulative() -> None:
    result = telemetry_quality(_telemetry([(10000, 180, 40), (12000, 145, 30), (13000, 210, 50)]))

    assert "engine_hours_decrease_count" not in result
    assert not any(issue["issue"] == "engine_hours_decreases" for issue in result["issues"])


def test_telemetry_invalid_negative_and_logical_hour_rules() -> None:
    negative = telemetry_quality(_telemetry([(10000, 180, 40), (12000, -10, 20)]))
    over_idle = telemetry_quality(_telemetry([(10000, 100, 120)]))
    valid = telemetry_quality(_telemetry([(10000, 100, 40)]))

    assert any(
        issue["issue"] == "negative_measurement" and issue["field"] == "engine_hours_month"
        for issue in negative["issues"]
    )
    assert over_idle["idle_hours_over_engine_hours_count"] == 1
    assert any(issue["issue"] == "idle_hours_exceed_engine_hours" for issue in over_idle["issues"])
    assert valid["idle_hours_over_engine_hours_count"] == 0


def test_total_odometer_decrease_remains_a_sequence_issue() -> None:
    result = telemetry_quality(_telemetry([(10000, 100, 40), (12000, 100, 40), (11000, 100, 40)]))

    assert result["odometer_decrease_count"] == 1
    assert any(issue["issue"] == "odometer_decreases" for issue in result["issues"])
