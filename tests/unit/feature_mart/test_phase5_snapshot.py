"""Direct claim snapshot and cardinality tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from warranty_analytics_model.feature_mart.direct_snapshot import build_direct_snapshot
from warranty_analytics_model.feature_mart.mart_contract import load_mart_contract
from warranty_analytics_model.feature_mart.models import FeatureMartError

from .test_phase5_bridges import _claims

ROOT = Path(__file__).resolve().parents[3]


def _direct_frames(policy_start: str = "2020-01-01") -> dict[str, pd.DataFrame]:
    claims = _claims()
    truck = pd.DataFrame(
        {
            "truck_key": [10],
            "truck_model_key": [1000],
            "production_batch_id": ["batch-1"],
            "warranty_policy_key": [2000],
            "manufacturing_plant": ["plant-1"],
            "assembly_line": ["line-1"],
            "build_date": pd.to_datetime(["2025-01-01"]),
            "delivery_date": pd.to_datetime(["2025-02-01"]),
            "in_service_date": pd.to_datetime(["2025-02-15"]),
            "axle_configuration": ["6x4"],
            "fuel_type": ["diesel"],
            "emission_standard": ["E6"],
        }
    )
    model = pd.DataFrame(
        {
            "truck_model_key": [1000],
            "brand": ["brand"],
            "model_name": ["model"],
            "model_year": [2025],
            "segment": ["heavy"],
            "application_type": ["haul"],
            "cab_type": ["day"],
            "engine_platform": ["engine"],
            "gvwr_class": ["class"],
        }
    )
    policy = pd.DataFrame(
        {
            "warranty_policy_key": [2000],
            "coverage_months": [36],
            "coverage_miles": [100000],
            "coverage_engine_hours": [5000],
            "deductible_amount": [100.0],
            "coverage_type": ["full"],
            "effective_start_date": pd.to_datetime([policy_start]),
            "effective_end_date": [pd.NaT],
        }
    )
    dates = pd.DataFrame(
        {
            "full_date": pd.to_datetime(["2026-03-15", "2026-05-15"]),
            "day_number": [15, 15],
            "day_name": ["Sunday", "Friday"],
            "week_number": [11, 20],
            "month_number": [3, 5],
            "month_name": ["March", "May"],
            "quarter_number": [1, 2],
            "year_number": [2026, 2026],
            "fiscal_month": [9, 11],
            "fiscal_quarter": [3, 4],
            "fiscal_year": [2026, 2026],
        }
    )
    centers = pd.DataFrame({"service_center_key": [500], "location_key": [600]})
    locations = pd.DataFrame(
        {
            "location_key": [600],
            "country": ["US"],
            "region": ["West"],
            "climate_zone": ["dry"],
            "terrain_type": ["flat"],
        }
    )
    return {
        "dbo.fact_warranty_claim": claims,
        "dbo.dim_truck": truck,
        "dbo.dim_truck_model": model,
        "dbo.dim_warranty_policy": policy,
        "dbo.dim_date": dates,
        "dbo.dim_service_center": centers,
        "dbo.dim_location": locations,
    }


def test_snapshot_is_one_row_per_eligible_claim_and_target_is_separate() -> None:
    contract, _ = load_mart_contract(ROOT)
    result = build_direct_snapshot(_direct_frames(), contract)
    snapshot = result.snapshot
    assert len(snapshot) == 2
    assert snapshot["warranty_claim_key"].is_unique
    assert snapshot["target__high_cost_claim_flag"].tolist() == [0, 1]
    direct_outputs = {mapping.output_column for mapping in contract.direct_feature_mappings}
    assert direct_outputs.issubset(snapshot.columns)
    assert "target__high_cost_claim_flag" not in direct_outputs
    assert "truck_key" not in snapshot.columns
    assert "total_claim_cost" not in snapshot.columns
    assert result.join_validation["claim_to_truck"]["multiplication_count"] == 0


def test_invalid_policy_timing_retains_claim_and_nulls_policy_features() -> None:
    contract, _ = load_mart_contract(ROOT)
    result = build_direct_snapshot(_direct_frames(policy_start="2026-04-01"), contract)
    snapshot = result.snapshot.set_index("warranty_claim_key")
    assert len(snapshot) == 2
    assert pd.isna(snapshot.loc[1, "warranty_policy__coverage_months"])
    assert bool(snapshot.loc[1, "lineage__policy_applicable"]) is False
    assert snapshot.loc[2, "warranty_policy__coverage_months"] == 36


def test_duplicate_truck_dimension_key_blocks_snapshot() -> None:
    contract, _ = load_mart_contract(ROOT)
    frames = _direct_frames()
    frames["dbo.dim_truck"] = pd.concat(
        [frames["dbo.dim_truck"], frames["dbo.dim_truck"]], ignore_index=True
    )
    with pytest.raises(FeatureMartError, match="duplicate key"):
        build_direct_snapshot(frames, contract)
