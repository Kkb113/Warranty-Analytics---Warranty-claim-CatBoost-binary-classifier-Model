"""Database-independent Phase 3 tests using fictional records only."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from warranty_analytics_model import cli
from warranty_analytics_model.cli import main
from warranty_analytics_model.config import DatabaseSettings, Settings
from warranty_analytics_model.profiling.association import (
    association_table,
    cramer_v,
    missingness_by_target,
    point_biserial,
)
from warranty_analytics_model.profiling.category_sparsity import category_sparsity
from warranty_analytics_model.profiling.column_profile import profile_series
from warranty_analytics_model.profiling.config import load_profiling_settings
from warranty_analytics_model.profiling.duplicate_audit import audit_duplicates
from warranty_analytics_model.profiling.extractor import (
    quote_identifier,
    table_count_sql,
    table_select_sql,
)
from warranty_analytics_model.profiling.findings import (
    Finding,
    finding_counts,
    make_finding,
    overall_status,
)
from warranty_analytics_model.profiling.identifier_audit import audit_identifiers
from warranty_analytics_model.profiling.relational_quality import (
    cost_arithmetic_audit,
    foreign_key_orphans,
)
from warranty_analytics_model.profiling.reporting import REQUIRED_REPORTS, write_phase3_reports
from warranty_analytics_model.profiling.runner import profile_dataframes
from warranty_analytics_model.profiling.synthetic_audit import group_purity, run_synthetic_audit
from warranty_analytics_model.profiling.table_profile import profile_table
from warranty_analytics_model.profiling.target_profile import (
    audit_target_generation,
    profile_target,
)
from warranty_analytics_model.profiling.temporal_quality import (
    component_supplier_quality,
    maintenance_quality,
    service_repair_quality,
    telemetry_quality,
    temporal_rules,
)
from warranty_analytics_model.profiling.text_audit import audit_text


def _claims() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "warranty_claim_key": [1, 2, 3, 4, 5, 6],
            "claim_id": ["FIC-1", "FIC-2", "FIC-3", "FIC-4", "FIC-5", "FIC-6"],
            "truck_key": [1, 1, 2, 2, 3, 3],
            "service_event_key": [1, 2, 3, 4, 5, 6],
            "service_center_key": [1, 1, 2, 2, 1, 1],
            "claim_date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02", "2024-03-01", "2024-03-02"]
            ),
            "repair_start_date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-02-02", "2024-02-03", "2024-03-02", "2024-03-03"]
            ),
            "repair_end_date": pd.to_datetime(
                ["2024-01-03", "2024-01-04", "2024-02-03", "2024-02-04", "2024-03-03", "2024-03-04"]
            ),
            "total_claim_cost": [100.0, 110.0, 500.0, 520.0, 100.0, 110.0],
            "labor_cost": [20.0, 20.0, 100.0, 100.0, 20.0, 20.0],
            "parts_cost": [80.0, 90.0, 400.0, 420.0, 80.0, 90.0],
            "approved_amount": [80.0, 90.0, 400.0, 420.0, 80.0, 90.0],
            "claim_type": ["standard", "standard", "complex", "complex", "standard", "standard"],
            "high_cost_claim_flag": [0, 0, 1, 1, 0, 0],
            "potential_recall_flag": [0, 0, 1, 1, 0, 0],
            "root_cause_category": ["wear", "wear", "thermal", "thermal", "wear", "wear"],
            "description": ["template-a", "template-a", "template-b", "template-b", None, None],
            "production_batch_id": ["B1", "B1", "B2", "B2", "B1", "B1"],
        }
    )


def _frames() -> dict[str, pd.DataFrame]:
    claims = _claims()
    return {
        "dbo.fact_warranty_claim": claims,
        "dbo.dim_truck": pd.DataFrame(
            {
                "truck_key": [1, 2, 3],
                "truck_model_key": [10, 10, 20],
                "customer_key": [1, 1, 2],
                "manufacturing_plant": ["P1", "P1", "P2"],
                "assembly_line": ["A", "A", "B"],
                "production_batch_id": ["B1", "B2", "B1"],
                "build_date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-02-01"]),
                "delivery_date": pd.to_datetime(["2023-01-05", "2023-01-05", "2023-02-05"]),
                "in_service_date": pd.to_datetime(["2023-01-10", "2023-01-10", "2023-02-10"]),
            }
        ),
        "dbo.dim_truck_model": pd.DataFrame(
            {
                "truck_model_key": [10, 20],
                "model_name": ["Model-A", "Model-B"],
                "model_year": [2022, 2023],
            }
        ),
        "dbo.dim_service_center": pd.DataFrame(
            {"service_center_key": [1, 2], "location_key": [1, 2], "dealer_group": ["D1", "D2"]}
        ),
        "dbo.dim_location": pd.DataFrame(
            {
                "location_key": [1, 2],
                "region": ["north", "south"],
                "climate_zone": ["cold", "warm"],
                "terrain_type": ["flat", "hilly"],
            }
        ),
        "dbo.fact_telemetry_monthly": pd.DataFrame(
            {
                "telemetry_month_key": [1, 2, 3, 4],
                "truck_key": [1, 1, 2, 3],
                "month_start_date": pd.to_datetime(
                    ["2024-01-01", "2024-03-01", "2024-01-01", "2024-01-01"]
                ),
                "total_odometer_miles": [10, 5, 10, 0],
                "engine_hours_month": [2.0, 3.0, 1.0, 0.0],
                "idle_hours_month": [1.0, 1.0, 0.0, 0.0],
                "route_severity_score": [1.0, 2.0, 1.0, 1.0],
                "maintenance_compliance_score": [0.9, 0.8, 0.9, 0.9],
            }
        ),
        "dbo.fact_maintenance_event": pd.DataFrame(
            {
                "maintenance_event_key": [1, 2],
                "truck_key": [1, 1],
                "maintenance_date": pd.to_datetime(["2023-11-01", "2023-12-01"]),
                "completed_on_time_flag": [1, 0],
                "overdue_days": [2, 5],
                "maintenance_cost": [50.0, 60.0],
                "technician_notes": ["fictional note", None],
            }
        ),
        "dbo.fact_service_event": pd.DataFrame(
            {
                "service_event_key": [1, 2],
                "truck_key": [1, 1],
                "service_center_key": [1, 1],
                "service_date": pd.to_datetime(["2023-12-01", "2023-12-15"]),
                "complaint_description": ["same fictional phrase", "same fictional phrase"],
                "diagnostic_summary": ["approved result", "approved result"],
                "downtime_hours": [2.0, 3.0],
                "roadside_assistance_flag": [0, 1],
                "repeat_visit_flag": [0, 1],
            }
        ),
        "dbo.fact_repair_line": pd.DataFrame(
            {
                "repair_line_key": [1, 2],
                "warranty_claim_key": [1, 2],
                "service_event_key": [1, 2],
                "part_quantity": [1, 1],
                "part_unit_cost": [10.0, 10.0],
                "labor_hours": [1.0, 1.0],
                "labor_rate": [20.0, 20.0],
                "line_cost": [30.0, 30.0],
                "part_no": ["PART-A", None],
                "technician_id": ["TECH-A", None],
            }
        ),
    }


def test_column_profiles_cover_numeric_categorical_date_text_and_privacy() -> None:
    numeric = profile_series(pd.Series([0, 1, -2, None], name="amount"))
    categorical = profile_series(pd.Series(["a", "a", "b", None], name="category"))
    dates = profile_series(
        pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]), name="claim_date")
    )
    text = profile_series(pd.Series(["Private VIN X", "Private VIN X", ""], name="vin"))
    assert numeric["negative_count"] == 1
    assert categorical["most_frequent_count"] == 2
    assert dates["data_type"] == "date"
    assert text["data_type"] in {"categorical", "text"}
    assert "Private VIN X" not in str(text)


def test_target_threshold_association_and_missingness() -> None:
    claims = _claims()
    target = profile_target(claims)
    audit = audit_target_generation(claims)
    assert target["positive_claims"] == 2
    assert target["positive_percentage"] == pytest.approx(33.333333)
    assert audit["total_claim_cost_deterministic"] is True
    assert "Empirical synthetic target-generation rule suspected" in audit["interpretation"]
    assert point_biserial(claims["total_claim_cost"], claims["high_cost_claim_flag"]) is not None
    assert cramer_v(claims["claim_type"], claims["high_cost_claim_flag"]) is not None
    associations = association_table(claims, columns=["total_claim_cost", "claim_type"])
    assert {row["field"] for row in associations} == {"total_claim_cost", "claim_type"}
    assert len(missingness_by_target(claims, columns=["description"])) == 1


def test_audit_modules_detect_fictional_patterns() -> None:
    frames = _frames()
    claims = _claims()
    identifiers = audit_identifiers(
        claims, columns=["claim_id", "production_batch_id"], minimum_group_support=2
    )
    assert "SYNTHETIC_IDENTIFIER_LEAKAGE" in identifiers["flags"]
    text = audit_text({"dbo.fact_service_event": frames["dbo.fact_service_event"]})
    assert text["fields"]
    assert group_purity(claims, ["production_batch_id"], minimum_support=2)
    duplicate_service = frames["dbo.fact_service_event"].copy()
    duplicate_service.loc[1, duplicate_service.columns.difference(["service_event_key"])] = (
        duplicate_service.loc[0, duplicate_service.columns.difference(["service_event_key"])]
    )
    duplicates = audit_duplicates({"dbo.fact_service_event": duplicate_service})
    assert duplicates["duplicates_found"] is True
    sparsity = category_sparsity(claims, ["claim_type"], target_column="high_cost_claim_flag")
    assert sparsity[0]["distinct_categories"] == 2
    synthetic = run_synthetic_audit(frames, claims)
    assert "target_generation" in synthetic


def test_relational_temporal_telemetry_and_operational_audits() -> None:
    frames = _frames()
    claims = _claims()
    fk = SimpleNamespace(
        name="FK_fake",
        referenced_table="dbo.dim_truck",
        parent_columns=["truck_key"],
        referenced_columns=["truck_key"],
    )
    table_spec = SimpleNamespace(foreign_keys=[fk])
    contract = SimpleNamespace(table_map={"dbo.fact_warranty_claim": table_spec})
    orphan_claims = claims.assign(truck_key=[1, 1, 2, 2, 3, 99])
    relational = foreign_key_orphans({**frames, "dbo.fact_warranty_claim": orphan_claims}, contract)
    assert relational[0]["orphan_count"] == 1
    repairs = frames["dbo.fact_repair_line"]
    assert cost_arithmetic_audit(repairs)["exact_match_percentage"] == 100.0
    temporal = temporal_rules(frames)
    assert temporal
    telemetry = telemetry_quality(frames["dbo.fact_telemetry_monthly"])
    assert telemetry["odometer_decrease_count"] == 1
    assert telemetry["missing_month_gap_count"] == 1
    maintenance = maintenance_quality(frames["dbo.fact_maintenance_event"])
    assert maintenance["logical_conflict_count"] == 1
    operations = service_repair_quality(frames["dbo.fact_service_event"], repairs)
    assert operations["service"]["repeat_visit_rate_percentage"] == 50.0
    components = component_supplier_quality(None, None, None)
    assert components == {"installation": {}, "component": {}, "supplier": {}}


def test_table_profile_pipeline_and_reports_are_secret_safe(tmp_path: Path) -> None:
    frames = _frames()
    table = profile_table("dbo.fact_warranty_claim", _claims())
    assert table["row_count"] == 6
    assert table["column_profiles"]["warranty_claim_key"]["data_type"] == "numeric"
    result = profile_dataframes(frames)
    assert result["target_profile"]["claims"] == 6
    report_dir = write_phase3_reports(
        result, tmp_path, run_timestamp=datetime(2024, 1, 1, tzinfo=UTC)
    )
    assert set(path.name for path in report_dir.iterdir()) == set(REQUIRED_REPORTS)
    content = "".join(path.read_text(encoding="utf-8") for path in report_dir.iterdir())
    assert "Private VIN" not in content
    assert "fictional note" not in content
    assert result["scope_confirmations"]["database_writes"] is False


def test_config_sql_safety_findings_and_live_cli_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = load_profiling_settings()
    assert settings.chunk_size == 10000
    assert settings.resolved_output_directory().name == "data_profiling"
    assert quote_identifier("safe_name") == "[safe_name]"
    with pytest.raises(ValueError):
        quote_identifier("unsafe]name")
    table = SimpleNamespace(schema="dbo", table="fake_table", columns=[SimpleNamespace(name="id")])
    assert "COUNT_BIG" in table_count_sql(table)
    assert "[id]" in table_select_sql(table)
    warning = make_finding("SAMPLE_WARNING", "WARNING", "test", "fictional")
    assert finding_counts([warning]) == {"ERROR": 0, "WARNING": 1, "INFO": 0}
    assert overall_status([warning]) == "READY WITH WARNINGS"
    assert Finding.model_validate(warning.model_dump())
    monkeypatch.setattr(cli, "load_settings", lambda: Settings(database=DatabaseSettings()))
    assert main(["data-profile", "--output-dir", str(tmp_path)]) == 2
    assert "WARRANTY_DB_SERVER" in capsys.readouterr().err
