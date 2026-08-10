"""Temporary Parquet bundle, manifest, and corruption-detection tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from warranty_analytics_model.cli import main
from warranty_analytics_model.feature_mart.models import FeatureMartError, FeatureMartSettings
from warranty_analytics_model.feature_mart.runner import (
    build_feature_mart_from_frames,
    validate_existing_mart,
)

from .test_phase5_bridges import _history_sources
from .test_phase5_snapshot import _direct_frames

ROOT = Path(__file__).resolve().parents[3]


def _all_frames() -> dict[str, pd.DataFrame]:
    sources = _history_sources()
    frames = _direct_frames()
    frames.update(
        {
            "dbo.fact_telemetry_monthly": sources["telemetry"],
            "dbo.fact_maintenance_event": sources["maintenance"],
            "dbo.fact_service_event": sources["service"],
            "dbo.fact_component_installation": sources["installations"],
            "dbo.dim_component": sources["components"],
            "dbo.dim_failure_code": sources["failure_codes"],
            "dbo.fact_repair_line": sources["repair_lines"],
        }
    )
    return frames


def test_parquet_manifest_round_trip_and_corruption_gate(tmp_path: Path) -> None:
    """A temporary bundle round-trips and a changed artifact is blocked."""

    pytest.importorskip("pyarrow")
    frames = _all_frames()
    result = build_feature_mart_from_frames(
        frames=frames,
        source_row_counts={name: len(frame) for name, frame in frames.items()},
        root=ROOT,
        settings=FeatureMartSettings(),
        environment="test",
        source_database="warranty_analytics",
        output_root=tmp_path / "artifacts",
        report_root=tmp_path / "reports",
        run_id="test-run",
    )
    mart_dir = Path(result.run_directory)
    assert result.status == "PASS WITH WARNINGS"
    assert (mart_dir / "claim_snapshot.parquet").is_file()
    assert (mart_dir / "history" / "telemetry_history.parquet").is_file()
    assert (mart_dir / "lineage" / "claim_group_membership.parquet").is_file()
    reports = Path(result.report_directory or "")
    assert (reports / "phase_5_summary.json").is_file()
    assert (reports / "phase_5_summary.md").is_file()
    assert not json.loads((mart_dir / "validation.json").read_text(encoding="utf-8"))["errors"]
    assert validate_existing_mart(mart_dir)["status"] == "PASS WITH WARNINGS"
    assert main(["phase5-validate", "--mart-dir", str(mart_dir)]) == 0

    snapshot = pd.read_parquet(mart_dir / "claim_snapshot.parquet")
    snapshot.loc[0, "claim__claim_type"] = "changed"
    snapshot.to_parquet(mart_dir / "claim_snapshot.parquet", index=False)
    with pytest.raises(FeatureMartError, match="checksum|fingerprint"):
        validate_existing_mart(mart_dir)
