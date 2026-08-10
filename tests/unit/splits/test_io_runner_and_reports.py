"""Fictional artifact, runner, report, and validator coverage for Phase 6."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from warranty_analytics_model.splits.cohorts import build_evaluation_cohorts
from warranty_analytics_model.splits.config import load_split_settings
from warranty_analytics_model.splits.group_exposure import build_group_exposure
from warranty_analytics_model.splits.input import load_phase5_mart
from warranty_analytics_model.splits.models import SplitError, SplitSettings
from warranty_analytics_model.splits.reporting import (
    build_phase6_summary,
    build_split_distribution,
    write_phase6_reports,
)
from warranty_analytics_model.splits.runner import build_phase6, phase6_plan_check_from_input
from warranty_analytics_model.splits.validation import validate_split_artifacts

ROOT = Path(__file__).resolve().parents[3]


def _fake_input() -> SimpleNamespace:
    snapshot = pd.DataFrame(
        {
            "warranty_claim_key": list(range(1, 13)),
            "claim__claim_date": pd.date_range("2025-01-01", periods=12, freq="D"),
            "target__high_cost_claim_flag": [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        }
    )
    group_rows: list[tuple[int, str, str]] = []
    for claim_key in range(1, 13):
        group_rows.extend(
            [
                (claim_key, "truck", f"truck-{claim_key // 3}"),
                (claim_key, "production_batch", f"batch-{claim_key // 4}"),
                (claim_key, "service_center", f"center-{claim_key // 5}"),
                (claim_key, "safe_scenario_fingerprint", f"fingerprint-{claim_key // 2}"),
            ]
        )
    groups = pd.DataFrame(
        {
            "warranty_claim_key": [row[0] for row in group_rows],
            "group_type": [row[1] for row in group_rows],
            "group_value_hash": [row[2] for row in group_rows],
            "group_value": [row[2] for row in group_rows],
            "source": ["fictional"] * len(group_rows),
            "is_model_feature": [False] * len(group_rows),
        }
    )
    phase4 = SimpleNamespace(
        target_checksum="target",
        feature_policy_checksum="feature",
        leakage_checksum="leakage",
    )
    return SimpleNamespace(
        root=ROOT,
        mart_dir=ROOT / "artifacts" / "feature_mart" / "fictional",
        manifest={"source_drift": {}, "validation_status": "PASS"},
        frames={"claim_snapshot": snapshot, "claim_group_membership": groups},
        phase5_validation={"status": "PASS", "warnings": []},
        mart_contract=SimpleNamespace(version="1.0.0"),
        mart_contract_checksum="mart",
        phase4_bundle=phase4,
        schema_contract_checksum="schema",
        mart_manifest_checksum="manifest",
        claim_snapshot_file_sha256="snapshot-file",
        claim_snapshot_content_sha256="snapshot-content",
        group_membership_file_sha256="group-file",
        group_membership_content_sha256="group-content",
    )


def _fake_contract() -> SimpleNamespace:
    return SimpleNamespace(contract_version="1.0.0")


def _fake_settings() -> SplitSettings:
    return SplitSettings.model_validate(
        {
            "strategy": "chronological",
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "preserve_same_date": True,
            "tie_break": "earlier_date",
            "min_positive_block_validation": 1,
            "min_positive_block_test": 1,
            "min_positive_warning_validation": 2,
            "min_positive_warning_test": 2,
            "min_positive_warning_train": 2,
        }
    )


def test_split_settings_loads_and_reports_as_dict() -> None:
    settings = load_split_settings(ROOT)

    assert settings.requested_fractions == {"TRAIN": 0.7, "VALIDATION": 0.15, "TEST": 0.15}


def test_plan_check_from_loaded_input_reports_group_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import warranty_analytics_model.splits.runner as runner_module

    phase5_input = _fake_input()
    monkeypatch.setattr(
        runner_module,
        "validate_split_contract",
        lambda *args, **kwargs: SimpleNamespace(
            errors=["fictional contract mismatch"],
            warnings=[],
            valid=False,
            model_dump=lambda **dump_kwargs: {},
        ),
    )
    result = phase6_plan_check_from_input(
        phase5_input,
        _fake_contract(),
        "split",
        _fake_settings(),
    )

    assert result["valid"] is False
    assert result["group_types"]["available_group_types"]
    assert result["mart_run"] == "fictional"


def test_phase6_runner_builds_atomic_artifacts_and_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyarrow")
    import warranty_analytics_model.splits.runner as runner_module
    import warranty_analytics_model.splits.validation as validation_module

    phase5_input = _fake_input()
    settings = _fake_settings()
    monkeypatch.setattr(runner_module, "load_phase5_mart", lambda *args, **kwargs: phase5_input)
    monkeypatch.setattr(runner_module, "load_split_settings", lambda *args, **kwargs: settings)
    monkeypatch.setattr(
        runner_module,
        "load_split_contract",
        lambda *args, **kwargs: (_fake_contract(), "split"),
    )
    monkeypatch.setattr(
        runner_module,
        "phase6_plan_check_from_input",
        lambda *args, **kwargs: {"valid": True, "errors": []},
    )
    monkeypatch.setattr(validation_module, "_contract_compatibility_errors", lambda *args: [])
    result = build_phase6(
        phase5_input.mart_dir,
        output_dir=tmp_path / "artifacts",
        report_dir=tmp_path / "reports",
        run_id="fictional-run",
        project_root=ROOT,
    )

    run_dir = Path(result.run_directory)
    assert result.status == "PASS WITH WARNINGS"
    assert (run_dir / "split_assignments.parquet").is_file()
    assert (run_dir / "group_exposure.parquet").is_file()
    assert (run_dir / "evaluation_cohorts.parquet").is_file()
    assert (run_dir / "test_lock.json").is_file()
    assert Path(result.report_directory or "").joinpath("phase_6_summary.md").is_file()


def test_validator_accepts_built_fake_artifacts_and_reports_missing_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyarrow")
    import json

    import warranty_analytics_model.splits.runner as runner_module
    import warranty_analytics_model.splits.validation as validation_module

    phase5_input = _fake_input()
    settings = _fake_settings()
    monkeypatch.setattr(runner_module, "load_phase5_mart", lambda *args, **kwargs: phase5_input)
    monkeypatch.setattr(runner_module, "load_split_settings", lambda *args, **kwargs: settings)
    monkeypatch.setattr(
        runner_module,
        "load_split_contract",
        lambda *args, **kwargs: (_fake_contract(), "split"),
    )
    monkeypatch.setattr(
        runner_module,
        "phase6_plan_check_from_input",
        lambda *args, **kwargs: {"valid": True, "errors": []},
    )
    monkeypatch.setattr(validation_module, "_contract_compatibility_errors", lambda *args: [])
    result = build_phase6(
        phase5_input.mart_dir,
        output_dir=tmp_path / "artifacts",
        report_dir=tmp_path / "reports",
        run_id="validator-run",
        project_root=ROOT,
    )
    run_dir = Path(result.run_directory)
    original_assignments = pd.read_parquet(run_dir / "split_assignments.parquet")
    original_exposure = build_group_exposure(
        original_assignments,
        phase5_input.frames["claim_group_membership"],
    )
    original_cohorts = build_evaluation_cohorts(
        original_assignments,
        phase5_input.frames["claim_group_membership"],
    )
    assignments = original_assignments.iloc[:-1]
    assignments.to_parquet(run_dir / "split_assignments.parquet", index=False)
    invalid = validate_split_artifacts(
        run_dir,
        project_root=ROOT,
        input_mart=phase5_input,
        expected_exposure=original_exposure,
        expected_cohorts=original_cohorts,
    )

    assert invalid["status"] == "BLOCKED"
    assert any("cover exactly" in error or "expected" in error for error in invalid["errors"])
    assert "warranty_claim_key" not in json.dumps(invalid)


def test_reports_are_aggregate_only(tmp_path: Path) -> None:
    assignments = pd.DataFrame(
        {
            "warranty_claim_key": [1, 2, 3],
            "claim_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "split": ["TRAIN", "VALIDATION", "TEST"],
        }
    )
    snapshot = pd.DataFrame(
        {
            "warranty_claim_key": [1, 2, 3],
            "target__high_cost_claim_flag": [0, 1, 0],
        }
    )
    distribution = build_split_distribution(assignments, snapshot)
    validation = {
        "status": "PASS WITH WARNINGS",
        "errors": [],
        "warnings": ["fictional warning"],
        "checks": {"same_date_integrity_valid": True, "claim_coverage_valid": True},
    }
    summary = build_phase6_summary(
        manifest={
            "validation_status": "PASS WITH WARNINGS",
            "input_mart_run": "fictional",
            "train_end_date": "2025-01-01",
            "validation_end_date": "2025-01-02",
            "warnings": ["fictional warning"],
        },
        validation=validation,
        split_distribution=distribution,
        group_overlap={"group_types": {}},
        cohort_summary={"by_split": {}},
        fingerprint_overlap={"overlap_severity": "WARNING"},
        phase5_validation={"status": "PASS WITH WARNINGS"},
        test_lock_valid=True,
    )
    paths = write_phase6_reports(
        output_root=tmp_path,
        summary=summary,
        split_distribution=distribution,
        group_overlap={"group_types": {}},
        cohort_summary={"by_split": {}},
        validation=validation,
    )

    assert len(paths) == 6
    assert (tmp_path / "phase_6_summary.md").is_file()
    assert "warranty_claim_key" not in (tmp_path / "phase_6_summary.json").read_text()


def test_phase5_input_missing_directory_blocks() -> None:
    with pytest.raises(SplitError, match="missing"):
        load_phase5_mart(ROOT / "artifacts" / "feature_mart" / "does-not-exist", project_root=ROOT)


def test_phase6_validator_missing_manifest_blocks(tmp_path: Path) -> None:
    result = validate_split_artifacts(tmp_path, project_root=ROOT)

    assert result["status"] == "BLOCKED"
    assert any("split_manifest" in error for error in result["errors"])
