"""Contract, manifest, report, and artifact-validation tests for Phase 7."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tests.unit.structured_features.test_builder import _frames
from warranty_analytics_model.feature_mart.manifest import content_sha256, sha256_file
from warranty_analytics_model.feature_mart.mart_contract import load_mart_contract
from warranty_analytics_model.splits.input import Phase5MartInput
from warranty_analytics_model.splits.manifest import (
    assignment_content_sha256,
    claim_key_sha256,
    unordered_claim_key_sha256,
)
from warranty_analytics_model.structured_features import contract as contract_module
from warranty_analytics_model.structured_features.builder import build_feature_matrix
from warranty_analytics_model.structured_features.config import load_structured_feature_settings
from warranty_analytics_model.structured_features.contract import (
    load_structured_feature_contract,
    validate_structured_feature_contract,
)
from warranty_analytics_model.structured_features.input import (
    load_phase7_inputs,
    phase7_plan_check,
    required_source_columns,
    verify_frozen_membership,
)
from warranty_analytics_model.structured_features.manifest import (
    definition_payload,
    feature_manifest,
    quality_diagnostics,
    source_coverage,
    write_feature_artifacts,
)
from warranty_analytics_model.structured_features.models import (
    Phase7Inputs,
    StructuredFeatureError,
    StructuredFeatureSettings,
)
from warranty_analytics_model.structured_features.reporting import write_phase7_reports
from warranty_analytics_model.structured_features.runner import (
    build_phase7,
    phase7_contract_check,
    phase7_run_id,
)
from warranty_analytics_model.structured_features.source_policy import validate_lineage_sources
from warranty_analytics_model.structured_features.validation import validate_feature_directory


def _assignments() -> pd.DataFrame:
    claims = _frames()["claim_snapshot"]
    return pd.DataFrame(
        {
            "warranty_claim_key": claims["warranty_claim_key"],
            "claim_date": claims["claim__claim_date"],
            "split": ["TRAIN", "VALIDATION", "TEST", "TEST"],
        }
    )


def test_phase7_contract_and_configuration_pass() -> None:
    settings = load_structured_feature_settings()
    assert settings.windows_months == (3, 6, 12, 24)
    contract, checksum = load_structured_feature_contract()
    assert contract["contract_version"] == "1.0.0"
    assert len(checksum) == 64
    result = validate_structured_feature_contract()
    assert result["valid"] is True
    assert phase7_contract_check()["valid"] is True
    assert len(phase7_run_id()) == 16
    assert set(required_source_columns()) >= {
        "claim_snapshot",
        "telemetry_history",
        "repair_history_index",
    }


def test_feature_lineage_distinguishes_values_from_controls() -> None:
    assignments = _assignments()
    built = build_feature_matrix(_frames(), assignments, StructuredFeatureSettings())
    definitions = {item.feature_name: item for item in built.definitions}
    event_count = definitions["maintenance__3m__event_count"]
    fault_sum = definitions["telemetry__12m__fault_code_count__sum"]
    recency = definitions["maintenance__days_since_last_event"]
    prior_count = definitions["prior_claim__3m__claim_count"]
    latest_type = definitions["maintenance__last_type"]
    assert "maintenance_event_key" in event_count.control_sources
    assert "maintenance_event_key" not in event_count.value_sources
    assert fault_sum.value_sources == ("telemetry__fault_code_count",)
    assert "telemetry__month_start_date" in fault_sum.control_sources
    assert set(recency.value_sources) == {
        "maintenance__maintenance_date",
        "claim__claim_date",
    }
    assert "prior_warranty_claim_key" in prior_count.control_sources
    assert "prior_warranty_claim_key" not in prior_count.value_sources
    assert latest_type.value_sources == ("maintenance__maintenance_type",)
    assert "maintenance_event_key" in latest_type.control_sources
    result = validate_lineage_sources(definition_payload(built.definitions))
    assert result["valid"] is True


@pytest.mark.parametrize(
    "source",
    [
        "supplier_key",
        "production_batch_id",
        "target__high_cost_claim_flag",
        "causal_part_no",
        "total_claim_cost",
    ],
)
def test_lineage_policy_rejects_unsafe_value_sources(source: str) -> None:
    result = validate_lineage_sources(
        {
            "fictional_feature": {
                "is_model_feature": True,
                "source_columns": [source],
                "value_sources": [source],
                "control_sources": [],
            }
        }
    )
    assert result["valid"] is False
    assert source in " ".join(result["errors"])


def test_control_only_source_is_allowed_only_as_control_metadata() -> None:
    result = validate_lineage_sources(
        {
            "event_count": {
                "is_model_feature": True,
                "source_columns": ["maintenance_event_key"],
                "value_sources": [],
                "control_sources": ["maintenance_event_key"],
            }
        }
    )
    assert result["valid"] is True


def test_telemetry_coverage_range_is_reported_as_structural_diagnostic() -> None:
    built = build_feature_matrix(_frames(), _assignments(), StructuredFeatureSettings())
    frame = built.frame.copy()
    frame.loc[0, "telemetry__3m__coverage_ratio"] = 1.1
    quality = quality_diagnostics(frame, built.definitions)
    assert quality["telemetry_coverage_out_of_range"] == {"telemetry__3m__coverage_ratio": 1}


def test_manifest_and_validation_preserve_frozen_membership(tmp_path: Path) -> None:
    frames = _frames()
    assignments = _assignments()
    built = build_feature_matrix(frames, assignments, StructuredFeatureSettings())
    feature_dir = tmp_path / "feature"
    metadata = write_feature_artifacts(
        feature_dir,
        frame=built.frame,
        definitions=built.definitions,
        settings=StructuredFeatureSettings(),
    )
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    assignments.to_parquet(split_dir / "split_assignments.parquet", index=False)
    test = assignments.loc[assignments["split"] == "TEST"]
    split_manifest = {
        "split_assignment_sha256": assignment_content_sha256(assignments),
        "train_claim_key_sha256": claim_key_sha256(
            assignments.loc[assignments["split"] == "TRAIN"]
        ),
        "validation_claim_key_sha256": claim_key_sha256(
            assignments.loc[assignments["split"] == "VALIDATION"]
        ),
        "test_claim_key_sha256": claim_key_sha256(test),
    }
    test_lock = {
        "ordered_test_claim_keys_sha256": claim_key_sha256(test),
        "unordered_test_claim_keys_sha256": unordered_claim_key_sha256(test),
        "test_assignment_content_sha256": assignment_content_sha256(test),
    }
    (split_dir / "split_manifest.json").write_text(json.dumps(split_manifest), encoding="utf-8")
    (split_dir / "test_lock.json").write_text(json.dumps(test_lock), encoding="utf-8")
    inputs = Phase7Inputs(
        root=tmp_path,
        mart_dir=tmp_path / "mart",
        split_dir=split_dir,
        mart_manifest={"artifact_file_sha256": {}, "artifact_content_fingerprints": {}},
        split_manifest=split_manifest,
        test_lock=test_lock,
        frames=frames,
        phase5_validation={"status": "PASS WITH WARNINGS", "warnings": []},
        phase6_validation={"status": "PASS WITH WARNINGS", "warnings": []},
        phase5_contract_checksum="mart-contract",
        phase6_contract_checksum="split-contract",
        phase5_manifest_checksum="mart-manifest",
    )
    frozen = verify_frozen_membership(inputs)
    assert frozen["valid"] is True
    mart_contract, _ = load_mart_contract()
    coverage = source_coverage(built.definitions, mart_contract)
    inventory = feature_manifest(built.definitions)
    quality = quality_diagnostics(built.frame, built.definitions)
    assert inventory["total_feature_count"] > 0
    assert definition_payload(built.definitions)
    manifest = {
        "input_phase5_mart": {"run": "mart"},
        "input_phase6_split": {"run": "split"},
        "split_assignment_sha256": split_manifest["split_assignment_sha256"],
        "test_lock_hashes": test_lock,
        "artifact_file_sha256": {"structured_features": metadata["file_sha256"]},
        "artifact_content_sha256": {"structured_features": metadata["content_sha256"]},
    }
    (feature_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    validation = validate_feature_directory(feature_dir, project_root=Path.cwd(), inputs=inputs)
    assert validation["errors"] == []
    assert validation["checks"]["test_lock_valid"] is True
    quality_path = feature_dir / "feature_quality.json"
    quality_payload = json.loads(quality_path.read_text(encoding="utf-8"))
    quality_payload["telemetry_coverage_out_of_range"] = {"telemetry__3m__coverage_ratio": 1}
    quality_path.write_text(json.dumps(quality_payload), encoding="utf-8")
    blocked = validate_feature_directory(feature_dir, project_root=Path.cwd(), inputs=inputs)
    assert blocked["status"] == "BLOCKED"
    assert any("outside [0, 1]" in error for error in blocked["errors"])
    write_phase7_reports(
        tmp_path / "reports",
        manifest={
            **manifest,
            "validation_status": validation["status"],
            "feature_manifest": inventory,
            "phase7_contract_checksum": "contract",
            "row_count": 4,
            "train_count": 1,
            "validation_count": 1,
            "test_count": 2,
        },
        validation=validation,
        quality=quality,
        lineage=json.loads((feature_dir / "feature_lineage.json").read_text()),
        coverage=coverage,
    )
    assert (tmp_path / "reports" / "phase_7_summary.md").is_file()
    assert sha256_file(feature_dir / "structured_features.parquet") == metadata["file_sha256"]
    assert (
        content_sha256(pd.read_parquet(feature_dir / "structured_features.parquet"))
        == metadata["content_sha256"]
    )


def test_load_phase7_inputs_uses_validated_offline_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = _frames()
    mart_dir = tmp_path / "mart" / "run"
    (mart_dir / "history").mkdir(parents=True)
    required = [
        "claim_snapshot.parquet",
        "history/telemetry_history.parquet",
        "history/maintenance_history.parquet",
        "history/service_history.parquet",
        "history/component_installation_history.parquet",
        "history/prior_claim_history.parquet",
        "history/repair_history_index.parquet",
    ]
    for relative in required:
        (mart_dir / relative).touch()
    phase5 = Phase5MartInput(
        root=tmp_path,
        mart_dir=mart_dir,
        manifest={"eligible_claims": 4},
        frames=frames,
        phase5_validation={"status": "PASS WITH WARNINGS", "warnings": []},
        mart_contract=None,
        mart_contract_checksum="mart",
        phase4_bundle=None,
        schema_contract_checksum="schema",
        mart_manifest_checksum="manifest",
        claim_snapshot_file_sha256="snapshot-file",
        claim_snapshot_content_sha256="snapshot-content",
        group_membership_file_sha256="group-file",
        group_membership_content_sha256="group-content",
    )
    monkeypatch.setattr(
        "warranty_analytics_model.structured_features.input.load_phase5_mart",
        lambda *_args, **_kwargs: phase5,
    )
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    assignments = _assignments()
    assignments.to_parquet(split_dir / "split_assignments.parquet", index=False)
    (split_dir / "split_manifest.json").write_text(
        json.dumps({"input_mart_run": "run", "split_contract_checksum": "split"}),
        encoding="utf-8",
    )
    (split_dir / "test_lock.json").write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setattr(
        "warranty_analytics_model.structured_features.input.validate_split_artifacts",
        lambda *_args, **_kwargs: {
            "errors": [],
            "warnings": [],
            "checks": {"test_lock_valid": True},
        },
    )
    loaded = load_phase7_inputs(mart_dir, split_dir, project_root=Path.cwd())
    assert loaded.mart_dir == mart_dir.resolve()
    assert loaded.phase6_contract_checksum == "split"


def test_runner_publishes_atomic_run_with_validated_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = _frames()
    assignments = _assignments()
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    assignments.to_parquet(split_dir / "split_assignments.parquet", index=False)
    fake_inputs = Phase7Inputs(
        root=Path.cwd(),
        mart_dir=tmp_path / "mart" / "run",
        split_dir=split_dir,
        mart_manifest={"artifact_file_sha256": {}, "artifact_content_fingerprints": {}},
        split_manifest={
            "split_assignment_sha256": "assignment",
            "train_claim_key_sha256": "train",
            "validation_claim_key_sha256": "validation",
            "test_claim_key_sha256": "test",
        },
        test_lock={},
        frames=frames,
        phase5_validation={"warnings": []},
        phase6_validation={"warnings": []},
        phase5_contract_checksum="mart",
        phase6_contract_checksum="split",
        phase5_manifest_checksum="manifest",
    )
    monkeypatch.setattr(
        "warranty_analytics_model.structured_features.runner.phase7_plan_check",
        lambda *_args, **_kwargs: {
            "valid": True,
            "warnings": [],
            "errors": [],
            "inputs": fake_inputs,
        },
    )
    monkeypatch.setattr(
        "warranty_analytics_model.structured_features.runner.verify_frozen_membership",
        lambda *_args, **_kwargs: {"valid": True, "errors": []},
    )
    monkeypatch.setattr(
        "warranty_analytics_model.structured_features.runner.validate_feature_directory",
        lambda *_args, **_kwargs: {
            "status": "PASS",
            "errors": [],
            "warnings": [],
            "checks": {},
            "row_count": 4,
            "model_feature_count": 1,
        },
    )
    result = build_phase7(
        tmp_path / "mart" / "run",
        split_dir,
        output_dir=tmp_path / "features",
        report_dir=tmp_path / "reports",
        run_id="TEST_RUN",
    )
    assert result["status"] == "PASS"
    assert (tmp_path / "features" / "TEST_RUN" / "structured_features.parquet").is_file()
    assert (tmp_path / "reports" / "TEST_RUN" / "phase_7_summary.json").is_file()


def test_configuration_and_contract_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("structured_features:\n  windows_months: [2]\n", encoding="utf-8")
    with pytest.raises(StructuredFeatureError):
        load_structured_feature_settings(path=bad_config)
    bad_config.write_text("structured_features:\n  compression: invalid\n", encoding="utf-8")
    with pytest.raises(StructuredFeatureError):
        load_structured_feature_settings(path=bad_config)
    payload, checksum = load_structured_feature_contract()
    payload["feature_grain"] = "wrong grain"
    payload["feature_tiers"] = {"RESTRICTED_EXPERIMENTAL": {}}
    payload["target_independence_policy"]["target_column_excluded"] = False
    payload["deferred_sources"] = []
    payload["structured_feature_families"] = ["supplier_key"]
    monkeypatch.setattr(
        contract_module,
        "load_structured_feature_contract",
        lambda *_args, **_kwargs: (payload, checksum),
    )
    result = contract_module.validate_structured_feature_contract()
    assert result["valid"] is False


def test_missing_inputs_and_feature_directory_block() -> None:
    result = phase7_plan_check(Path("missing-mart"), Path("missing-split"))
    assert result["valid"] is False
    validation = validate_feature_directory(Path("missing-feature-run"))
    assert validation["status"] == "BLOCKED"
