"""Fail-closed Phase 6 input and contract guard tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from warranty_analytics_model.database.schema_contract import load_schema_contract
from warranty_analytics_model.feature_mart.mart_contract import load_mart_contract
from warranty_analytics_model.policy.loader import load_phase4_contracts
from warranty_analytics_model.splits.config import load_split_settings, validate_split_settings
from warranty_analytics_model.splits.input import (
    _required_artifact_hash,
    _verify_artifact_hashes,
    load_phase5_mart,
)
from warranty_analytics_model.splits.models import SplitError
from warranty_analytics_model.splits.split_contract import (
    load_split_contract,
    validate_split_contract,
)
from warranty_analytics_model.splits.validation import _artifact_check, _date_string, _read_json

ROOT = Path(__file__).resolve().parents[3]


def _input_frames() -> dict[str, pd.DataFrame]:
    snapshot = pd.DataFrame(
        {
            "warranty_claim_key": [1, 2, 3],
            "claim__claim_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "target__high_cost_claim_flag": [0, 1, 0],
        }
    )
    groups = pd.DataFrame(
        {
            "warranty_claim_key": [1, 2, 3],
            "group_type": ["truck", "truck", "truck"],
            "group_value_hash": ["a", "b", "c"],
            "is_model_feature": [False, False, False],
        }
    )
    return {"claim_snapshot": snapshot, "claim_group_membership": groups}


def test_phase5_input_success_path_is_checksum_and_contract_gated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import warranty_analytics_model.splits.input as input_module

    mart_dir = tmp_path / "mart"
    mart_dir.mkdir()
    (mart_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (mart_dir / "claim_snapshot.parquet").write_bytes(b"snapshot")
    (mart_dir / "lineage").mkdir()
    (mart_dir / "lineage" / "claim_group_membership.parquet").write_bytes(b"groups")
    frames = _input_frames()
    manifest = {
        "validation_status": "PASS WITH WARNINGS",
        "eligible_claims": 3,
        "mart_contract_checksum": "mart",
        "schema_contract_checksum": "schema",
        "target_contract_checksum": "target",
        "feature_policy_checksum": "feature",
        "leakage_policy_checksum": "leakage",
        "artifact_paths": {
            "claim_snapshot": "claim_snapshot.parquet",
            "claim_group_membership": "lineage/claim_group_membership.parquet",
        },
        "artifact_file_sha256": {"claim_snapshot": "file", "claim_group_membership": "file"},
        "artifact_content_fingerprints": {
            "claim_snapshot": "content",
            "claim_group_membership": "content",
        },
    }
    phase4 = SimpleNamespace(
        target_checksum="target",
        feature_policy_checksum="feature",
        leakage_checksum="leakage",
    )
    mart = SimpleNamespace(version="1.0.0")
    monkeypatch.setattr(
        input_module,
        "validate_artifact_integrity",
        lambda path: ({"manifest": manifest, "column_manifest": []}, frames),
    )
    monkeypatch.setattr(
        input_module,
        "validate_mart_directory",
        lambda *args, **kwargs: {"status": "PASS WITH WARNINGS", "errors": [], "warnings": []},
    )
    monkeypatch.setattr(input_module, "load_schema_contract", lambda root: (object(), "schema"))
    monkeypatch.setattr(input_module, "load_phase4_contracts", lambda root: phase4)
    monkeypatch.setattr(input_module, "load_mart_contract", lambda root: (mart, "mart"))
    monkeypatch.setattr(
        input_module,
        "validate_mart_contract",
        lambda *args, **kwargs: SimpleNamespace(valid=True, errors=[]),
    )
    monkeypatch.setattr(input_module, "sha256_file", lambda path: "file")
    monkeypatch.setattr(input_module, "content_sha256", lambda frame: "content")

    loaded = load_phase5_mart(mart_dir, project_root=ROOT)

    assert loaded.mart_contract_checksum == "mart"
    assert loaded.claim_snapshot_content_sha256 == "content"
    assert loaded.group_membership_file_sha256 == "file"


def test_invalid_split_contract_reports_all_material_safety_failures() -> None:
    contract, checksum = load_split_contract(ROOT)
    settings = load_split_settings(ROOT)
    schema_contract, schema_checksum = load_schema_contract(ROOT)
    phase4 = load_phase4_contracts(ROOT)
    mart, mart_checksum = load_mart_contract(ROOT)
    bad = contract.model_copy(
        update={
            "requested_fractions": {"TRAIN": 0.8, "VALIDATION": 0.1, "TEST": 0.1, "OTHER": 0.0},
            "split_strategy": "random_stratified",
            "tie_breaking_rule": "later_date",
            "same_day_policy": {"preserve_same_date": False},
            "boundary_algorithm": {
                "name": "random_target_rate",
                "target_independent": False,
            },
            "prediction_reference": "submission_timestamp",
            "input_mart_contract_version": "0.0.0",
            "input_mart_contract_checksum": "bad",
            "input_schema_contract_checksum": "bad",
            "input_target_contract_checksum": "bad",
            "input_feature_policy_checksum": "bad",
            "input_leakage_policy_checksum": "bad",
            "test_access_policy": {
                "allowed_first_target_evaluation_phase": 14,
                "phase_9_to_14_target_evaluation_for_development": True,
            },
            "group_exposure_policy": {"enabled": False},
            "scenario_fingerprint_policy": {
                "fingerprint_clean_cohort_defined": False,
                "overlap_severity": "BLOCK",
            },
            "artifact_layout": {"split_assignments": "wrong.parquet"},
            "validation_policy": {"claim_coverage_blocking": False},
            "development_status": {
                "development_mode": "production",
                "production_approved": True,
                "real_data_reapproval_required": False,
                "business_target_definition_confirmed": True,
                "precise_submission_timestamp_available": True,
            },
        }
    )
    result = validate_split_contract(
        bad,
        mart_contract=mart,
        mart_contract_checksum=mart_checksum,
        schema_contract_checksum=schema_checksum,
        phase4_bundle=phase4,
        settings=settings,
        split_contract_checksum_value=checksum,
    )

    assert not result.valid
    assert any("chronological" in error for error in result.errors)
    assert any("same-date" in error for error in result.errors)
    assert any("Phase 15" in error for error in result.errors)
    assert any("production approval" in error for error in result.errors)


def test_split_boundary_contract_rejects_target_used_in_boundary_inputs() -> None:
    contract, checksum = load_split_contract(ROOT)
    settings = load_split_settings(ROOT)
    schema_contract, schema_checksum = load_schema_contract(ROOT)
    phase4 = load_phase4_contracts(ROOT)
    mart, mart_checksum = load_mart_contract(ROOT)
    bad = contract.model_copy(
        update={
            "boundary_algorithm": {
                "name": "date_count_cumulative_closest",
                "inputs": ["claim__claim_date", "target__high_cost_claim_flag"],
                "target_independent": True,
                "target_column_used": True,
            }
        }
    )

    result = validate_split_contract(
        bad,
        mart_contract=mart,
        mart_contract_checksum=mart_checksum,
        schema_contract_checksum=schema_checksum,
        phase4_bundle=phase4,
        settings=settings,
        split_contract_checksum_value=checksum,
    )

    assert not result.valid
    assert any("target_column_used" in error for error in result.errors)
    assert any("inputs must not include" in error for error in result.errors)


def test_phase5_input_bad_status_blocks_before_contract_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import warranty_analytics_model.splits.input as input_module

    mart_dir = tmp_path / "mart"
    mart_dir.mkdir()
    (mart_dir / "manifest.json").write_text("{}", encoding="utf-8")
    frames = _input_frames()
    monkeypatch.setattr(
        input_module,
        "validate_artifact_integrity",
        lambda path: (
            {"manifest": {"validation_status": "BLOCKED"}, "column_manifest": []},
            frames,
        ),
    )

    with pytest.raises(SplitError, match="blocks Phase 6"):
        load_phase5_mart(mart_dir, project_root=ROOT)


def test_split_settings_fail_closed_for_all_boundary_safety_rules() -> None:
    settings = load_split_settings(ROOT)
    invalid = settings.model_copy(
        update={
            "train_fraction": 0.50,
            "validation_fraction": 0.10,
            "test_fraction": 0.10,
            "strategy": "random",
            "preserve_same_date": False,
            "tie_break": "later_date",
            "min_positive_block_validation": 5,
            "min_positive_warning_validation": 4,
            "min_positive_block_test": 5,
            "min_positive_warning_test": 4,
        }
    )

    errors = validate_split_settings(invalid)

    assert len(errors) == 6
    assert any("sum to 1.0" in error for error in errors)
    assert any("chronological" in error for error in errors)
    assert any("same-date" in error for error in errors)
    assert any("earlier date" in error for error in errors)
    assert any("Validation warning" in error for error in errors)
    assert any("Test warning" in error for error in errors)


def test_split_settings_loader_rejects_missing_malformed_and_invalid_files(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(SplitError, match="missing"):
        load_split_settings(ROOT, path=missing)

    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("split: [", encoding="utf-8")
    with pytest.raises(SplitError, match="Could not read"):
        load_split_settings(ROOT, path=malformed)

    wrong_shape = tmp_path / "wrong-shape.yaml"
    wrong_shape.write_text("other: {}\n", encoding="utf-8")
    with pytest.raises(SplitError, match="top-level split"):
        load_split_settings(ROOT, path=wrong_shape)

    invalid_type = tmp_path / "invalid-type.yaml"
    invalid_type.write_text("split:\n  train_fraction: not-a-number\n", encoding="utf-8")
    with pytest.raises(SplitError, match="Invalid Phase 6"):
        load_split_settings(ROOT, path=invalid_type)

    invalid_semantics = tmp_path / "invalid-semantics.yaml"
    invalid_semantics.write_text(
        "split:\n"
        "  strategy: chronological\n"
        "  train_fraction: 0.50\n"
        "  validation_fraction: 0.10\n"
        "  test_fraction: 0.10\n"
        "  preserve_same_date: true\n"
        "  tie_break: earlier_date\n",
        encoding="utf-8",
    )
    with pytest.raises(SplitError, match="sum to 1.0"):
        load_split_settings(ROOT, path=invalid_semantics)


def test_phase5_artifact_hash_guards_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import warranty_analytics_model.splits.input as input_module

    frame = pd.DataFrame({"warranty_claim_key": [1]})
    with pytest.raises(SplitError, match="missing artifact_file_sha256"):
        _required_artifact_hash({}, "claim_snapshot", hash_name="artifact_file_sha256")
    with pytest.raises(SplitError, match="missing the claim_snapshot path"):
        _verify_artifact_hashes(
            tmp_path,
            {},
            {"claim_snapshot": frame},
            "claim_snapshot",
        )

    missing_file_manifest = {"artifact_paths": {"claim_snapshot": "snapshot.parquet"}}
    with pytest.raises(SplitError, match="artifact is missing"):
        _verify_artifact_hashes(
            tmp_path,
            missing_file_manifest,
            {"claim_snapshot": frame},
            "claim_snapshot",
        )

    snapshot_path = tmp_path / "snapshot.parquet"
    snapshot_path.write_bytes(b"fictional")
    monkeypatch.setattr(input_module, "sha256_file", lambda path: "actual-file")
    monkeypatch.setattr(input_module, "content_sha256", lambda value: "actual-content")
    file_mismatch = {
        "artifact_paths": {"claim_snapshot": "snapshot.parquet"},
        "artifact_file_sha256": {"claim_snapshot": "wrong-file"},
        "artifact_content_fingerprints": {"claim_snapshot": "actual-content"},
    }
    with pytest.raises(SplitError, match="file checksum"):
        _verify_artifact_hashes(
            tmp_path,
            file_mismatch,
            {"claim_snapshot": frame},
            "claim_snapshot",
        )

    content_mismatch = {
        "artifact_paths": {"claim_snapshot": "snapshot.parquet"},
        "artifact_file_sha256": {"claim_snapshot": "actual-file"},
        "artifact_content_fingerprints": {"claim_snapshot": "wrong-content"},
    }
    with pytest.raises(SplitError, match="content checksum"):
        _verify_artifact_hashes(
            tmp_path,
            content_mismatch,
            {"claim_snapshot": frame},
            "claim_snapshot",
        )


def test_phase6_validation_helpers_reject_malformed_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    with pytest.raises(SplitError, match="not valid JSON"):
        _read_json(invalid_json, "invalid.json")

    non_object = tmp_path / "array.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(SplitError, match="JSON object"):
        _read_json(non_object, "array.json")

    assert _date_string(None) == ""
    frame = pd.DataFrame({"warranty_claim_key": [1]})
    assert "artifact is missing" in _artifact_check(tmp_path, {}, "split_assignments", frame)[0]
    (tmp_path / "split_assignments.parquet").write_bytes(b"fictional")
    assert (
        "missing artifact checksums" in _artifact_check(tmp_path, {}, "split_assignments", frame)[0]
    )

    import warranty_analytics_model.splits.validation as validation_module

    captured: list[Path] = []
    sentinel = object()
    monkeypatch.setattr(
        validation_module,
        "load_phase5_mart",
        lambda path, project_root: captured.append(path) or sentinel,
    )
    assert (
        validation_module._load_input_from_manifest({"input_mart_run": "fictional-run"}, tmp_path)
        is sentinel
    )
    assert captured[-1] == tmp_path / "artifacts" / "feature_mart" / "fictional-run"
    assert (
        validation_module._load_input_from_manifest(
            {"input_mart_relative_path": "relative/mart"}, tmp_path
        )
        is sentinel
    )
    assert captured[-1] == tmp_path / "relative" / "mart"
    with pytest.raises(SplitError, match="does not identify"):
        validation_module._load_input_from_manifest({}, tmp_path)
