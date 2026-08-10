"""Fictional, target-free regression tests for Phase 8 text candidates."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from warranty_analytics_model.feature_mart.manifest import content_sha256, sha256_file
from warranty_analytics_model.text_features import config as text_config
from warranty_analytics_model.text_features import contract as text_contract
from warranty_analytics_model.text_features import input as text_input
from warranty_analytics_model.text_features import runner as text_runner
from warranty_analytics_model.text_features import validation as text_validation
from warranty_analytics_model.text_features.config import (
    load_text_feature_settings,
    settings_payload,
)
from warranty_analytics_model.text_features.contract import validate_text_feature_contract
from warranty_analytics_model.text_features.documents import build_historical_documents
from warranty_analytics_model.text_features.lexical import build_lexical_features
from warranty_analytics_model.text_features.manifest import ordered_text_frame
from warranty_analytics_model.text_features.models import (
    Phase8Inputs,
    TextFeatureError,
    TextFeatureSettings,
)
from warranty_analytics_model.text_features.normalize import normalize_description
from warranty_analytics_model.text_features.reporting import write_phase8_reports
from warranty_analytics_model.text_features.source_policy import (
    validate_text_lineage_sources,
    validate_text_source,
)
from warranty_analytics_model.text_features.validation import validate_text_directory


def _inputs(prior: pd.DataFrame) -> Phase8Inputs:
    assignments = pd.DataFrame(
        {
            "warranty_claim_key": [1, 2, 3],
            "claim_date": pd.to_datetime(["2025-01-15", "2025-01-15", "2025-01-16"]),
            "split": ["TRAIN", "VALIDATION", "TEST"],
        }
    )
    empty: dict[str, object] = {}
    return Phase8Inputs(
        root=Path("."),
        mart_dir=Path("mart"),
        split_dir=Path("split"),
        structured_dir=Path("structured"),
        prior_claim_history=prior,
        assignments=assignments,
        phase5_manifest=empty,
        phase6_manifest=empty,
        test_lock=empty,
        phase7_manifest=empty,
        phase7_lineage=empty,
        phase5_validation=empty,
        phase6_validation=empty,
        phase7_validation=empty,
        phase5_contract_checksum="",
        phase6_contract_checksum="",
        phase5_manifest_checksum="",
        phase7_contract_checksum="",
        phase7_content_sha256="",
    )


def _prior() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "current_warranty_claim_key": [1, 1, 1, 3],
            "prior_warranty_claim_key": [20, 10, 30, 40],
            "prior_claim__claim_date": pd.to_datetime(
                ["2024-07-15", "2024-07-15", "2024-01-01", "2025-01-15"]
            ),
            "prior_failure__failure_description": [
                "Brake fault",
                "  Brake\tFAULT\n",
                "older failure",
                "one day before",
            ],
        }
    )


def test_normalization_is_unicode_safe_and_deterministic() -> None:
    assert normalize_description("  E\u0301ngine\tFAULT\nDetected  ") == "éngine fault detected"
    assert normalize_description(None) is None
    assert normalize_description("   \t\n") is None


def test_window_boundary_duplicate_and_no_history_semantics() -> None:
    inputs = _inputs(_prior())
    documents, audit = build_historical_documents(inputs, TextFeatureSettings())
    result = build_lexical_features(documents, TextFeatureSettings())
    claim_one = result.frame.loc[result.frame["warranty_claim_key"] == 1].iloc[0]
    claim_two = result.frame.loc[result.frame["warranty_claim_key"] == 2].iloc[0]
    assert claim_one["prior_failure_text__6m__document"] == "brake fault [SEP] brake fault"
    assert claim_one["prior_failure_text__all__document"] == (
        "older failure [SEP] brake fault [SEP] brake fault"
    )
    assert claim_one["text__6m__description_count"] == 2
    assert claim_one["text__6m__unique_description_count"] == 1
    assert pd.isna(claim_two["prior_failure_text__6m__document"])
    assert claim_two["text__6m__description_count"] == 0
    assert pd.isna(claim_two["text__6m__avg_description_token_count"])
    assert audit["same_day_text_records"] == 0
    assert audit["future_text_records"] == 0


def test_same_date_tie_uses_prior_key_only_for_order() -> None:
    prior = _prior().iloc[[0, 1]].copy()
    prior.loc[prior["prior_warranty_claim_key"] == 20, "prior_failure__failure_description"] = (
        "later key"
    )
    prior.loc[prior["prior_warranty_claim_key"] == 10, "prior_failure__failure_description"] = (
        "earlier key"
    )
    documents, _ = build_historical_documents(_inputs(prior), TextFeatureSettings())
    value = documents.loc[
        documents["warranty_claim_key"] == 1, "prior_failure_text__6m__document"
    ].iloc[0]
    assert value == "earlier key [SEP] later key"


def test_same_day_prior_record_blocks() -> None:
    prior = _prior().copy()
    prior.loc[len(prior)] = [1, 99, pd.Timestamp("2025-01-15"), "same day"]
    with pytest.raises(TextFeatureError, match="same_day=1"):
        build_historical_documents(_inputs(prior), TextFeatureSettings())


def test_shuffling_source_rows_does_not_change_content_hash() -> None:
    settings = TextFeatureSettings()
    first, _ = build_historical_documents(_inputs(_prior()), settings)
    shuffled, _ = build_historical_documents(
        _inputs(_prior().sample(frac=1.0, random_state=7)), settings
    )
    first_frame = build_lexical_features(first, settings).frame
    shuffled_frame = build_lexical_features(shuffled, settings).frame
    assert content_sha256(ordered_text_frame(first_frame)) == content_sha256(
        ordered_text_frame(shuffled_frame)
    )


def test_changing_prior_identifier_without_a_tie_does_not_change_text() -> None:
    first = _prior().iloc[[2, 3]].copy()
    changed = first.copy()
    changed.loc[changed["prior_warranty_claim_key"] == 30, "prior_warranty_claim_key"] = 300
    settings = TextFeatureSettings()
    first_documents, _ = build_historical_documents(_inputs(first), settings)
    changed_documents, _ = build_historical_documents(_inputs(changed), settings)
    assert first_documents.filter(like="document").equals(changed_documents.filter(like="document"))


def test_target_labels_are_irrelevant_and_never_emitted() -> None:
    settings = TextFeatureSettings()
    prior = _prior()
    with_target = prior.assign(target_label=[1, 0, 1, 0])
    first_documents, _ = build_historical_documents(_inputs(prior), settings)
    target_documents, _ = build_historical_documents(_inputs(with_target), settings)
    first_frame = ordered_text_frame(build_lexical_features(first_documents, settings).frame)
    target_frame = ordered_text_frame(build_lexical_features(target_documents, settings).frame)
    pd.testing.assert_frame_equal(first_frame, target_frame)
    assert not any("target" in column.lower() for column in target_frame.columns)


def test_phase8_contract_is_authoritative_and_warning_only() -> None:
    result = validate_text_feature_contract(Path.cwd())
    assert result["valid"] is True
    assert result["status"] == "PASS WITH WARNINGS", result
    assert result["contract_version"] == "1.0.0"
    assert result["contract_checksum"]


def test_phase8_settings_are_versioned_and_deterministic() -> None:
    settings = load_text_feature_settings(Path.cwd())
    payload = settings_payload(settings)
    assert settings.windows_months == (6, 12, 24)
    assert settings.document_separator == " [SEP] "
    assert payload["normalization"]["unicode_form"] == "NFKC"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text_features": {"windows_months": [3]}},
        {"text_features": {"include_all_history": False}},
        {"text_features": {"normalization": []}},
    ],
)
def test_phase8_settings_reject_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, payload: dict[str, object]
) -> None:
    import yaml

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "text_features.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    monkeypatch.setattr(text_config, "discover_repository_root", lambda root: tmp_path)
    with pytest.raises(TextFeatureError):
        text_config.load_text_feature_settings(tmp_path)


def test_phase8_input_helpers_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "valid.json").write_text('{"ok": true}', encoding="utf-8")
    assert text_input._read_json(tmp_path / "valid.json", "valid")["ok"] is True
    with pytest.raises(TextFeatureError):
        text_input._read_json(tmp_path / "missing.json", "missing")
    (tmp_path / "invalid.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(TextFeatureError):
        text_input._read_json(tmp_path / "invalid.json", "invalid")
    (tmp_path / "list.json").write_text("[]", encoding="utf-8")
    with pytest.raises(TextFeatureError):
        text_input._read_json(tmp_path / "list.json", "list")


def test_phase8_input_and_validation_missing_artifacts_block(tmp_path: Path) -> None:
    with pytest.raises(TextFeatureError, match="Phase 5/6 validation blocks"):
        text_input.load_phase8_inputs(
            tmp_path / "mart", tmp_path / "split", tmp_path / "structured", project_root=Path.cwd()
        )
    result = validate_text_directory(tmp_path / "missing", project_root=Path.cwd())
    assert result["status"] == "BLOCKED"


def test_phase8_contract_rejects_unsafe_policy_values(monkeypatch: pytest.MonkeyPatch) -> None:
    import copy

    authoritative, _ = text_contract.load_text_feature_contract(Path.cwd())
    invalid = copy.deepcopy(authoritative)
    invalid.update(
        {
            "contract_version": "9.9.9",
            "feature_grain": "one row = many claims",
            "approved_text_sources": [{"source": "current_failure_description", "policy": "BAD"}],
            "historical_windows": {"fixed_months": ["3m"], "include_all_history": False},
            "document_construction_policy": {
                "separator": "|",
                "preserve_repeated_descriptions": False,
            },
            "text_normalization_policy": {"unicode_form": "NFC"},
            "fitted_transform_policy": {"vectorizer": True},
            "target_independence_policy": {"target_column_excluded": False},
            "test_lock_policy": {"consume_existing_split": False},
            "dimension_versioning_warning": {"code": "MISSING"},
            "development_status": {"production_approved": True},
        }
    )
    phase4 = SimpleNamespace(
        target_checksum=authoritative["target_contract_checksum"],
        feature_policy_checksum=authoritative["feature_policy_checksum"],
        leakage_checksum=authoritative["leakage_policy_checksum"],
    )
    monkeypatch.setattr(text_contract, "load_text_feature_contract", lambda root: (invalid, "bad"))
    monkeypatch.setattr(
        text_contract,
        "load_schema_contract",
        lambda root: ({}, authoritative["schema_contract_checksum"]),
    )
    monkeypatch.setattr(text_contract, "load_phase4_contracts", lambda root: phase4)
    monkeypatch.setattr(
        text_contract,
        "load_mart_contract",
        lambda root: ({}, authoritative["phase5_mart_contract_checksum"]),
    )
    monkeypatch.setattr(
        text_contract,
        "load_split_contract",
        lambda root: ({}, authoritative["phase6_split_contract_checksum"]),
    )
    monkeypatch.setattr(
        text_contract,
        "load_structured_feature_contract",
        lambda root: ({}, authoritative["phase7_structured_feature_contract_checksum"]),
    )
    result = text_contract.validate_text_feature_contract(Path.cwd())
    assert result["valid"] is False
    assert len(result["errors"]) >= 8


def test_phase8_public_checks_delegate_to_validators(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = {"status": "PASS WITH WARNINGS", "valid": True}
    monkeypatch.setattr(text_runner, "validate_text_feature_contract", lambda root: expected)
    monkeypatch.setattr(text_runner, "validate_text_directory", lambda *args, **kwargs: expected)
    assert text_runner.phase8_contract_check(tmp_path) is expected
    assert (
        text_runner.validate_existing_text_run(tmp_path / "run", project_root=tmp_path) is expected
    )


def test_phase8_document_wrapper_delegates() -> None:
    documents, audit = text_runner._build_documents(_inputs(_prior()), TextFeatureSettings())
    assert "prior_failure_text__6m__document" in documents
    assert audit["same_day_text_records"] == 0


def test_phase8_lineage_allowlist_accepts_controls_and_rejects_overlap() -> None:
    valid = {
        "prior_failure_text__6m__document": {
            "is_model_feature": True,
            "value_sources": ["prior_failure__failure_description"],
            "control_sources": ["prior_warranty_claim_key"],
            "source_artifacts": ["prior_claim_history"],
            "target_dependent": False,
            "fitted_transformation": None,
        },
        "warranty_claim_key": {"is_model_feature": False},
    }
    assert validate_text_lineage_sources(valid)["valid"] is True
    invalid = {
        "text__bad": {
            "is_model_feature": True,
            "value_sources": ["prior_failure__failure_description"],
            "control_sources": ["prior_failure__failure_description"],
            "source_artifacts": ["prior_claim_history"],
            "target_dependent": True,
            "fitted_transformation": "fit",
        }
    }
    rejected = validate_text_lineage_sources(invalid)
    assert rejected["valid"] is False
    assert rejected["errors"]


def test_phase8_reports_are_aggregate_only(tmp_path: Path) -> None:
    manifest = {
        "validation_status": "PASS WITH WARNINGS",
        "input_phase5_run": "mart",
        "input_phase6_run": "split",
        "input_phase7_run": "structured",
        "row_count": 3,
        "train_count": 1,
        "validation_count": 1,
        "test_count": 1,
        "text_document_feature_count": 4,
        "lexical_feature_count": 29,
        "text_feature_names": ["prior_failure_text__6m__document"],
    }
    quality = {
        window: {"coverage_percentage": 0.0, "warnings": []}
        for window in ("6m", "12m", "24m", "all")
    }
    validation = {
        "status": "PASS WITH WARNINGS",
        "temporal_audit": {"same_day_text_records": 0, "future_text_records": 0},
        "leakage_audit": {"prohibited_sources": 0},
        "warnings": ["warning"],
        "errors": [],
    }
    paths = write_phase8_reports(
        output_root=tmp_path,
        run_id="run",
        manifest=manifest,
        quality=quality,
        source_coverage={"approved": ["prior_failure__failure_description"]},
        validation=validation,
    )
    assert {path.name for path in paths} == {
        "phase_8_summary.json",
        "phase_8_summary.md",
        "text_feature_inventory.json",
        "text_quality.json",
        "source_coverage.json",
        "validation.json",
    }
    assert "SAFE TO START PHASE 9" in (tmp_path / "run" / "phase_8_summary.md").read_text()


def test_phase8_plan_check_reports_input_population_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_inputs = _inputs(_prior())
    monkeypatch.setattr(
        text_input,
        "validate_text_feature_contract",
        lambda root: {"errors": [], "warnings": [], "valid": True},
    )
    monkeypatch.setattr(text_input, "load_phase8_inputs", lambda *args, **kwargs: fake_inputs)
    result = text_input.phase8_plan_check(Path("mart"), Path("split"), Path("structured"))
    assert result["status"] == "BLOCKED"
    assert any("8,500" in error for error in result["errors"])


def test_phase8_runner_publishes_atomically_without_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = TextFeatureSettings()
    inputs = _inputs(_prior())
    frame = pd.DataFrame(
        {
            "warranty_claim_key": [1],
            "split": ["TRAIN"],
            "claim__claim_date": [pd.Timestamp("2025-01-15")],
            "prior_failure_text__6m__document": pd.Series(["brake fault"], dtype="string"),
        }
    )
    result = SimpleNamespace(
        frame=frame,
        definitions=[
            text_runner.TextFeatureDefinition(
                feature_name="prior_failure_text__6m__document",
                tier="TEXT_EXTENDED",
                feature_type="text",
                source_artifacts=("prior_claim_history",),
                source_columns=("prior_failure__failure_description",),
                value_sources=("prior_failure__failure_description",),
                control_sources=("prior_warranty_claim_key",),
                window="6m",
            )
        ],
        quality={"train_feature_warnings": []},
        source_coverage={"approved": ["prior_failure__failure_description"]},
        warnings=[],
    )

    monkeypatch.setattr(
        text_runner,
        "validate_text_feature_contract",
        lambda root: {"valid": True, "errors": [], "warnings": []},
    )
    monkeypatch.setattr(
        text_runner,
        "phase8_plan_check",
        lambda *args, **kwargs: {
            "valid": True,
            "errors": [],
            "warnings": [],
            "inputs": inputs,
        },
    )
    monkeypatch.setattr(text_runner, "load_text_feature_settings", lambda root: settings)
    monkeypatch.setattr(
        text_runner,
        "_build_documents",
        lambda source_inputs, source_settings: (
            pd.DataFrame({"warranty_claim_key": [1]}),
            {"same_day_text_records": 0, "future_text_records": 0},
        ),
    )
    monkeypatch.setattr(text_runner, "build_lexical_features", lambda documents, config: result)
    monkeypatch.setattr(
        text_runner,
        "load_text_feature_contract",
        lambda root: ({"contract_version": "1.0.0"}, "contract-sha"),
    )
    monkeypatch.setattr(
        text_runner,
        "validate_text_directory",
        lambda *args, **kwargs: {
            "status": "PASS WITH WARNINGS",
            "errors": [],
            "warnings": [],
            "temporal_audit": {},
            "leakage_audit": {},
        },
    )
    monkeypatch.setattr(
        text_runner,
        "write_text_artifact",
        lambda frame, path, compression: {
            "file_sha256": "file-sha",
            "content_sha256": "content-sha",
        },
    )

    def fake_write_json(path: Path, payload: object) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")

    monkeypatch.setattr(text_runner, "write_json", fake_write_json)

    def fake_reports(*, output_root: Path, run_id: str, **kwargs: object) -> list[Path]:
        report_dir = output_root / run_id
        report_dir.mkdir(parents=True, exist_ok=True)
        return []

    monkeypatch.setattr(text_runner, "write_phase8_reports", fake_reports)
    published = text_runner.build_phase8(
        Path("mart"),
        Path("split"),
        Path("structured"),
        output_dir=tmp_path / "artifacts",
        report_dir=tmp_path / "reports",
        run_id="run",
        no_report=False,
        project_root=Path.cwd(),
    )
    assert published["status"] == "PASS WITH WARNINGS"
    assert published["report_directory"] == tmp_path / "reports" / "run"
    assert (tmp_path / "artifacts" / "run" / "manifest.json").is_file()


def test_phase8_validation_accepts_minimal_locked_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json
    from dataclasses import replace

    keys = list(range(1, 8501))
    frame = pd.DataFrame(
        {
            "warranty_claim_key": keys,
            "split": ["TRAIN"] * len(keys),
            "claim__claim_date": pd.date_range("2025-01-01", periods=len(keys), freq="D"),
            "prior_failure_text__6m__document": pd.Series([pd.NA] * len(keys), dtype="string"),
            "text__6m__description_count": [0] * len(keys),
        }
    )
    text_dir = tmp_path / "artifacts" / "text_features" / "run"
    text_dir.mkdir(parents=True)
    parquet_path = text_dir / "text_features.parquet"
    ordered = ordered_text_frame(frame)
    ordered.to_parquet(parquet_path, index=False)
    feature_names = [
        "prior_failure_text__6m__document",
        "text__6m__description_count",
    ]
    lineage = {
        "warranty_claim_key": {
            "is_model_feature": False,
            "is_control": True,
            "target_dependent": False,
            "fitted_transformation": None,
        },
        "split": {
            "is_model_feature": False,
            "is_control": True,
            "target_dependent": False,
            "fitted_transformation": None,
        },
        "claim__claim_date": {
            "is_model_feature": False,
            "is_control": True,
            "target_dependent": False,
            "fitted_transformation": None,
        },
        "prior_failure_text__6m__document": {
            "is_model_feature": True,
            "is_control": False,
            "value_sources": ["prior_failure__failure_description"],
            "control_sources": ["prior_warranty_claim_key"],
            "source_artifacts": ["prior_claim_history"],
            "target_dependent": False,
            "fitted_transformation": None,
        },
        "text__6m__description_count": {
            "is_model_feature": True,
            "is_control": False,
            "value_sources": ["prior_failure__failure_description"],
            "control_sources": ["prior_warranty_claim_key"],
            "source_artifacts": ["prior_claim_history"],
            "target_dependent": False,
            "fitted_transformation": None,
        },
    }
    manifest = {
        "input_phase5_run": "mart",
        "input_phase6_run": "split",
        "input_phase7_run": "structured",
        "text_feature_count": 2,
        "lexical_feature_count": 1,
        "artifact_content_sha256": {"text_features": content_sha256(ordered)},
        "artifact_file_sha256": {"text_features": sha256_file(parquet_path)},
        "split_assignment_sha256": None,
        "train_claim_key_sha256": None,
        "validation_claim_key_sha256": None,
        "test_claim_key_sha256": None,
        "test_lock_hashes": {},
    }
    structured_dir = tmp_path / "artifacts" / "structured_features" / "structured"
    structured_dir.mkdir(parents=True)
    pd.DataFrame({"placeholder": [1]}).to_parquet(
        structured_dir / "structured_features.parquet", index=False
    )
    (structured_dir / "feature_lineage.json").write_text("{}", encoding="utf-8")
    (text_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (text_dir / "text_feature_manifest.json").write_text(
        json.dumps({"feature_names": feature_names}), encoding="utf-8"
    )
    (text_dir / "text_feature_lineage.json").write_text(json.dumps(lineage), encoding="utf-8")
    (text_dir / "text_quality.json").write_text(
        json.dumps({"train_feature_warnings": []}), encoding="utf-8"
    )
    assignments = pd.DataFrame({"warranty_claim_key": keys, "split": ["TRAIN"] * len(keys)})
    inputs = replace(
        _inputs(
            pd.DataFrame(
                {
                    "current_warranty_claim_key": [1],
                    "prior_warranty_claim_key": [2],
                    "prior_claim__claim_date": [pd.Timestamp("2024-01-01")],
                    "prior_failure__failure_description": ["historical description"],
                }
            )
        ),
        root=tmp_path,
        assignments=assignments,
        phase6_manifest={},
        test_lock={},
    )
    monkeypatch.setattr(text_validation, "load_phase8_inputs", lambda *args, **kwargs: inputs)
    monkeypatch.setattr(text_validation, "discover_repository_root", lambda root: tmp_path)
    monkeypatch.setattr(
        text_validation, "load_text_feature_settings", lambda root: TextFeatureSettings()
    )
    monkeypatch.setattr(
        text_validation,
        "build_historical_documents",
        lambda source_inputs, settings: (
            pd.DataFrame({"warranty_claim_key": keys}),
            {"same_day_text_records": 0, "future_text_records": 0},
        ),
    )
    monkeypatch.setattr(
        text_validation,
        "build_lexical_features",
        lambda documents, settings: SimpleNamespace(frame=frame),
    )
    result = validate_text_directory(
        text_dir, project_root=tmp_path, report_dir=tmp_path / "reports"
    )
    assert result["status"] == "PASS WITH WARNINGS", result
    assert result["checks"]["deterministic_rebuild_valid"] is True


@pytest.mark.parametrize(
    "source",
    [
        "complaint_description",
        "diagnostic_summary",
        "technician_notes",
        "repair_notes",
        "current_failure_description",
        "total_claim_cost",
        "warranty_claim_key",
        "supplier_key",
    ],
)
def test_phase8_text_allowlist_rejects_unsafe_sources(source: str) -> None:
    with pytest.raises(ValueError):
        validate_text_source(source, source_artifact="claim_snapshot")
