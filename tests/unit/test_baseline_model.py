"""Fictional regression and safety tests for Phase 9 baseline modeling."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from warranty_analytics_model.baseline_model.adapters import (
    adapt_matrix,
    build_development_feature_frame,
)
from warranty_analytics_model.baseline_model.catboost_baseline import (
    fit_classifier,
    load_model,
    predict_probabilities,
    save_model,
)
from warranty_analytics_model.baseline_model.config import load_baseline_settings
from warranty_analytics_model.baseline_model.contract import validate_baseline_contract
from warranty_analytics_model.baseline_model.feature_sets import (
    audit_phase8_sources,
    resolve_feature_sets,
)
from warranty_analytics_model.baseline_model.metrics import (
    calculate_metrics,
    performance_warnings,
    prevalence_probabilities,
    probability_sha256,
    select_champion,
    validate_probabilities,
)
from warranty_analytics_model.baseline_model.models import (
    BaselineModelError,
    DevelopmentTargets,
    ExperimentResult,
    FeatureSetSpec,
)
from warranty_analytics_model.baseline_model.provenance import (
    MODEL_CORE_PARAMETERS,
    model_policy_errors,
    prediction_content_sha256,
    runtime_provenance,
    runtime_provenance_errors,
    validate_runtime_dependency_constraints,
)
from warranty_analytics_model.baseline_model.target import (
    KEY,
    TARGET,
    load_development_targets,
    target_summary,
)
from warranty_analytics_model.baseline_model.validation import (
    _validate_ap_lift,
    _validate_champion,
    _validate_configuration_policy,
    _validate_experiment_inventory,
    _validate_feature_sets,
    _validate_input_hashes,
    _validate_predictions,
    _validate_target_metadata,
    _validate_test_seal,
    _walk_target_hashes,
    compare_phase9_runs,
)


def _lineage(feature_type: str, *, tier: str | None = None) -> dict[str, object]:
    return {
        "is_model_feature": True,
        "is_control": False,
        "target_dependent": False,
        "feature_type": feature_type,
        "tier": tier,
        "value_sources": ["prior_failure__failure_description"],
        "fitted_transformation": None,
    }


def test_contract_and_fixed_settings_pass_offline() -> None:
    result = validate_baseline_contract(Path.cwd())
    settings = load_baseline_settings(Path.cwd())
    assert result["valid"] is True
    assert settings.catboost_parameters["iterations"] == 500
    assert settings.catboost_parameters["thread_count"] == 1
    assert not {"class_weights", "auto_class_weights", "scale_pos_weight"} & set(
        settings.catboost_parameters
    )


def test_feature_sets_are_lineage_ordered_nested_and_exactly_typed() -> None:
    phase7 = {
        "core_numeric": _lineage("numeric", tier="CORE"),
        "extended_category": _lineage("categorical", tier="EXTENDED"),
        "core_boolean": _lineage("boolean", tier="CORE"),
    }
    phase8 = {"lexical": _lineage("numeric"), "document": _lineage("text")}
    result = resolve_feature_sets(phase7, phase8)
    assert result["E1"].feature_names == ("core_numeric", "core_boolean")
    assert result["E2"].feature_names == tuple(phase7)
    assert result["E3"].feature_names == (*phase7, "lexical")
    assert result["E4"].feature_names == (*phase7, "lexical", "document")
    assert result["E4"].text_features == ("document",)


def test_every_phase8_value_source_and_vocabulary_policy_are_checked() -> None:
    lineage = {"safe": _lineage("text"), "unsafe": _lineage("numeric")}
    lineage["unsafe"]["value_sources"] = ["current_claim__complaint"]
    policy = {
        "tfidf": False,
        "count_vectorizer": False,
        "embeddings": False,
        "llm": False,
        "vocabulary_learning": True,
        "model_training": False,
    }
    result = audit_phase8_sources(lineage, {"fitted_transform_policy": policy})
    assert result["model_candidate_count"] == 2
    assert result["value_source_entry_count"] == 2
    assert result["unauthorized_value_source_count"] == 1
    assert result["vocabulary_learning"] is True
    assert result["valid"] is False


def test_adapters_preserve_numeric_missing_and_use_fixed_sentinels() -> None:
    settings = load_baseline_settings(Path.cwd())
    spec = FeatureSetSpec(
        "E4",
        ("number", "category", "flag", "document"),
        ("number",),
        ("category",),
        ("flag",),
        ("document",),
        1,
        0,
        1,
        1,
        "hash",
    )
    frame = pd.DataFrame(
        {
            "number": [1.0, np.nan],
            "category": [None, "A"],
            "flag": [True, False],
            "document": [None, "history"],
        }
    )
    matrix = adapt_matrix(frame, spec, settings)
    assert np.isnan(matrix.loc[1, "number"])
    assert matrix.loc[0, "category"] == "__MISSING__"
    assert matrix["flag"].tolist() == [1.0, 0.0]
    assert matrix.loc[0, "document"] == "__NO_HISTORY__"


def test_metrics_use_fixed_threshold_and_e0_train_prevalence() -> None:
    y_train = np.array([0, 0, 0, 1])
    y_validation = np.array([0, 1, 0, 1])
    probabilities = prevalence_probabilities(y_train, len(y_validation))
    metrics = calculate_metrics(y_validation, probabilities)
    assert np.all(probabilities == 0.25)
    assert metrics["threshold"] == 0.5
    assert (metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]) == (0, 0, 2, 2)
    with pytest.raises(BaselineModelError, match="0.5"):
        calculate_metrics(y_validation, probabilities, threshold=0.4)


def test_champion_tie_break_prefers_simpler_experiment() -> None:
    metrics = {"average_precision": 0.7, "roc_auc": 0.8, "log_loss": 0.2}
    results = [
        ExperimentResult(experiment_id, "CatBoostClassifier", "SUCCESS", None, dict(metrics), None)
        for experiment_id in ("E4", "E2", "E1", "E3")
    ]
    assert select_champion(results).experiment_id == "E1"


@pytest.mark.parametrize(
    ("metrics_by_id", "expected"),
    [
        (
            {
                "E1": {"average_precision": 0.7, "roc_auc": 0.7, "log_loss": 0.2},
                "E2": {"average_precision": 0.7, "roc_auc": 0.8, "log_loss": 0.3},
                "E3": {"average_precision": 0.6, "roc_auc": 0.9, "log_loss": 0.1},
                "E4": {"average_precision": 0.5, "roc_auc": 0.9, "log_loss": 0.1},
            },
            "E2",
        ),
        (
            {
                "E1": {"average_precision": 0.7, "roc_auc": 0.8, "log_loss": 0.3},
                "E2": {"average_precision": 0.7, "roc_auc": 0.8, "log_loss": 0.2},
                "E3": {"average_precision": 0.6, "roc_auc": 0.9, "log_loss": 0.1},
                "E4": {"average_precision": 0.5, "roc_auc": 0.9, "log_loss": 0.1},
            },
            "E2",
        ),
    ],
)
def test_champion_ties_apply_roc_and_logloss_breaks(
    metrics_by_id: dict[str, dict[str, float]], expected: str
) -> None:
    results = [
        ExperimentResult(experiment_id, "CatBoostClassifier", "SUCCESS", None, metrics, None)
        for experiment_id, metrics in metrics_by_id.items()
    ]
    assert select_champion(results).experiment_id == expected


def test_target_loader_materializes_only_train_and_validation(tmp_path: Path) -> None:
    snapshot = pd.DataFrame({KEY: [1, 2, 3, 4, 5, 6], TARGET: [0, 1, 0, 1, 0, 1]})
    path = tmp_path / "claim_snapshot.parquet"
    snapshot.to_parquet(path, index=False)
    assignments = pd.DataFrame(
        {
            KEY: [1, 2, 3, 4, 5, 6],
            "split": ["TRAIN", "TRAIN", "VALIDATION", "VALIDATION", "TEST", "TEST"],
        }
    )
    targets = load_development_targets(path, assignments)
    summary = target_summary(targets)
    assert targets.audit["test_target_rows_loaded"] == 0
    assert targets.audit["test_predictions_created"] == 0
    assert set(targets.train[KEY]) == {1, 2}
    assert set(targets.validation[KEY]) == {3, 4}
    assert "test" not in summary


def _write_target_fixture(tmp_path: Path, values: list[object]) -> tuple[Path, pd.DataFrame]:
    snapshot = pd.DataFrame({KEY: [1, 2, 3, 4, 5, 6], TARGET: values})
    if any(isinstance(value, str) for value in values):
        snapshot[TARGET] = snapshot[TARGET].astype("string")
    path = tmp_path / "claim_snapshot.parquet"
    snapshot.to_parquet(path, index=False)
    assignments = pd.DataFrame(
        {
            KEY: [1, 2, 3, 4, 5, 6],
            "split": ["TRAIN", "TRAIN", "VALIDATION", "VALIDATION", "TEST", "TEST"],
        }
    )
    return path, assignments


@pytest.mark.parametrize(
    "invalid", [0.5, -1, 2, float("inf"), float("-inf"), np.nan, "not-a-number"]
)
def test_target_loader_rejects_non_exact_binary_values(tmp_path: Path, invalid: object) -> None:
    path, assignments = _write_target_fixture(tmp_path, [0, 1, 0, invalid, 0, 1])
    with pytest.raises(BaselineModelError, match="exact binary|non-numeric|NULL"):
        load_development_targets(path, assignments)


def test_target_loader_accepts_binary_float_labels_without_truncation(tmp_path: Path) -> None:
    path, assignments = _write_target_fixture(tmp_path, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    targets = load_development_targets(path, assignments)
    assert targets.train[TARGET].tolist() == [0, 1]
    assert targets.validation[TARGET].tolist() == [0, 1]


def test_small_catboost_fit_is_repeatable_and_reload_safe(tmp_path: Path) -> None:
    base = load_baseline_settings(Path.cwd())
    settings = replace(base, catboost_parameters={**base.catboost_parameters, "iterations": 8})
    spec = FeatureSetSpec(
        "E1", ("number", "category"), ("number",), ("category",), (), (), 2, 0, 0, 0, "hash"
    )
    matrix = pd.DataFrame({"number": np.arange(40, dtype=float), "category": ["a", "b"] * 20})
    target = np.array([0, 1] * 20, dtype="int8")
    first = fit_classifier(matrix, target, spec, settings)
    second = fit_classifier(matrix, target, spec, settings)
    first_probabilities = predict_probabilities(first, matrix, spec)
    second_probabilities = predict_probabilities(second, matrix, spec)
    assert np.allclose(first_probabilities, second_probabilities, rtol=0.0, atol=1.0e-12)
    path = tmp_path / "model.cbm"
    save_model(first, path)
    reloaded = predict_probabilities(load_model(path), matrix, spec)
    assert probability_sha256(first_probabilities) == probability_sha256(reloaded)


def test_frozen_membership_and_feature_membership_audits() -> None:
    from warranty_analytics_model.baseline_model import input as phase9_input
    from warranty_analytics_model.splits.manifest import (
        assignment_content_sha256,
        claim_key_sha256,
        unordered_claim_key_sha256,
    )

    assignments = pd.DataFrame(
        {
            KEY: [1, 2, 3],
            "claim_date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]),
            "split": ["TRAIN", "VALIDATION", "TEST"],
        }
    )
    manifest = {
        "split_assignment_sha256": assignment_content_sha256(assignments),
        "train_claim_key_sha256": claim_key_sha256(assignments.iloc[[0]]),
        "validation_claim_key_sha256": claim_key_sha256(assignments.iloc[[1]]),
        "test_claim_key_sha256": claim_key_sha256(assignments.iloc[[2]]),
    }
    test = assignments.iloc[[2]]
    lock = {
        "ordered_test_claim_keys_sha256": claim_key_sha256(test),
        "unordered_test_claim_keys_sha256": unordered_claim_key_sha256(test),
        "test_assignment_content_sha256": assignment_content_sha256(test),
    }
    frozen = phase9_input._verify_frozen_membership(assignments, manifest, lock)
    assert frozen["valid"] is True
    assert frozen["counts"] == {"TRAIN": 1, "VALIDATION": 1, "TEST": 1}
    features = assignments.assign(feature=[1.0, 2.0, 3.0])
    assert phase9_input._validate_feature_membership(assignments, features, "fictional") == []
    changed = features.copy()
    changed.loc[2, "split"] = "TRAIN"
    assert phase9_input._validate_feature_membership(assignments, changed, "fictional")


def test_phase7_source_audit_rejects_target_and_control_sources() -> None:
    from warranty_analytics_model.baseline_model import input as phase9_input

    safe = _lineage("numeric", tier="CORE")
    safe["phase4_source_policy"] = "ALLOW_BASELINE_POC"
    safe["value_sources"] = ["truck__model_year"]
    unsafe = dict(safe)
    unsafe.update(
        {
            "is_control": True,
            "target_dependent": True,
            "value_sources": ["claim__total_claim_cost"],
        }
    )
    result = phase9_input._audit_phase7_sources({"safe": safe, "unsafe": unsafe})
    assert result["model_feature_count"] == 2
    assert result["prohibited_source_count"] == 1
    assert result["target_dependent_feature_count"] == 1
    assert result["raw_identifier_feature_count"] == 1
    assert result["valid"] is False


def test_experiment_runner_uses_train_target_and_validation_rows_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from warranty_analytics_model.baseline_model import experiments

    settings = load_baseline_settings(Path.cwd())
    specs = {
        experiment_id: FeatureSetSpec(
            experiment_id, ("number",), ("number",), (), (), (), 1, 0, 0, 0, experiment_id
        )
        for experiment_id in ("E1", "E2", "E3", "E4")
    }
    development = pd.DataFrame(
        {
            KEY: [1, 2, 3, 4],
            "split": ["TRAIN", "TRAIN", "VALIDATION", "VALIDATION"],
            "number": [0.0, 1.0, 2.0, 3.0],
        }
    )
    targets = DevelopmentTargets(
        train=pd.DataFrame({KEY: [1, 2], TARGET: [0, 1]}),
        validation=pd.DataFrame({KEY: [3, 4], TARGET: [0, 1]}),
        train_target_content_sha256="train",
        validation_target_content_sha256="validation",
        audit={},
    )
    fake_model = SimpleNamespace()
    monkeypatch.setattr(experiments, "fit_classifier", lambda *args: fake_model)
    monkeypatch.setattr(experiments, "predict_probabilities", lambda *args: np.array([0.2, 0.8]))
    monkeypatch.setattr(experiments, "effective_parameters", lambda model: {"fixed": True})

    def fake_save(_model: object, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fictional-model")

    monkeypatch.setattr(experiments, "save_model", fake_save)
    results = experiments.run_experiments(development, targets, specs, settings, tmp_path)
    assert [item.experiment_id for item in results] == ["E0", "E1", "E2", "E3", "E4"]
    assert all(item.status == "SUCCESS" for item in results)
    assert all(len(item.probabilities) == 2 for item in results if item.probabilities is not None)


@pytest.mark.parametrize("comparison_valid", [True, False])
def test_phase9_runner_blocks_failed_comparison_and_accepts_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, comparison_valid: bool
) -> None:
    from warranty_analytics_model.baseline_model import runner

    base = load_baseline_settings(Path.cwd())
    settings = replace(
        base,
        output_directory=str(tmp_path / "models"),
        report_directory=str(tmp_path / "reports"),
    )
    specs = {
        experiment_id: FeatureSetSpec(
            experiment_id, ("number",), ("number",), (), (), (), 1, 0, 0, 0, experiment_id
        )
        for experiment_id in ("E1", "E2", "E3", "E4")
    }
    targets = DevelopmentTargets(
        pd.DataFrame({KEY: [1, 2], TARGET: [0, 1]}),
        pd.DataFrame({KEY: [3, 4], TARGET: [0, 1]}),
        "train-hash",
        "validation-hash",
        {
            "test_target_rows_loaded": 0,
            "test_predictions_created": 0,
            "test_metrics_computed": False,
            "test_target_access_allowed": False,
            "first_allowed_test_target_phase": 15,
        },
    )
    inputs = SimpleNamespace(
        mart_dir=tmp_path / "mart",
        split_dir=tmp_path / "split",
        structured_dir=tmp_path / "structured",
        text_dir=tmp_path / "text",
        assignments=pd.DataFrame(
            {KEY: [1, 2, 3, 4, 5], "split": ["TRAIN", "TRAIN", "VALIDATION", "VALIDATION", "TEST"]}
        ),
        phase5_manifest={"artifact_content_fingerprints": {"claim_snapshot": "p5"}},
        phase7_manifest={"artifact_content_sha256": {"structured_features": "p7"}},
        phase8_manifest={"artifact_content_sha256": {"text_features": "p8"}},
        frozen_membership={
            "split_assignment_sha256": "p6",
            "counts": {"TRAIN": 2, "VALIDATION": 2, "TEST": 1},
        },
        source_audit={"valid": True},
    )
    development = pd.DataFrame(
        {
            KEY: [1, 2, 3, 4],
            "split": ["TRAIN", "TRAIN", "VALIDATION", "VALIDATION"],
            "number": [0.0, 1.0, 2.0, 3.0],
        }
    )
    base_metrics = calculate_metrics(np.array([0, 1]), np.array([0.2, 0.8]))
    results = [
        ExperimentResult(
            "E0",
            "constant",
            "SUCCESS",
            None,
            dict(base_metrics),
            pd.Series([0.5, 0.5]),
            validation_probability_sha256=probability_sha256(np.array([0.5, 0.5])),
        ),
        ExperimentResult(
            "E1",
            "CatBoostClassifier",
            "SUCCESS",
            specs["E1"],
            dict(base_metrics),
            pd.Series([0.2, 0.8]),
            model_file="models/e1.cbm",
            model_sha256="hash",
            validation_probability_sha256=probability_sha256(np.array([0.2, 0.8])),
        ),
    ]
    monkeypatch.setattr(runner, "discover_repository_root", lambda root: tmp_path)
    monkeypatch.setattr(runner, "load_baseline_settings", lambda root: settings)
    monkeypatch.setattr(
        runner,
        "phase9_plan_check",
        lambda *args, **kwargs: {
            "valid": True,
            "errors": [],
            "warnings": ["POC"],
            "inputs": inputs,
            "feature_sets": specs,
        },
    )
    monkeypatch.setattr(runner, "load_development_targets", lambda *args: targets)
    monkeypatch.setattr(runner, "build_development_feature_frame", lambda *args: development)
    monkeypatch.setattr(runner, "run_experiments", lambda *args: results)
    monkeypatch.setattr(runner, "load_baseline_contract", lambda root: ({}, "contract-hash"))
    monkeypatch.setattr(
        runner,
        "validate_model_directory",
        lambda *args, **kwargs: {"status": "PASS", "valid": True, "errors": [], "warnings": []},
    )
    monkeypatch.setattr(runner, "git_commit_sha", lambda root: "commit")
    monkeypatch.setattr(
        runner,
        "validate_runtime_dependency_constraints",
        lambda *args: {
            "status": "PASS",
            "valid": True,
            "errors": [],
            "checked_requirements": {},
        },
    )
    comparison_source = tmp_path / "models" / "20260810T_PHASE9"
    comparison_source.mkdir(parents=True)
    comparison = {
        "valid": comparison_valid,
        "status": "PASS" if comparison_valid else "BLOCKED",
        "errors": [] if comparison_valid else ["validation prediction hash changed"],
    }
    monkeypatch.setattr(runner, "compare_phase9_runs", lambda *args: comparison)
    if not comparison_valid:
        with pytest.raises(
            BaselineModelError,
            match="Phase 9 before/after semantic comparison failed: "
            "validation prediction hash changed",
        ):
            runner.build_phase9(Path("p5"), Path("p6"), Path("p7"), Path("p8"), run_id="fictional")
        assert not (tmp_path / "models" / "fictional").exists()
        return
    result = runner.build_phase9(Path("p5"), Path("p6"), Path("p7"), Path("p8"), run_id="fictional")
    run_dir = Path(result["run_directory"])
    assert result["champion_experiment_id"] == "E1"
    assert (run_dir / "validation_predictions.parquet").is_file()
    predictions = pd.read_parquet(run_dir / "validation_predictions.parquet")
    assert list(predictions.columns) == [KEY, "experiment_id", "probability"]
    assert set(predictions[KEY]) == {3, 4}
    assert (tmp_path / "reports" / "fictional" / "baseline_model_report.md").is_file()
    assert result["comparison"] == comparison


def test_phase9_validator_blocks_missing_artifacts(tmp_path: Path) -> None:
    from warranty_analytics_model.baseline_model.validation import validate_model_directory

    result = validate_model_directory(tmp_path)
    assert result["valid"] is False
    assert "Missing artifacts" in result["errors"][0]


def test_load_phase9_inputs_rechecks_hashes_sources_and_membership(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json

    from warranty_analytics_model.baseline_model import input as phase9_input
    from warranty_analytics_model.splits.manifest import (
        assignment_content_sha256,
        claim_key_sha256,
        unordered_claim_key_sha256,
    )

    mart = tmp_path / "p5"
    split = tmp_path / "p6"
    structured = tmp_path / "p7"
    text = tmp_path / "p8"
    structured.mkdir()
    text.mkdir()
    assignments = pd.DataFrame(
        {
            KEY: [1, 2, 3],
            "claim_date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]),
            "split": ["TRAIN", "VALIDATION", "TEST"],
        }
    )
    structured_frame = assignments[[KEY, "split"]].assign(core=[1.0, 2.0, 3.0])
    text_frame = assignments[[KEY, "split"]].assign(lexical=[1.0, 0.0, 1.0])
    structured_frame.to_parquet(structured / "structured_features.parquet", index=False)
    text_frame.to_parquet(text / "text_features.parquet", index=False)
    phase7_lineage = {
        "core": {**_lineage("numeric", tier="CORE"), "phase4_source_policy": "ALLOW_BASELINE_POC"}
    }
    phase7_lineage["core"]["value_sources"] = ["truck__model_year"]
    phase8_lineage = {"lexical": _lineage("numeric")}
    (text / "manifest.json").write_text(
        json.dumps(
            {
                "input_phase5_run": mart.name,
                "input_phase6_run": split.name,
                "input_phase7_run": structured.name,
                "artifact_content_sha256": {
                    "text_features": phase9_input.EXPECTED_PHASE8_CONTENT_SHA256
                },
            }
        ),
        encoding="utf-8",
    )
    (text / "text_feature_lineage.json").write_text(json.dumps(phase8_lineage), encoding="utf-8")
    test_rows = assignments.loc[assignments["split"] == "TEST"]
    split_manifest = {
        "split_assignment_sha256": assignment_content_sha256(assignments),
        "train_claim_key_sha256": claim_key_sha256(assignments.iloc[[0]]),
        "validation_claim_key_sha256": claim_key_sha256(assignments.iloc[[1]]),
        "test_claim_key_sha256": claim_key_sha256(test_rows),
    }
    lock = {
        "ordered_test_claim_keys_sha256": claim_key_sha256(test_rows),
        "unordered_test_claim_keys_sha256": unordered_claim_key_sha256(test_rows),
        "test_assignment_content_sha256": assignment_content_sha256(test_rows),
    }
    upstream = SimpleNamespace(
        mart_dir=mart,
        split_dir=split,
        structured_dir=structured,
        assignments=assignments,
        phase7_lineage=phase7_lineage,
        phase7_manifest={
            "artifact_content_sha256": {
                "structured_features": phase9_input.EXPECTED_PHASE7_CONTENT_SHA256
            }
        },
        phase6_manifest=split_manifest,
        test_lock=lock,
        phase5_manifest={},
        phase5_validation={"warnings": []},
        phase6_validation={"warnings": []},
        phase7_validation={"warnings": []},
    )
    monkeypatch.setattr(phase9_input, "discover_repository_root", lambda root: tmp_path)
    monkeypatch.setattr(phase9_input, "load_phase8_inputs", lambda *args, **kwargs: upstream)
    monkeypatch.setattr(
        phase9_input,
        "validate_text_directory",
        lambda *args, **kwargs: {"errors": [], "warnings": []},
    )
    policy = {
        key: False
        for key in (
            "tfidf",
            "count_vectorizer",
            "embeddings",
            "llm",
            "vocabulary_learning",
            "model_training",
        )
    }
    monkeypatch.setattr(
        phase9_input,
        "load_text_feature_contract",
        lambda root: (
            {"fitted_transform_policy": policy},
            phase9_input.EXPECTED_PHASE8_CONTRACT_SHA256,
        ),
    )
    result = phase9_input.load_phase9_inputs(mart, split, structured, text, project_root=tmp_path)
    assert result.frozen_membership["total_count"] == 3
    assert result.source_audit["unauthorized_value_source_count"] == 0
    assert result.source_audit["phase7"]["prohibited_source_count"] == 0


def test_validation_reloads_models_and_recomputes_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json

    from warranty_analytics_model.baseline_model import validation as phase9_validation
    from warranty_analytics_model.feature_mart.manifest import sha256_file

    settings = load_baseline_settings(Path.cwd())
    spec = FeatureSetSpec("E1", ("number",), ("number",), (), (), (), 1, 0, 0, 0, "feature-hash")
    development = pd.DataFrame(
        {
            KEY: [1, 2, 3, 4],
            "split": ["TRAIN", "TRAIN", "VALIDATION", "VALIDATION"],
            "number": [0.0, 1.0, 2.0, 3.0],
        }
    )
    assignments = pd.concat(
        [development[[KEY, "split"]], pd.DataFrame({KEY: [5], "split": ["TEST"]})],
        ignore_index=True,
    )
    inputs = SimpleNamespace(
        mart_dir=tmp_path,
        assignments=assignments,
        phase7_lineage={},
        phase8_lineage={},
        phase5_manifest={"artifact_content_fingerprints": {"claim_snapshot": "p5"}},
        phase7_manifest={"artifact_content_sha256": {"structured_features": "p7"}},
        phase8_manifest={"artifact_content_sha256": {"text_features": "p8"}},
        frozen_membership={"split_assignment_sha256": "p6"},
    )
    targets = DevelopmentTargets(
        pd.DataFrame({KEY: [1, 2], TARGET: [0, 1]}),
        pd.DataFrame({KEY: [3, 4], TARGET: [0, 1]}),
        "train",
        "validation",
        {},
    )
    e0_probability = np.array([0.5, 0.5])
    e1_probability = np.array([0.2, 0.8])
    predictions = pd.concat(
        [
            pd.DataFrame({KEY: [3, 4], "experiment_id": "E0", "probability": e0_probability}),
            pd.DataFrame({KEY: [3, 4], "experiment_id": "E1", "probability": e1_probability}),
        ],
        ignore_index=True,
    )
    predictions.to_parquet(tmp_path / "validation_predictions.parquet", index=False)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_file = model_dir / "e1.cbm"
    model_file.write_bytes(b"fictional")
    metrics_payload = {
        "experiments": {
            "E0": {
                "status": "SUCCESS",
                "metrics": calculate_metrics(np.array([0, 1]), e0_probability),
                "validation_probability_sha256": probability_sha256(e0_probability),
            },
            "E1": {
                "status": "SUCCESS",
                "metrics": calculate_metrics(np.array([0, 1]), e1_probability),
                "validation_probability_sha256": probability_sha256(e1_probability),
            },
        }
    }
    payloads = {
        "experiment_manifest.json": {
            "input_directories": {},
            "settings": {},
            "input_hashes": {
                "phase5_claim_snapshot": "p5",
                "phase6_split_assignment": "p6",
                "phase7_structured_features": "p7",
                "phase8_text_features": "p8",
            },
            "artifact_file_sha256": {},
        },
        "validation_metrics.json": metrics_payload,
        "feature_sets.json": {},
        "model_input_schema.json": {},
        "target_access_audit.json": {
            "test_target_rows_loaded": 0,
            "test_predictions_created": 0,
            "test_metrics_computed": False,
            "test_target_access_allowed": False,
            "first_allowed_test_target_phase": 15,
        },
        "model_manifest.json": {
            "models": {
                "E1": {"model_file": "models/e1.cbm", "model_sha256": sha256_file(model_file)}
            }
        },
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(phase9_validation, "resolve_feature_sets", lambda *args: {"E1": spec})
    monkeypatch.setattr(phase9_validation, "load_baseline_settings", lambda root: settings)
    monkeypatch.setattr(phase9_validation, "load_development_targets", lambda *args: targets)
    monkeypatch.setattr(
        phase9_validation, "build_development_feature_frame", lambda *args: development
    )
    monkeypatch.setattr(phase9_validation, "load_model", lambda path: object())
    monkeypatch.setattr(phase9_validation, "predict_probabilities", lambda *args: e1_probability)
    result = phase9_validation.validate_model_directory(tmp_path, inputs=inputs)
    assert result["valid"] is True, result
    assert result["metrics_recomputed"] is True
    manifest_path = tmp_path / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_file_sha256"] = {"validation_metrics.json": "incorrect"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    changed = phase9_validation.validate_model_directory(tmp_path, inputs=inputs)
    assert changed["valid"] is False
    assert any("Phase 9 artifact file hash differs" in error for error in changed["errors"])


def test_development_join_excludes_test_and_is_test_mutation_invariant() -> None:
    specs = {
        "E4": FeatureSetSpec(
            "E4",
            ("structured", "lexical"),
            ("structured", "lexical"),
            (),
            (),
            (),
            1,
            0,
            1,
            0,
            "hash",
        )
    }
    assignments = pd.DataFrame({KEY: [1, 2, 3], "split": ["TRAIN", "VALIDATION", "TEST"]})
    structured = pd.DataFrame({KEY: [1, 2, 3], "structured": [10.0, 20.0, 30.0]})
    text = pd.DataFrame({KEY: [1, 2, 3], "lexical": [1.0, 2.0, 3.0]})
    inputs = SimpleNamespace(
        assignments=assignments,
        structured_features=structured,
        text_features=text,
        phase7_lineage={"structured": {}},
    )
    first = build_development_feature_frame(inputs, specs)
    inputs.text_features.loc[inputs.text_features[KEY] == 3, "lexical"] = 999999.0
    second = build_development_feature_frame(inputs, specs)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["split"]) == {"TRAIN", "VALIDATION"}
    assert 3 not in set(first[KEY])


def test_adapter_and_probability_guards_fail_closed() -> None:
    settings = load_baseline_settings(Path.cwd())
    boolean_spec = FeatureSetSpec("E1", ("flag",), (), (), ("flag",), (), 1, 0, 0, 0, "hash")
    with pytest.raises(BaselineModelError, match="non-binary"):
        adapt_matrix(pd.DataFrame({"flag": [2]}), boolean_spec, settings)
    with pytest.raises(BaselineModelError, match="missing features"):
        adapt_matrix(pd.DataFrame({"other": [1]}), boolean_spec, settings)
    for bad in (np.array([[0.5]]), np.array([np.nan]), np.array([-0.1]), np.array([1.1])):
        with pytest.raises(BaselineModelError):
            validate_probabilities(bad)


def test_champion_and_performance_warning_guards() -> None:
    with pytest.raises(BaselineModelError, match="No trained"):
        select_champion([ExperimentResult("E0", "constant", "SUCCESS", None, {}, None)])
    champion = ExperimentResult(
        "E1",
        "CatBoostClassifier",
        "SUCCESS",
        None,
        {"average_precision": 0.95, "roc_auc": 0.99, "log_loss": 0.1},
        None,
    )
    warnings = performance_warnings(champion, {"average_precision": 0.96})
    assert set(warnings) == {"WEAK_BASELINE_SIGNAL", "SUSPICIOUSLY_HIGH_BASELINE_PERFORMANCE"}
    champion.metrics = {"average_precision": 0.2, "roc_auc": 0.4, "log_loss": 0.5}
    assert performance_warnings(champion, {"average_precision": 0.1}) == ["NO_RANKING_SIGNAL"]


def test_feature_resolution_rejects_unsafe_metadata() -> None:
    invalid_type = _lineage("unsupported", tier="CORE")
    with pytest.raises(BaselineModelError, match="Unsupported"):
        resolve_feature_sets(
            {"bad": invalid_type}, {"lex": _lineage("numeric"), "doc": _lineage("text")}
        )
    missing_tier = _lineage("numeric")
    with pytest.raises(BaselineModelError, match="CORE or EXTENDED"):
        resolve_feature_sets(
            {"bad": missing_tier}, {"lex": _lineage("numeric"), "doc": _lineage("text")}
        )
    unsafe = _lineage("numeric", tier="CORE")
    unsafe["target_dependent"] = True
    with pytest.raises(BaselineModelError, match="Unsafe"):
        resolve_feature_sets({"bad": unsafe}, {"lex": _lineage("numeric"), "doc": _lineage("text")})


def test_phase9_plan_check_reports_input_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from warranty_analytics_model.baseline_model import input as phase9_input

    monkeypatch.setattr(
        phase9_input,
        "validate_baseline_contract",
        lambda root: {"errors": [], "warnings": ["contract-warning"]},
    )
    monkeypatch.setattr(
        phase9_input,
        "load_phase9_inputs",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            BaselineModelError("fictional input failure")
        ),
    )
    result = phase9_input.phase9_plan_check(
        Path("p5"), Path("p6"), Path("p7"), Path("p8"), project_root=Path.cwd()
    )
    assert result["valid"] is False
    assert "fictional input failure" in result["errors"]


def test_small_public_helpers_and_single_class_metric_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from warranty_analytics_model.baseline_model import runner
    from warranty_analytics_model.baseline_model.catboost_baseline import effective_parameters

    with pytest.raises(BaselineModelError, match="both binary classes"):
        calculate_metrics(np.array([0, 0]), np.array([0.1, 0.2]))
    assert effective_parameters(SimpleNamespace(get_all_params=lambda: {"depth": 6})) == {
        "depth": 6
    }
    monkeypatch.setattr(
        runner,
        "validate_baseline_contract",
        lambda root: {"valid": True},
    )
    assert runner.phase9_contract_check(tmp_path) == {"valid": True}
    monkeypatch.setattr(
        runner,
        "validate_model_directory",
        lambda path, project_root=None: {"valid": True},
    )
    assert runner.validate_existing_model_run(tmp_path)["valid"] is True
    assert runner.phase9_run_id().endswith("Z")


def test_persisted_target_hashes_must_match_fresh_train_and_validation_hashes(
    tmp_path: Path,
) -> None:
    train = pd.DataFrame({KEY: [1, 2], TARGET: [0, 1]})
    validation = pd.DataFrame({KEY: [3, 4], TARGET: [0, 1]})
    targets = SimpleNamespace(
        train=train,
        validation=validation,
        train_target_content_sha256="train-digest",
        validation_target_content_sha256="validation-digest",
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps({"target_hashes": {"train": "train-digest", "validation": "validation-digest"}}),
        encoding="utf-8",
    )
    audit = {
        "train": {"rows": 2, "positive_count": 1, "negative_count": 1},
        "validation": {"rows": 2, "positive_count": 1, "negative_count": 1},
    }
    errors: list[str] = []
    _validate_target_metadata(
        tmp_path,
        audit,
        {"target_summary": {"train": {}, "validation": {}}},
        targets,
        hardened=True,
        errors=errors,
        warnings=[],
    )
    assert errors == []
    (tmp_path / "metadata.json").write_text(
        json.dumps({"target_hashes": {"train": "wrong", "validation": "validation-digest"}}),
        encoding="utf-8",
    )
    errors = []
    _validate_target_metadata(
        tmp_path,
        audit,
        {"target_summary": {"train": {}, "validation": {}}},
        targets,
        hardened=True,
        errors=errors,
        warnings=[],
    )
    assert any("TRAIN" in error for error in errors)


def test_test_target_hash_is_rejected_recursively() -> None:
    found, errors = _walk_target_hashes(
        {"nested": [{"test_target_content_sha256": "must-not-exist"}]}
    )
    assert found == []
    assert any("TEST target hash" in error for error in errors)


def _inventory_fixture(
    tmp_path: Path, *, e4_status: str
) -> tuple[dict[str, object], dict[str, object]]:
    models: dict[str, object] = {}
    for experiment_id in ("E1", "E2", "E3"):
        path = tmp_path / "models" / f"{experiment_id.casefold()}.cbm"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(experiment_id.encode("ascii"))
        from warranty_analytics_model.feature_mart.manifest import sha256_file

        models[experiment_id] = {
            "model_file": f"models/{experiment_id.casefold()}.cbm",
            "model_sha256": sha256_file(path),
        }
    if e4_status == "SUCCESS":
        path = tmp_path / "models" / "e4.cbm"
        path.write_bytes(b"E4")
        from warranty_analytics_model.feature_mart.manifest import sha256_file

        models["E4"] = {"model_file": "models/e4.cbm", "model_sha256": sha256_file(path)}
    experiments = {
        experiment_id: {"status": "SUCCESS", "metrics": {}}
        for experiment_id in ("E0", "E1", "E2", "E3")
    }
    experiments["E4"] = {"status": e4_status, "metrics": {}}
    return {"experiments": experiments}, {"models": models}


def test_experiment_inventory_allows_unavailable_e4_but_rejects_missing_and_extra_ids(
    tmp_path: Path,
) -> None:
    metrics, model_manifest = _inventory_fixture(tmp_path, e4_status="UNAVAILABLE_WITH_WARNING")
    errors: list[str] = []
    statuses = _validate_experiment_inventory(
        metrics,
        model_manifest,
        tmp_path,
        hardened=True,
        errors=errors,
        warnings=[],
    )
    assert not errors
    assert statuses["E4"]["status"] == "UNAVAILABLE_WITH_WARNING"
    missing_e4 = {"experiments": dict(metrics["experiments"])}
    del missing_e4["experiments"]["E4"]
    errors = []
    _validate_experiment_inventory(
        missing_e4,
        model_manifest,
        tmp_path,
        hardened=True,
        errors=errors,
        warnings=[],
    )
    assert any("exactly E0" in error for error in errors)
    extra = {"experiments": {**metrics["experiments"], "E5": {"status": "SUCCESS"}}}
    errors = []
    _validate_experiment_inventory(
        extra,
        model_manifest,
        tmp_path,
        hardened=True,
        errors=errors,
        warnings=[],
    )
    assert any("exactly E0" in error for error in errors)


def _prediction_fixture(
    *, e4_status: str = "SUCCESS"
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    statuses = {experiment_id: {"status": "SUCCESS"} for experiment_id in ("E0", "E1", "E2", "E3")}
    statuses["E4"] = {"status": e4_status}
    rows = [
        {KEY: key, "experiment_id": experiment_id, "probability": 0.1 + 0.1 * key}
        for experiment_id, key in (
            (experiment_id, key) for experiment_id in statuses for key in (1, 2)
        )
        if statuses[experiment_id]["status"] == "SUCCESS"
    ]
    return pd.DataFrame(rows, columns=[KEY, "experiment_id", "probability"]), statuses


def test_prediction_audit_requires_exact_per_experiment_validation_membership() -> None:
    predictions, statuses = _prediction_fixture()
    manifest = {
        "prediction_artifact": {
            "row_count": len(predictions),
            "column_count": 3,
            "canonical_content_sha256": prediction_content_sha256(predictions),
        }
    }
    errors: list[str] = []
    _validate_predictions(
        predictions,
        statuses,
        {1, 2},
        {3},
        {4},
        manifest,
        hardened=True,
        errors=errors,
        warnings=[],
    )
    assert errors == []
    unavailable_predictions, unavailable_statuses = _prediction_fixture(
        e4_status="UNAVAILABLE_WITH_WARNING"
    )
    unavailable_manifest = {
        "prediction_artifact": {
            "row_count": len(unavailable_predictions),
            "column_count": 3,
            "canonical_content_sha256": prediction_content_sha256(unavailable_predictions),
        }
    }
    errors = []
    _validate_predictions(
        unavailable_predictions,
        unavailable_statuses,
        {1, 2},
        {3},
        {4},
        unavailable_manifest,
        hardened=True,
        errors=errors,
        warnings=[],
    )
    assert errors == []
    bad = predictions.iloc[:-1].copy()
    errors = []
    _validate_predictions(
        bad,
        statuses,
        {1, 2},
        {3},
        {4},
        manifest,
        hardened=True,
        errors=errors,
        warnings=[],
    )
    assert any("membership" in error or "row count" in error for error in errors)
    duplicate = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    errors = []
    _validate_predictions(
        duplicate,
        statuses,
        {1, 2},
        {3},
        {4},
        manifest,
        hardened=True,
        errors=errors,
        warnings=[],
    )
    assert any("duplicate" in error for error in errors)
    test_row = predictions.copy()
    test_row.loc[0, KEY] = 4
    errors = []
    _validate_predictions(
        test_row,
        statuses,
        {1, 2},
        {3},
        {4},
        manifest,
        hardened=True,
        errors=errors,
        warnings=[],
    )
    assert any("TEST" in error for error in errors)


def test_e4_success_requires_model_and_predictions(tmp_path: Path) -> None:
    metrics, model_manifest = _inventory_fixture(tmp_path, e4_status="SUCCESS")
    del model_manifest["models"]["E4"]
    errors: list[str] = []
    _validate_experiment_inventory(
        metrics,
        model_manifest,
        tmp_path,
        hardened=True,
        errors=errors,
        warnings=[],
    )
    assert any("model inventory" in error or "E4 model" in error for error in errors)
    predictions, statuses = _prediction_fixture(e4_status="SUCCESS")
    errors = []
    _validate_predictions(
        predictions.iloc[:-2],
        statuses,
        {1, 2},
        {3},
        {4},
        {"prediction_artifact": {}},
        hardened=True,
        errors=errors,
        warnings=[],
    )
    assert any("E4" in error for error in errors)


def test_model_policy_rejects_active_class_weighting_and_parameter_drift() -> None:
    valid = dict(MODEL_CORE_PARAMETERS)
    assert model_policy_errors(valid) == []
    active = {**valid, "auto_class_weights": "Balanced"}
    assert any("active class weighting" in error for error in model_policy_errors(active))
    changed = {**valid, "depth": 8}
    assert any("depth" in error for error in model_policy_errors(changed))


def test_runtime_provenance_is_complete_and_secret_free() -> None:
    runtime = runtime_provenance()
    assert set(runtime) >= {
        "python_version",
        "python_implementation",
        "catboost_version",
        "scikit_learn_version",
        "pandas_version",
        "numpy_version",
        "pyarrow_version",
        "platform",
        "machine",
        "os",
    }
    assert runtime_provenance_errors(runtime) == []
    assert runtime_provenance_errors({"python_version": "3.14", "password": "secret"})


def _fictional_compliant_runtime() -> dict[str, str]:
    return {
        "python_version": "3.12.8",
        "python_implementation": "CPython",
        "catboost_version": "1.2.10",
        "scikit_learn_version": "1.7.2",
        "pandas_version": "2.3.3",
        "numpy_version": "2.2.6",
        "pyarrow_version": "19.0.1",
        "platform": "fictional",
        "machine": "AMD64",
        "os": "Windows",
    }


def test_runtime_dependency_constraints_accept_declared_phase9_environment() -> None:
    result = validate_runtime_dependency_constraints(Path.cwd(), _fictional_compliant_runtime())
    assert result["valid"] is True, result
    checked = result["checked_requirements"]
    assert set(checked) == {"python", "numpy", "pandas", "pyarrow", "catboost", "scikit-learn"}
    assert all(item["compatible"] for item in checked.values())


@pytest.mark.parametrize(
    ("runtime_key", "version", "dependency"),
    [
        ("pandas_version", "3.0.5", "pandas"),
        ("pyarrow_version", "25.0.1", "pyarrow"),
        ("catboost_version", "2.0", "catboost"),
        ("scikit_learn_version", "2.0", "scikit-learn"),
        ("numpy_version", "3.0", "numpy"),
    ],
)
def test_runtime_dependency_constraints_block_incompatible_phase9_environment(
    runtime_key: str, version: str, dependency: str
) -> None:
    runtime = _fictional_compliant_runtime()
    runtime[runtime_key] = version
    result = validate_runtime_dependency_constraints(Path.cwd(), runtime)
    assert result["valid"] is False
    assert result["status"] == "BLOCKED"
    assert any(dependency in error for error in result["errors"])


def test_feature_hash_and_champion_mismatch_are_blocking(tmp_path: Path) -> None:
    spec = FeatureSetSpec("E1", ("number",), ("number",), (), (), (), 1, 0, 0, 0, "hash")
    persisted = {"E1": {**spec.as_dict(), "feature_set_sha256": "wrong"}}
    schema = {
        "E1": {
            "ordered_features": ["number"],
            "numeric": ["number"],
            "categorical": [],
            "boolean": [],
            "text": [],
        }
    }
    errors: list[str] = []
    _validate_feature_sets(
        persisted,
        schema,
        {"models": {}},
        {"E1": spec},
        hardened=True,
        errors=errors,
        warnings=[],
    )
    assert any("feature sets" in error for error in errors)
    common = {"average_precision": 0.7, "roc_auc": 0.8, "log_loss": 0.2}
    statuses = {
        experiment_id: {"status": "SUCCESS", "model_type": "CatBoost", "metrics": dict(common)}
        for experiment_id in ("E0", "E1", "E2", "E3", "E4")
    }
    errors = []
    champion = _validate_champion(
        tmp_path,
        {"champion_experiment_id": "E4"},
        {"champion_experiment_id": "E4"},
        tmp_path / "missing-report.json",
        {},
        statuses,
        hardened=True,
        allow_report_pending=False,
        errors=errors,
        warnings=[],
    )
    assert champion == "E1"
    assert any("champion" in error for error in errors)


def _write_comparison_run(path: Path, run_id: str, *, drift: bool = False) -> None:
    path.mkdir(parents=True)
    target_hash = "target-after" if drift else "target"
    (path / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "frozen_membership": {
                    "counts": {"TRAIN": 2, "VALIDATION": 2, "TEST": 1 if not drift else 2}
                },
                "target_summary": {
                    "train": {"target_content_sha256": target_hash},
                    "validation": {"target_content_sha256": target_hash},
                },
                "champion_experiment_id": "E1",
            }
        ),
        encoding="utf-8",
    )
    score = 0.8 if drift else 0.7
    (path / "validation_metrics.json").write_text(
        json.dumps(
            {
                "experiments": {
                    "E0": {"status": "SUCCESS", "metrics": {"average_precision": 0.2}},
                    "E1": {"status": "SUCCESS", "metrics": {"average_precision": score}},
                }
            }
        ),
        encoding="utf-8",
    )
    feature_hash = "feature-after" if drift else "feature"
    (path / "feature_sets.json").write_text(
        json.dumps({"E1": {"feature_count": 1, "feature_set_sha256": feature_hash}}),
        encoding="utf-8",
    )
    predictions = pd.DataFrame(
        {
            KEY: [1, 2],
            "experiment_id": ["E0", "E1"],
            "probability": [0.2, 0.8 if not drift else 0.7],
        }
    )
    predictions.to_parquet(path / "validation_predictions.parquet", index=False)
    (path / "model_manifest.json").write_text(
        json.dumps({"models": {"E1": {"model_sha256": "binary-after" if drift else "binary"}}}),
        encoding="utf-8",
    )


def test_before_after_comparison_allows_model_binary_drift_but_blocks_semantic_drift(
    tmp_path: Path,
) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_comparison_run(before, "before")
    _write_comparison_run(after, "after")
    passed = compare_phase9_runs(before, after)
    assert passed["valid"] is True
    assert passed["model_binary_hashes_required_to_match"] is False
    drifted = tmp_path / "drifted"
    _write_comparison_run(drifted, "drifted", drift=True)
    blocked = compare_phase9_runs(before, drifted)
    assert blocked["valid"] is False
    assert blocked["errors"]


def test_manifest_hash_and_test_seal_guards_fail_closed(tmp_path: Path) -> None:
    inputs = SimpleNamespace(
        phase5_manifest={"artifact_content_fingerprints": {"claim_snapshot": "p5"}},
        phase6_manifest={},
        phase7_manifest={"artifact_content_sha256": {"structured_features": "p7"}},
        phase8_manifest={"artifact_content_sha256": {"text_features": "p8"}},
        frozen_membership={"split_assignment_sha256": "p6"},
    )
    errors: list[str] = []
    _validate_input_hashes(
        tmp_path,
        {
            "input_hashes": {
                "phase5_claim_snapshot": "wrong",
                "phase6_split_assignment": "p6",
                "phase7_structured_features": "p7",
                "phase8_text_features": "p8",
            },
            "artifact_file_sha256": {},
        },
        inputs,
        hardened=True,
        errors=errors,
        warnings=[],
    )
    assert any("input hashes" in error for error in errors)
    (tmp_path / "test_metrics.json").write_text("{}", encoding="utf-8")
    errors = []
    _validate_test_seal(
        tmp_path,
        {"test_target_rows_loaded": 1},
        {"test_seal": {"test_target_rows_loaded": 0}},
        errors,
    )
    assert any("TEST" in error for error in errors)


def test_ap_lift_and_configuration_policy_are_recomputed() -> None:
    statuses = {
        "E0": {
            "status": "SUCCESS",
            "metrics": {"average_precision": 0.2, "ap_lift_over_prevalence_baseline": 1.0},
        },
        "E1": {
            "status": "SUCCESS",
            "metrics": {"average_precision": 0.4, "ap_lift_over_prevalence_baseline": 2.0},
        },
    }
    errors: list[str] = []
    _validate_ap_lift(statuses, hardened=True, errors=errors, warnings=[])
    assert errors == []
    statuses["E1"]["metrics"]["ap_lift_over_prevalence_baseline"] = 3.0
    _validate_ap_lift(statuses, hardened=True, errors=errors, warnings=[])
    assert any("AP lift" in error for error in errors)
    zero_statuses = {
        "E0": {
            "status": "SUCCESS",
            "metrics": {"average_precision": 0.0, "ap_lift_over_prevalence_baseline": None},
        },
        "E1": {
            "status": "SUCCESS",
            "metrics": {"average_precision": 0.0, "ap_lift_over_prevalence_baseline": None},
        },
    }
    errors = []
    _validate_ap_lift(zero_statuses, hardened=True, errors=errors, warnings=[])
    assert errors == []
    settings = load_baseline_settings(Path.cwd())
    contract = {
        "fixed_catboost_parameters": {"eval_set": "none", "early_stopping": "none"},
        "class_imbalance_policy": {"class_weights": "none"},
    }
    errors = []
    _validate_configuration_policy(settings, contract, errors)
    assert errors == []
    contract["fixed_catboost_parameters"]["eval_set"] = "validation"
    _validate_configuration_policy(settings, contract, errors)
    assert any("eval_set" in error for error in errors)
