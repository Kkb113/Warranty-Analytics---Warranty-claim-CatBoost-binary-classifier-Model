"""Focused Phase 10 contract, chronology, selection, and reproducibility tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from warranty_analytics_model.baseline_model.models import FeatureSetSpec, Phase9Inputs
from warranty_analytics_model.baseline_model.provenance import (
    runtime_provenance,
    validate_runtime_dependency_constraints,
)
from warranty_analytics_model.catboost_optimization.config import (
    TRACK_TO_EXPERIMENT,
    load_optimization_settings,
    settings_payload,
)
from warranty_analytics_model.catboost_optimization.contract import (
    REQUIRED_PHASE9_RUN_ID,
    load_optimization_contract,
    validate_optimization_contract,
)
from warranty_analytics_model.catboost_optimization.finalists import fit_phase10_finalists
from warranty_analytics_model.catboost_optimization.inner_folds import build_inner_fold_plan
from warranty_analytics_model.catboost_optimization.input import (
    EXPECTED_PHASE9_TARGET_HASHES,
    load_validation_targets_after_freeze,
)
from warranty_analytics_model.catboost_optimization.manifest import (
    freeze_payload_sha256,
    write_json,
    write_table,
)
from warranty_analytics_model.catboost_optimization.metrics import (
    aggregate_fold_metrics,
    compare_metrics,
    validate_prediction_frame,
)
from warranty_analytics_model.catboost_optimization.models import (
    OptimizationError,
    Phase10Inputs,
    StudyResult,
)
from warranty_analytics_model.catboost_optimization.objective import evaluate_parameters
from warranty_analytics_model.catboost_optimization.provenance import (
    ACCEPTANCE_OVERLAY_FILENAME,
    canonical_json_sha256,
    fold_content_sha256,
    prediction_sha256,
    write_acceptance_overlay,
)
from warranty_analytics_model.catboost_optimization.reporting import write_phase10_reports
from warranty_analytics_model.catboost_optimization.search_space import (
    parameter_sha256,
    suggest_trial_parameters,
    validate_trial_parameters,
)
from warranty_analytics_model.catboost_optimization.selection import (
    replacement_decision,
    select_best_trial,
    select_development_champion,
)
from warranty_analytics_model.catboost_optimization.study import (
    TRIAL_HISTORY_COLUMNS,
    require_trial_history_schema,
    run_track_study,
)
from warranty_analytics_model.catboost_optimization.validation import (
    TRIAL_FOLD_METRIC_COLUMNS,
    _reproduce_winning_trials,
    _validate_trial_fold_evidence,
    validate_optimization_directory,
)


def _valid_parameters() -> dict[str, object]:
    return {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
        "border_count": 254,
        "rsm": 1.0,
    }


def _synthetic_train() -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = np.arange(1, 49, dtype="int64")
    dates = pd.date_range("2024-01-01", periods=24, freq="D").repeat(2)
    metadata = pd.DataFrame({"warranty_claim_key": keys, "claim_date": dates})
    target = pd.DataFrame(
        {
            "warranty_claim_key": keys,
            "target__high_cost_claim_flag": np.tile([0, 1], 24),
        }
    )
    return metadata, target


def test_phase10_config_and_track_immutability() -> None:
    settings = load_optimization_settings()
    assert settings.tracks == ("T1", "T3")
    assert TRACK_TO_EXPERIMENT == {"T1": "E1", "T3": "E3"}
    assert list(settings.search_space) == [
        "iterations",
        "learning_rate",
        "depth",
        "l2_leaf_reg",
        "random_strength",
        "bagging_temperature",
        "border_count",
        "rsm",
    ]


def test_phase10_config_guards_fail_closed() -> None:
    from warranty_analytics_model.catboost_optimization.config import (
        _as_float_tuple,
        _validate_fixed_parameters,
        _validate_search_space,
    )

    with pytest.raises(OptimizationError, match="non-empty list"):
        _as_float_tuple([], "fractions")
    with pytest.raises(OptimizationError, match="contain numbers"):
        _as_float_tuple(["bad"], "fractions")
    with pytest.raises(OptimizationError, match="exactly the allowlisted"):
        _validate_search_space({})
    altered_space = dict(load_optimization_settings().search_space)
    altered_space["iterations"] = {"type": "categorical", "values": [1]}
    with pytest.raises(OptimizationError, match="search-space definition changed"):
        _validate_search_space(altered_space)
    with pytest.raises(OptimizationError, match="fixed_parameters"):
        _validate_fixed_parameters(None)
    with pytest.raises(OptimizationError, match="fixed CatBoost setting"):
        _validate_fixed_parameters({"loss_function": "Logloss"})
    overlapping_fixed = dict(load_optimization_settings().fixed_parameters)
    overlapping_fixed["iterations"] = 500
    with pytest.raises(OptimizationError, match="search parameters"):
        _validate_fixed_parameters(overlapping_fixed)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("iterations", 301),
        ("l2_leaf_reg", 0.5),
        ("random_strength", 4.0),
        ("bagging_temperature", 6.0),
        ("rsm", 0.5),
        ("depth", 12),
        ("depth", 6.0),
        ("learning_rate", 0.5),
        ("border_count", 999),
        ("unknown", 1),
    ],
)
def test_phase10_search_space_rejects_invalid_values(name: str, value: object) -> None:
    parameters = _valid_parameters()
    if name == "unknown":
        parameters[name] = value
    else:
        parameters[name] = value
    with pytest.raises(OptimizationError):
        validate_trial_parameters(parameters)


@pytest.mark.parametrize(
    "key", ["class_weights", "auto_class_weights", "scale_pos_weight", "early_stopping_rounds"]
)
def test_phase10_fixed_policy_rejects_weighting_and_early_stopping(key: str) -> None:
    from warranty_analytics_model.catboost_optimization.config import _validate_fixed_parameters

    fixed = {
        "loss_function": "Logloss",
        "bootstrap_type": "Bayesian",
        "random_seed": 20260810,
        "task_type": "CPU",
        "thread_count": 10,
        "allow_writing_files": False,
        "verbose": False,
        "use_best_model": False,
        key: "Balanced" if key == "auto_class_weights" else 2,
    }
    with pytest.raises(OptimizationError):
        _validate_fixed_parameters(fixed)


def test_inner_folds_are_date_grouped_chronological_and_target_independent() -> None:
    metadata, targets = _synthetic_train()
    plan = build_inner_fold_plan(
        metadata,
        targets,
        minimum_train_positive=1,
        minimum_validation_positive=1,
    )
    assert len(plan.folds) == 3
    for fold in plan.folds:
        assert fold.train_max_date < fold.validation_min_date
        assert set(fold.train_keys).isdisjoint(fold.validation_keys)
        roles = plan.assignments.loc[plan.assignments["fold_id"] == fold.fold_id]
        assert not roles.groupby("claim_date")["role"].nunique().gt(1).any()
    shuffled = build_inner_fold_plan(
        metadata.sample(frac=1.0, random_state=17).reset_index(drop=True),
        targets.sample(frac=1.0, random_state=29).reset_index(drop=True),
        minimum_train_positive=1,
        minimum_validation_positive=1,
    )
    pd.testing.assert_frame_equal(plan.assignments, shuffled.assignments)
    assert plan.content_sha256 == shuffled.content_sha256
    altered = targets.copy()
    altered["target__high_cost_claim_flag"] = 1 - altered["target__high_cost_claim_flag"]
    altered_plan = build_inner_fold_plan(
        metadata,
        altered,
        minimum_train_positive=1,
        minimum_validation_positive=1,
    )
    assert altered_plan.content_sha256 == plan.content_sha256


def test_best_trial_deterministic_tie_breaks() -> None:
    history = pd.DataFrame(
        [
            {
                "trial_number": 1,
                "state": "COMPLETE",
                "mean_average_precision": 0.5,
                "min_average_precision": 0.4,
                "std_average_precision": 0.03,
                "mean_roc_auc": 0.7,
                "mean_log_loss": 0.4,
                "depth": 6,
                "iterations": 700,
            },
            {
                "trial_number": 2,
                "state": "COMPLETE",
                "mean_average_precision": 0.5,
                "min_average_precision": 0.4,
                "std_average_precision": 0.03,
                "mean_roc_auc": 0.7,
                "mean_log_loss": 0.4,
                "depth": 5,
                "iterations": 1400,
            },
        ]
    )
    assert select_best_trial(history)["trial_number"] == 2


def test_replacement_and_champion_rules() -> None:
    baseline = {"average_precision": 0.4, "roc_auc": 0.7, "log_loss": 0.4}
    gain = {"average_precision": 0.400002, "roc_auc": 0.69, "log_loss": 0.5}
    assert replacement_decision(baseline, gain)["optimized_beats_baseline"] is True
    assert (
        replacement_decision(baseline, {**gain, "average_precision": 0.4000005})[
            "fallback_to_baseline"
        ]
        is True
    )
    candidates = [
        {"candidate_id": "P9_E1_BASELINE", "metrics": baseline, "feature_count": 301},
        {
            "candidate_id": "P10_T1_E1_OPTIMIZED",
            "metrics": {**gain, "average_precision": 0.4000005},
            "feature_count": 301,
        },
        {
            "candidate_id": "P9_E3_BASELINE",
            "metrics": baseline,
            "feature_count": 536,
        },
    ]
    assert select_development_champion(candidates)["candidate_id"] == "P9_E1_BASELINE"


def test_validation_loader_requires_study_freeze() -> None:
    with pytest.raises(OptimizationError, match="study_freeze"):
        load_validation_targets_after_freeze(object(), study_frozen=False)  # type: ignore[arg-type]


def test_locked_input_and_authorized_target_loaders_are_split_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import warranty_analytics_model.catboost_optimization.input as optimization_input
    from warranty_analytics_model.baseline_model.target import target_content_sha256

    def feature_spec(experiment_id: str, count: int, digest: str) -> FeatureSetSpec:
        names = tuple(f"feature_{index}" for index in range(count))
        return FeatureSetSpec(experiment_id, names, names, (), (), (), count, 0, 0, 0, digest)

    specs = {
        "E1": feature_spec("E1", 301, optimization_input.EXPECTED_FEATURE_SETS["E1"][1]),
        "E3": feature_spec("E3", 536, optimization_input.EXPECTED_FEATURE_SETS["E3"][1]),
    }
    keys = np.arange(1, 7, dtype="int64")
    assignments = pd.DataFrame(
        {
            "warranty_claim_key": keys,
            "split": ["TRAIN", "TRAIN", "TRAIN", "VALIDATION", "VALIDATION", "TEST"],
        }
    )
    structured = pd.DataFrame(
        {
            "warranty_claim_key": keys,
            "claim__claim_date": pd.date_range("2024-01-01", periods=len(keys)),
        }
    )
    train_targets = pd.DataFrame(
        {"warranty_claim_key": [1, 2, 3], "target__high_cost_claim_flag": [0, 1, 0]}
    )
    validation_targets = pd.DataFrame(
        {"warranty_claim_key": [4, 5], "target__high_cost_claim_flag": [1, 0]}
    )
    snapshot = tmp_path / "mart" / "claim_snapshot.parquet"
    snapshot.parent.mkdir(parents=True)
    pd.concat([train_targets, validation_targets], ignore_index=True).assign(
        **{"warranty_claim_key": [1, 2, 3, 4, 5]}
    ).to_parquet(snapshot, index=False)

    train_hash = target_content_sha256(train_targets)
    validation_hash = target_content_sha256(validation_targets)
    monkeypatch.setitem(optimization_input.EXPECTED_PHASE9_TARGET_HASHES, "train", train_hash)
    monkeypatch.setitem(
        optimization_input.EXPECTED_PHASE9_TARGET_HASHES, "validation", validation_hash
    )
    phase9_dir = tmp_path / "phase9"
    phase9_dir.mkdir()
    (phase9_dir / "feature_sets.json").write_text(
        json.dumps(
            {key: {"feature_set_sha256": value.feature_set_sha256} for key, value in specs.items()}
        ),
        encoding="utf-8",
    )
    input_hashes = {
        "phase5_claim_snapshot": "p5",
        "phase6_split_assignment": "p6",
        "phase7_structured_features": "p7",
        "phase8_text_features": "p8",
    }
    phase9_inputs = Phase9Inputs(
        root=tmp_path,
        mart_dir=tmp_path / "mart",
        split_dir=tmp_path / "split",
        structured_dir=tmp_path / "structured",
        text_dir=tmp_path / "text",
        assignments=assignments,
        structured_features=structured,
        text_features=pd.DataFrame(),
        phase7_lineage={},
        phase8_lineage={},
        phase5_manifest={"artifact_content_fingerprints": {"claim_snapshot": "p5"}},
        phase6_manifest={},
        phase7_manifest={"artifact_content_sha256": {"structured_features": "p7"}},
        phase8_manifest={"artifact_content_sha256": {"text_features": "p8"}},
        test_lock={},
        upstream_validations={},
        frozen_membership={"split_assignment_sha256": "p6"},
        source_audit={},
    )
    development = assignments.iloc[:5].copy()
    manifest = {
        "phase": 9,
        "hardened_status": "HARDENED_PASS",
        "input_directories": {
            key: str(tmp_path / key) for key in ("phase5", "phase6", "phase7", "phase8")
        },
        "input_hashes": input_hashes,
    }
    monkeypatch.setattr(
        optimization_input, "_validate_locked_phase9_directory", lambda path: manifest
    )
    monkeypatch.setattr(
        optimization_input, "load_phase9_inputs", lambda *args, **kwargs: phase9_inputs
    )
    monkeypatch.setattr(optimization_input, "resolve_feature_sets", lambda *args: specs)
    monkeypatch.setattr(
        optimization_input,
        "build_development_feature_frame",
        lambda *args: development.copy(),
    )

    phase10_inputs = optimization_input.load_locked_phase9_inputs(phase9_dir, project_root=tmp_path)
    assert "claim__claim_date" in phase10_inputs.development.columns
    train, train_audit = optimization_input.load_train_targets_for_optimization(phase10_inputs)
    pd.testing.assert_frame_equal(
        train,
        train_targets.astype({"target__high_cost_claim_flag": "int8"}),
    )
    assert train_audit["split"] == "TRAIN"
    with pytest.raises(OptimizationError, match="study_freeze"):
        optimization_input.load_validation_targets_after_freeze(phase10_inputs, study_frozen=False)
    validation, validation_audit = optimization_input.load_validation_targets_after_freeze(
        phase10_inputs, study_frozen=True
    )
    pd.testing.assert_frame_equal(
        validation,
        validation_targets.astype({"target__high_cost_claim_flag": "int8"}),
    )
    assert validation_audit["split"] == "VALIDATION"
    with pytest.raises(OptimizationError, match="never loads TEST"):
        optimization_input._load_target_rows(phase10_inputs, "TEST", frozen=True)


def test_parameter_hash_is_deterministic() -> None:
    params = _valid_parameters()
    assert parameter_sha256(params) == parameter_sha256(dict(reversed(list(params.items()))))


def test_locked_phase9_manifest_gate_checks_hardening_and_artifact_hashes(
    tmp_path: Path,
) -> None:
    import warranty_analytics_model.catboost_optimization.input as optimization_input
    from warranty_analytics_model.feature_mart.manifest import sha256_file

    required = (
        "experiment_manifest.json",
        "feature_sets.json",
        "model_input_schema.json",
        "model_manifest.json",
        "target_access_audit.json",
        "validation_metrics.json",
        "validation.json",
        "validation_predictions.parquet",
    )
    for name in required:
        path = tmp_path / name
        if path.suffix == ".parquet":
            pd.DataFrame({"warranty_claim_key": [1]}).to_parquet(path, index=False)
        else:
            path.write_text("{}", encoding="utf-8")
    (tmp_path / "validation.json").write_text(
        json.dumps({"valid": True, "hardening_status": "HARDENED_PASS"}),
        encoding="utf-8",
    )
    (tmp_path / "target_access_audit.json").write_text(
        json.dumps(
            {
                "test_target_rows_loaded": 0,
                "test_predictions_created": 0,
                "test_metrics_computed": False,
                "test_target_access_allowed": False,
                "first_allowed_test_target_phase": 15,
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "phase": 9,
        "hardened_status": "HARDENED_PASS",
        "hardening_version": "phase9_corrective_hardening_v1",
        "target_hashes": optimization_input.EXPECTED_PHASE9_TARGET_HASHES,
        "artifact_file_sha256": {"feature_sets.json": sha256_file(tmp_path / "feature_sets.json")},
    }
    (tmp_path / "experiment_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert optimization_input._validate_locked_phase9_directory(tmp_path)["phase"] == 9
    manifest["hardened_status"] = "BLOCKED"
    (tmp_path / "experiment_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(OptimizationError, match="HARDENED_PASS"):
        optimization_input._validate_locked_phase9_directory(tmp_path)


def test_phase10_input_guards_reject_malformed_json_and_target_snapshots(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    import warranty_analytics_model.catboost_optimization.input as optimization_input

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(OptimizationError, match="not valid JSON"):
        optimization_input._read_json(malformed, "fixture")
    malformed.write_text("[]", encoding="utf-8")
    with pytest.raises(OptimizationError, match="must be a JSON object"):
        optimization_input._read_json(malformed, "fixture")

    assignments = pd.DataFrame({"warranty_claim_key": [1, 2], "split": ["TRAIN", "TRAIN"]})
    phase10_inputs = Phase10Inputs(
        root=tmp_path,
        phase9_dir=tmp_path,
        phase9_manifest={},
        phase9_inputs=SimpleNamespace(assignments=assignments),  # type: ignore[arg-type]
        feature_sets={},
        development=pd.DataFrame(),
        claim_snapshot_path=tmp_path / "snapshot.parquet",
    )
    phase10_inputs.claim_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"warranty_claim_key": [1, 2], "target__high_cost_claim_flag": [0, 2]}).to_parquet(
        phase10_inputs.claim_snapshot_path, index=False
    )
    with pytest.raises(OptimizationError, match="binary"):
        optimization_input.load_train_targets_for_optimization(phase10_inputs)
    pd.DataFrame({"warranty_claim_key": [1, 2]}).to_parquet(
        phase10_inputs.claim_snapshot_path, index=False
    )
    with pytest.raises(OptimizationError, match="authoritative target"):
        optimization_input.load_train_targets_for_optimization(phase10_inputs)
    empty_inputs = Phase10Inputs(
        root=tmp_path,
        phase9_dir=tmp_path,
        phase9_manifest={},
        phase9_inputs=SimpleNamespace(
            assignments=pd.DataFrame({"warranty_claim_key": [1], "split": ["VALIDATION"]})
        ),  # type: ignore[arg-type]
        feature_sets={},
        development=pd.DataFrame(),
        claim_snapshot_path=phase10_inputs.claim_snapshot_path,
    )
    with pytest.raises(OptimizationError, match="population is empty"):
        optimization_input.load_train_targets_for_optimization(empty_inputs)


def test_optuna_tiny_reproducibility() -> None:
    optuna = pytest.importorskip("optuna")

    def run() -> list[dict[str, object]]:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=20260810, n_startup_trials=2),
            pruner=optuna.pruners.NopPruner(),
        )
        study.optimize(
            lambda trial: float(-((trial.suggest_float("x", -1.0, 1.0) - 0.25) ** 2)),
            n_trials=5,
            n_jobs=1,
        )
        return [{"params": trial.params, "value": trial.value} for trial in study.trials]

    assert run() == run()


def test_optuna_dependency_gate_rejects_incompatible_version() -> None:
    runtime = runtime_provenance(include_optimization=True)
    runtime["optuna_version"] = "0.1.0"
    result = validate_runtime_dependency_constraints(Path.cwd(), runtime, include_optimization=True)
    assert result["valid"] is False
    assert any("optuna" in error for error in result["errors"])


def test_acceptance_overlay_does_not_invent_missing_legacy_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "phase10"
    run_dir.mkdir()
    write_json(
        run_dir / "optimization_manifest.json",
        {
            "run_id": "20260811T_PHASE10",
            "git_commit_sha": "legacy-run-sha",
            "created_at_utc": "2026-08-12T10:23:30+00:00",
            "phase9_run_id": REQUIRED_PHASE9_RUN_ID,
            "contract_version": "phase10_catboost_optimization_v2",
            "contract_checksum": "contract-sha",
            "contract_policy_snapshot": {},
            "phase9_target_hashes": {},
            "phase9_feature_set_hashes": {},
            "trials_per_track": 50,
            "objective_metric": "mean_average_precision",
            "artifact_file_sha256": {"trial_history.parquet": "history-sha"},
            "test_target_rows_loaded": 0,
            "test_predictions_created": 0,
            "test_metrics_computed": False,
            "test_target_access_allowed": False,
            "first_allowed_test_target_phase": 15,
        },
    )
    write_json(
        run_dir / "model_manifest.json",
        {"models": {"P10_T1_E1_OPTIMIZED": {"model_sha256": "model-sha"}}},
    )
    write_json(run_dir / "validation.json", {"status": "PASS WITH WARNINGS"})

    overlay_path, created = write_acceptance_overlay(
        run_dir,
        validation_result={
            "status": "PASS WITH WARNINGS",
            "hardening_status": "HARDENED_PASS",
        },
        validator_commit_sha="hardening-code-sha",
    )
    assert created is True
    assert overlay_path.name == ACCEPTANCE_OVERLAY_FILENAME
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    assert overlay["source_manifest"]["sha256"]
    assert overlay["legacy_manifest"] == {
        "preserved": False,
        "path": None,
        "sha256": None,
        "status": "UNAVAILABLE",
        "note": (
            "No pre-v2 optimization manifest copy was present when this overlay was "
            "created. The current v2 manifest is recorded as post-hardening evidence "
            "only; its hash is not claimed as the original manifest hash."
        ),
    }
    assert overlay["hardening"]["validator_commit_sha"] == "hardening-code-sha"

    second_path, second_created = write_acceptance_overlay(
        run_dir,
        validation_result={"status": "BLOCKED", "hardening_status": "BLOCKED"},
        validator_commit_sha="different-sha",
    )
    assert second_path == overlay_path
    assert second_created is False
    assert json.loads(second_path.read_text(encoding="utf-8")) == overlay


def test_phase10_contract_hashes_metrics_and_reports(tmp_path: Path) -> None:
    contract = validate_optimization_contract()
    assert contract["valid"] is True
    payload = {"phase": 10, "tracks": ["T1", "T3"]}
    payload["study_freeze_sha256"] = freeze_payload_sha256(payload)
    assert canonical_json_sha256({"phase": 10})
    frame = pd.DataFrame(
        {
            "warranty_claim_key": [1, 2],
            "candidate_id": ["P10_T1_E1_OPTIMIZED"] * 2,
            "high_cost_probability": [0.1, 0.9],
        }
    )
    validate_prediction_frame(frame, {"P10_T1_E1_OPTIMIZED"})
    assert prediction_sha256(frame)
    fold_frame = pd.DataFrame(
        {
            "warranty_claim_key": [1, 2],
            "claim_date": ["2024-01-01", "2024-01-02"],
            "fold_id": [1, 1],
            "role": ["TRAIN", "VALIDATION"],
        }
    )
    assert fold_content_sha256(fold_frame)
    aggregate = aggregate_fold_metrics(
        [
            {
                "average_precision": 0.4,
                "roc_auc": 0.7,
                "log_loss": 0.5,
                "brier_score": 0.2,
            },
            {
                "average_precision": 0.5,
                "roc_auc": 0.8,
                "log_loss": 0.4,
                "brier_score": 0.1,
            },
        ]
    )
    assert aggregate["mean_average_precision"] == pytest.approx(0.45)
    comparison = compare_metrics(
        {"average_precision": 0.4, "roc_auc": 0.7, "log_loss": 0.5, "brier_score": 0.2},
        {"average_precision": 0.5, "roc_auc": 0.8, "log_loss": 0.4, "brier_score": 0.1},
    )
    assert comparison["optimized_beats_baseline"] is True
    write_table(fold_frame, tmp_path / "folds.parquet", "snappy")
    write_phase10_reports(
        tmp_path / "reports",
        {
            "status": "PASS",
            "phase10_development_champion": "P9_E1_BASELINE",
            "optimization_comparison": {"T1": comparison},
            "warnings": [],
            "inner_cv_summary": {},
            "best_parameters": {},
            "validation_metrics": {},
            "validation": {},
        },
    )
    assert (tmp_path / "reports" / "phase_10_summary.md").is_file()


def test_phase10_contract_validator_is_fail_closed_on_policy_and_dependency_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    import warranty_analytics_model.catboost_optimization.contract as contract

    settings = load_optimization_settings()
    declared_extra = contract._validate_declared_optimization_extra
    monkeypatch.setattr(contract, "discover_repository_root", lambda root=None: tmp_path)
    monkeypatch.setattr(
        contract,
        "load_optimization_contract",
        lambda root=None: ({"phase10": {}}, "synthetic-checksum"),
    )
    monkeypatch.setattr(contract, "load_optimization_settings", lambda root=None: settings)
    monkeypatch.setattr(
        contract,
        "_validate_declared_optimization_extra",
        lambda root: "synthetic optimization extra error",
    )
    result = contract.validate_optimization_contract(tmp_path)
    assert result["valid"] is False
    assert any("policy differs" in error for error in result["errors"])
    assert "synthetic optimization extra error" in result["errors"]

    monkeypatch.setattr(
        contract,
        "load_optimization_contract",
        lambda root=None: ({"phase10": "not-a-mapping"}, "synthetic-checksum"),
    )
    malformed = contract.validate_optimization_contract(tmp_path)
    assert malformed["valid"] is False
    assert "Phase 10 contract must contain phase10 mapping." in malformed["errors"]

    assert "Could not read pyproject" in declared_extra(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project.optional-dependencies]\noptimization = ["wrong"]\n', encoding="utf-8"
    )
    assert "must contain exactly" in declared_extra(tmp_path)

    altered_settings = replace(
        settings,
        tracks=("T1",),
        trials_per_track=49,
        search_space={},
        fixed_parameters={},
    )
    expected_policy = contract._expected_policy(settings)
    monkeypatch.setattr(contract, "_expected_policy", lambda _: expected_policy)
    monkeypatch.setattr(contract, "load_optimization_settings", lambda root=None: altered_settings)
    monkeypatch.setattr(contract, "_validate_declared_optimization_extra", lambda root: None)
    monkeypatch.setattr(
        contract,
        "load_optimization_contract",
        lambda root=None: ({"phase10": {}}, "synthetic-checksum"),
    )
    drift = contract.validate_optimization_contract(tmp_path)
    assert any("tracks are not exactly" in error for error in drift["errors"])
    assert any("exactly 50 trials" in error for error in drift["errors"])
    assert any("search space differs" in error for error in drift["errors"])
    assert any("fixed CatBoost" in error for error in drift["errors"])

    monkeypatch.setattr(
        contract,
        "load_optimization_contract",
        lambda root=None: (_ for _ in ()).throw(RuntimeError("synthetic contract failure")),
    )
    failed = contract.validate_optimization_contract(tmp_path)
    assert "synthetic contract failure" in failed["errors"]


def test_trial_fold_evidence_recomputes_successful_trial_aggregates() -> None:
    settings = load_optimization_settings()
    fold_specs = [
        {"fold_id": fold_id, "train_rows": 10, "validation_rows": 5} for fold_id in (1, 2, 3)
    ]
    history_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for track in settings.tracks:
        trial_fold_rows = [
            {
                "fold_id": fold_id,
                "average_precision": 0.20 + 0.01 * fold_id,
                "roc_auc": 0.60 + 0.01 * fold_id,
                "log_loss": 0.40 - 0.01 * fold_id,
                "brier_score": 0.20 - 0.005 * fold_id,
                "train_rows": 10,
                "validation_rows": 5,
            }
            for fold_id in (1, 2, 3)
        ]
        aggregate = aggregate_fold_metrics(trial_fold_rows)
        history_rows.append(
            {
                "track": track,
                "trial_number": 0,
                "state": "COMPLETE",
                **_valid_parameters(),
                **aggregate,
                "training_seconds": 0.1,
            }
        )
        fold_rows.extend({"track": track, "trial_number": 0, **row} for row in trial_fold_rows)

    history = pd.DataFrame(history_rows, columns=TRIAL_HISTORY_COLUMNS)
    fold_metrics = pd.DataFrame(fold_rows, columns=TRIAL_FOLD_METRIC_COLUMNS)
    errors: list[str] = []
    evidence = _validate_trial_fold_evidence(
        history, fold_metrics, settings, {"folds": fold_specs}, errors
    )

    assert errors == []
    assert evidence == {
        "required_fold_ids": [1, 2, 3],
        "successful_trial_count": 2,
        "fold_row_count": 6,
        "aggregate_reproduction": "PASS",
    }


def test_trial_fold_evidence_rejects_malformed_schema_and_memberships() -> None:
    settings = load_optimization_settings()
    errors: list[str] = []
    empty = _validate_trial_fold_evidence(
        pd.DataFrame(), pd.DataFrame(), settings, {"folds": []}, errors
    )
    assert empty["aggregate_reproduction"] == "NOT_RUN"
    assert "trial_fold_metrics schema differs" in errors[0]

    history = pd.DataFrame(
        [
            {
                "track": "BAD",
                "trial_number": 0,
                "state": "COMPLETE",
                **_valid_parameters(),
                **aggregate_fold_metrics(
                    [
                        {
                            "average_precision": 0.2,
                            "roc_auc": 0.6,
                            "log_loss": 0.4,
                            "brier_score": 0.2,
                        }
                    ]
                    * 3
                ),
                "training_seconds": 0.1,
            }
        ],
        columns=TRIAL_HISTORY_COLUMNS,
    )
    fold_row = {
        "track": "BAD",
        "trial_number": 0,
        "fold_id": 4,
        "average_precision": 0.2,
        "roc_auc": 0.6,
        "log_loss": 0.4,
        "brier_score": 0.2,
        "train_rows": 10,
        "validation_rows": 5,
    }
    fold_metrics = pd.DataFrame([fold_row, fold_row], columns=TRIAL_FOLD_METRIC_COLUMNS)
    errors = []
    evidence = _validate_trial_fold_evidence(
        history,
        fold_metrics,
        settings,
        {"folds": [{}, "bad", {"fold_id": "not-an-int"}]},
        errors,
    )
    assert evidence["aggregate_reproduction"] == "BLOCKED"
    assert any("unexpected track" in error for error in errors)
    assert any("duplicate" in error for error in errors)
    assert any("outside {1,2,3}" in error for error in errors)
    assert any("non-integer fold ID" in error for error in errors)


def test_winning_trial_reproduction_replays_each_track_without_optuna(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import warranty_analytics_model.catboost_optimization.validation as validation

    settings = load_optimization_settings()
    history_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    fold_metrics = [
        {
            "fold_id": fold_id,
            "average_precision": 0.20 + 0.01 * fold_id,
            "roc_auc": 0.60 + 0.01 * fold_id,
            "log_loss": 0.40 - 0.01 * fold_id,
            "brier_score": 0.20 - 0.005 * fold_id,
            "train_rows": 10,
            "validation_rows": 5,
        }
        for fold_id in (1, 2, 3)
    ]
    aggregate = aggregate_fold_metrics(fold_metrics)
    for track in settings.tracks:
        history_rows.append(
            {
                "track": track,
                "trial_number": 7,
                "state": "COMPLETE",
                **_valid_parameters(),
                **aggregate,
                "training_seconds": 0.1,
            }
        )
        fold_rows.extend({"track": track, "trial_number": 7, **row} for row in fold_metrics)
    history = pd.DataFrame(history_rows, columns=TRIAL_HISTORY_COLUMNS)
    persisted_fold_metrics = pd.DataFrame(fold_rows, columns=TRIAL_FOLD_METRIC_COLUMNS)
    fake_evaluation = SimpleNamespace(fold_metrics=tuple(fold_metrics), aggregate=aggregate)
    monkeypatch.setattr(validation, "select_best_trial", lambda rows: rows.iloc[0].to_dict())
    monkeypatch.setattr(
        validation,
        "evaluate_parameters",
        lambda *args, **kwargs: fake_evaluation,
    )
    inputs = SimpleNamespace(
        development=pd.DataFrame({"warranty_claim_key": [1, 2], "split": ["TRAIN", "TRAIN"]}),
        feature_sets={"E1": object(), "E3": object()},
    )
    replayed = _reproduce_winning_trials(
        inputs,
        pd.DataFrame(),
        history,
        persisted_fold_metrics,
        SimpleNamespace(),
        settings,
        Path.cwd(),
        [],
    )

    assert replayed == {
        "T1": {
            "trial_number": 7,
            "fold_count": 3,
            "mean_average_precision": aggregate["mean_average_precision"],
            "mean_roc_auc": aggregate["mean_roc_auc"],
            "mean_log_loss": aggregate["mean_log_loss"],
            "mean_brier_score": aggregate["mean_brier_score"],
        },
        "T3": {
            "trial_number": 7,
            "fold_count": 3,
            "mean_average_precision": aggregate["mean_average_precision"],
            "mean_roc_auc": aggregate["mean_roc_auc"],
            "mean_log_loss": aggregate["mean_log_loss"],
            "mean_brier_score": aggregate["mean_brier_score"],
        },
    }

    def fail_evaluation(*args: Any, **kwargs: Any) -> Any:
        raise OptimizationError("synthetic replay failure")

    monkeypatch.setattr(validation, "evaluate_parameters", fail_evaluation)
    errors: list[str] = []
    assert (
        _reproduce_winning_trials(
            inputs,
            pd.DataFrame(),
            history,
            persisted_fold_metrics,
            SimpleNamespace(),
            settings,
            Path.cwd(),
            errors,
        )
        == {}
    )
    assert len(errors) == 2


def test_phase10_metric_and_prediction_guards() -> None:
    from warranty_analytics_model.baseline_model.models import BaselineModelError
    from warranty_analytics_model.catboost_optimization.metrics import (
        metrics_for_predictions,
        metrics_payload,
    )

    with pytest.raises(OptimizationError, match="empty"):
        aggregate_fold_metrics([])
    with pytest.raises(OptimizationError, match="threshold"):
        metrics_for_predictions(np.array([0, 1]), np.array([0.2, 0.8]), 0.4)
    zero_baseline = compare_metrics(
        {"average_precision": 0.0, "roc_auc": 0.5, "log_loss": 0.5, "brier_score": 0.3},
        {"average_precision": 0.1, "roc_auc": 0.6, "log_loss": 0.4, "brier_score": 0.2},
    )
    assert zero_baseline["average_precision_relative_lift"] is None
    assert metrics_payload({"T1": zero_baseline})["T1"] == zero_baseline

    with pytest.raises(OptimizationError, match="schema"):
        validate_prediction_frame(pd.DataFrame({"wrong": [1]}), {"candidate"})
    frame = pd.DataFrame(
        {
            "warranty_claim_key": [1],
            "candidate_id": ["other"],
            "high_cost_probability": [0.5],
        }
    )
    with pytest.raises(OptimizationError, match="unexpected candidates"):
        validate_prediction_frame(frame, {"candidate"})
    with pytest.raises(OptimizationError, match="duplicate"):
        validate_prediction_frame(
            pd.concat([frame.assign(candidate_id="candidate")] * 2, ignore_index=True),
            {"candidate"},
        )
    with pytest.raises(BaselineModelError, match="finite"):
        validate_prediction_frame(
            pd.DataFrame(
                {
                    "warranty_claim_key": [1],
                    "candidate_id": ["candidate"],
                    "high_cost_probability": [np.nan],
                }
            ),
            {"candidate"},
        )


def test_search_space_suggestion_uses_exact_allowlist() -> None:
    class FakeTrial:
        def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
            return choices[0]

        def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
            return low

        def suggest_int(self, name: str, low: int, high: int) -> int:
            return low

    assert set(suggest_trial_parameters(FakeTrial())) == set(_valid_parameters())


def test_objective_and_study_run_on_tiny_synthetic_data() -> None:
    metadata, targets = _synthetic_train()
    plan = build_inner_fold_plan(
        metadata,
        targets,
        minimum_train_positive=1,
        minimum_validation_positive=1,
    )
    matrix = pd.DataFrame(
        {"warranty_claim_key": metadata["warranty_claim_key"], "feature": np.sin(metadata.index)}
    )
    feature_set = FeatureSetSpec(
        experiment_id="E1",
        feature_names=("feature",),
        numeric_features=("feature",),
        categorical_features=(),
        boolean_features=(),
        text_features=(),
        phase7_core_count=1,
        phase7_extended_count=0,
        phase8_lexical_count=0,
        phase8_text_count=0,
        feature_set_sha256="tiny",
    )
    settings = load_optimization_settings()
    evaluation = evaluate_parameters(
        matrix,
        targets,
        feature_set,
        plan,
        settings.fixed_parameters,
        _valid_parameters(),
    )
    assert evaluation.aggregate["fold_count"] == 3
    result = run_track_study(
        "T1",
        matrix,
        targets,
        feature_set,
        plan,
        settings.fixed_parameters,
        trials=1,
        seed=20260810,
        n_startup_trials=1,
        threshold=0.5,
    )
    assert result.best_trial_number == 0
    assert len(result.fold_metrics) == 3
    assert tuple(result.trial_history.columns) == TRIAL_HISTORY_COLUMNS
    assert result.trial_history.loc[0, "max_average_precision"] == pytest.approx(
        evaluation.aggregate["max_average_precision"]
    )
    assert result.trial_history.loc[0, "fold_count"] == 3


def test_trial_history_schema_guard_and_failed_run_preservation(tmp_path: Path) -> None:
    import warranty_analytics_model.catboost_optimization.runner as runner

    with pytest.raises(OptimizationError, match="schema differs before publication"):
        require_trial_history_schema(
            pd.DataFrame([{"track": "T1", "trial_number": 0, "state": "COMPLETE"}])
        )

    temporary = tmp_path / ".phase10_test_deadbeef.tmp"
    temporary.mkdir()
    marker = temporary / "trial_history.parquet"
    marker.write_bytes(b"expensive-results")
    failure = OptimizationError("synthetic publication failure")
    preserved = runner._preserve_failed_run(temporary, "phase10_test", failure)

    assert preserved == temporary.with_suffix(".failed")
    assert not temporary.exists()
    assert (preserved / "trial_history.parquet").read_bytes() == b"expensive-results"
    payload = json.loads((preserved / "failure.json").read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED"
    assert payload["error"] == "synthetic publication failure"
    assert payload["run_id"] == "phase10_test"


def test_finalist_builder_writes_only_two_candidate_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import warranty_analytics_model.catboost_optimization.finalists as finalists

    keys = np.arange(1, 49, dtype="int64")
    development = pd.DataFrame(
        {
            "warranty_claim_key": keys,
            "split": ["TRAIN"] * 32 + ["VALIDATION"] * 16,
            "feature": np.linspace(0.0, 1.0, len(keys)),
        }
    )
    targets = pd.DataFrame(
        {
            "warranty_claim_key": keys,
            "target__high_cost_claim_flag": np.tile([0, 1], 24),
        }
    )
    phase9_dir = tmp_path / "phase9"
    phase9_dir.mkdir()
    baseline = pd.DataFrame(
        {
            "warranty_claim_key": np.tile(keys[32:], 2),
            "experiment_id": ["E1"] * 16 + ["E3"] * 16,
            "probability": np.tile(np.linspace(0.1, 0.9, 16), 2),
        }
    )
    baseline.to_parquet(phase9_dir / "validation_predictions.parquet", index=False)
    feature_set = FeatureSetSpec(
        "E1",
        ("feature",),
        ("feature",),
        (),
        (),
        (),
        1,
        0,
        0,
        0,
        "tiny",
    )
    phase10_inputs = Phase10Inputs(
        root=Path.cwd(),
        phase9_dir=phase9_dir,
        phase9_manifest={},
        phase9_inputs=None,  # type: ignore[arg-type]
        feature_sets={"E1": feature_set, "E3": feature_set},
        development=development,
        claim_snapshot_path=tmp_path / "snapshot.parquet",
    )
    study = StudyResult(
        track="T1",
        phase9_experiment_id="E1",
        study_name="tiny",
        trial_history=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        baseline_inner_cv_metrics={},
        best_trial_number=0,
        best_params=_valid_parameters(),
        best_inner_metrics={"std_average_precision": 0.01},
        best_param_sha256=parameter_sha256(_valid_parameters()),
    )

    class FakeModel:
        def predict_proba(self, pool: Any) -> np.ndarray:
            probabilities = np.linspace(0.1, 0.9, pool.num_row())
            return np.column_stack([1.0 - probabilities, probabilities])

    def fake_fit(*args: Any, **kwargs: Any) -> tuple[FakeModel, float]:
        return FakeModel(), 0.01

    monkeypatch.setattr(finalists, "fit_model", fake_fit)
    monkeypatch.setattr(finalists, "save_model", lambda model, path: path.write_bytes(b"model"))
    monkeypatch.setattr(finalists, "effective_parameters", lambda model: {"task_type": "CPU"})
    predictions, metrics, baseline_metrics, model_manifest = fit_phase10_finalists(
        phase10_inputs,
        targets.iloc[:32].copy(),
        targets.iloc[32:].copy(),
        {"T1": study, "T3": study},
        load_optimization_settings().fixed_parameters,
        tmp_path / "models",
    )
    assert set(predictions["candidate_id"]) == {"P10_T1_E1_OPTIMIZED", "P10_T3_E3_OPTIMIZED"}
    assert len(metrics) == 2
    assert len(baseline_metrics) == 2
    assert len(model_manifest["models"]) == 2


def test_standalone_validator_reports_missing_artifacts(tmp_path: Path) -> None:
    result = validate_optimization_directory(tmp_path)
    assert result["valid"] is False
    assert result["hardening_status"] == "BLOCKED"


def test_runner_publishes_atomic_run_with_frozen_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    import warranty_analytics_model.catboost_optimization.runner as runner

    settings = replace(
        load_optimization_settings(),
        output_directory="out",
        report_directory="reports",
    )
    feature_set = FeatureSetSpec(
        "E1",
        ("feature",),
        ("feature",),
        (),
        (),
        (),
        1,
        0,
        0,
        0,
        "tiny",
    )
    keys = np.arange(1, 9, dtype="int64")
    development = pd.DataFrame(
        {
            "warranty_claim_key": keys,
            "split": ["TRAIN"] * 6 + ["VALIDATION"] * 2,
            "feature": np.linspace(0.0, 1.0, len(keys)),
        }
    )
    targets = pd.DataFrame(
        {
            "warranty_claim_key": keys[:6],
            "target__high_cost_claim_flag": [0, 1, 0, 1, 0, 1],
        }
    )
    validation_targets = pd.DataFrame(
        {
            "warranty_claim_key": keys[6:],
            "target__high_cost_claim_flag": [0, 1],
        }
    )
    phase10_inputs = Phase10Inputs(
        root=tmp_path,
        phase9_dir=tmp_path / "locked",
        phase9_manifest={
            "run_id": "20260811T_PHASE9_FINAL",
            "input_hashes": {},
            "frozen_membership": {"counts": {"TRAIN": 6, "VALIDATION": 2, "TEST": 0}},
        },
        phase9_inputs=None,  # type: ignore[arg-type]
        feature_sets={"E1": feature_set, "E3": feature_set},
        development=development,
        claim_snapshot_path=tmp_path / "snapshot.parquet",
    )
    fold_plan = SimpleNamespace(
        assignments=pd.DataFrame(
            {
                "warranty_claim_key": [1, 2],
                "claim_date": ["2024-01-01", "2024-01-02"],
                "fold_id": [1, 1],
                "role": ["TRAIN", "VALIDATION"],
            }
        ),
        manifest={"fold_count": 3, "source_outer_split": "TRAIN"},
        content_sha256="fold-hash",
    )
    trial_history = pd.DataFrame(
        [
            {
                "track": "T1",
                "trial_number": 0,
                "state": "COMPLETE",
                **_valid_parameters(),
                "mean_average_precision": 0.5,
                "min_average_precision": 0.5,
                "max_average_precision": 0.5,
                "std_average_precision": 0.0,
                "mean_roc_auc": 0.7,
                "min_roc_auc": 0.7,
                "mean_log_loss": 0.4,
                "mean_brier_score": 0.2,
                "fold_count": 3,
                "training_seconds": 0.01,
            }
        ],
        columns=TRIAL_HISTORY_COLUMNS,
    )
    fold_metrics = pd.DataFrame(
        [
            {
                "track": "T1",
                "trial_number": 0,
                "fold_id": 1,
                "average_precision": 0.5,
            }
        ]
    )

    def study_for(track: str) -> StudyResult:
        return StudyResult(
            track=track,
            phase9_experiment_id=TRACK_TO_EXPERIMENT[track],
            study_name=f"phase10_{track}",
            trial_history=trial_history.assign(track=track),
            fold_metrics=fold_metrics.assign(track=track),
            baseline_inner_cv_metrics={"mean_average_precision": 0.4},
            best_trial_number=0,
            best_params=_valid_parameters(),
            best_inner_metrics={"std_average_precision": 0.01},
            best_param_sha256=parameter_sha256(_valid_parameters()),
        )

    def metric(average_precision: float) -> dict[str, float]:
        return {
            "average_precision": average_precision,
            "pr_auc_trapezoidal": average_precision,
            "roc_auc": 0.7,
            "log_loss": 0.4,
            "brier_score": 0.2,
        }

    comparisons = {
        track: {
            "baseline_average_precision": 0.4,
            "optimized_average_precision": 0.5,
            "optimized_beats_baseline": True,
            "fallback_to_baseline": False,
        }
        for track in ("T1", "T3")
    }

    def fake_finalists(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        predictions = pd.DataFrame(
            {
                "warranty_claim_key": [7, 8, 7, 8],
                "candidate_id": [
                    "P10_T1_E1_OPTIMIZED",
                    "P10_T1_E1_OPTIMIZED",
                    "P10_T3_E3_OPTIMIZED",
                    "P10_T3_E3_OPTIMIZED",
                ],
                "high_cost_probability": [0.1, 0.9, 0.2, 0.8],
            }
        )
        return (
            predictions,
            {"P10_T1_E1_OPTIMIZED": metric(0.5), "P10_T3_E3_OPTIMIZED": metric(0.5)},
            {"P9_E1_BASELINE": metric(0.4), "P9_E3_BASELINE": metric(0.4)},
            {"models": {}, "comparisons": comparisons},
        )

    monkeypatch.setattr(runner, "discover_repository_root", lambda root=None: tmp_path)
    monkeypatch.setattr(runner, "load_optimization_settings", lambda root=None: settings)
    monkeypatch.setattr(
        runner,
        "validate_optimization_contract",
        lambda root: {"valid": True, "errors": [], "warnings": []},
    )
    monkeypatch.setattr(runner, "runtime_provenance", lambda: {"python_version": "3.12.13"})
    monkeypatch.setattr(
        runner,
        "validate_runtime_dependency_constraints",
        lambda *args, **kwargs: {"valid": True, "errors": []},
    )
    monkeypatch.setattr(
        runner,
        "phase10_plan_check",
        lambda *args, **kwargs: {
            "valid": True,
            "errors": [],
            "inputs": phase10_inputs,
            "train_targets": targets,
            "inner_fold_plan": fold_plan,
        },
    )
    monkeypatch.setattr(
        runner,
        "_baseline_inner_metrics",
        lambda *args, **kwargs: {
            "T1": {"mean_average_precision": 0.4},
            "T3": {"mean_average_precision": 0.4},
        },
    )
    monkeypatch.setattr(runner, "run_track_study", lambda track, *args, **kwargs: study_for(track))
    monkeypatch.setattr(
        runner,
        "load_train_targets_for_optimization",
        lambda inputs: (targets, {"target_content_sha256": "train-hash"}),
    )
    monkeypatch.setattr(
        runner,
        "load_validation_targets_after_freeze",
        lambda inputs, study_frozen: (
            validation_targets,
            {"target_content_sha256": "validation-hash"},
        ),
    )
    monkeypatch.setattr(runner, "fit_phase10_finalists", fake_finalists)
    monkeypatch.setattr(runner, "git_commit_sha", lambda root: "test-commit")
    monkeypatch.setattr(
        runner,
        "validate_optimization_directory",
        lambda *args, **kwargs: {
            "status": "PASS",
            "valid": True,
            "errors": [],
            "warnings": [],
            "hardening_status": "HARDENED_PASS",
        },
    )

    summary = runner.build_phase10(
        tmp_path / "locked",
        run_id="20260811T_PHASE10_TEST",
        project_root=tmp_path,
    )
    output = tmp_path / "out" / "20260811T_PHASE10_TEST"
    assert summary["status"] == "PASS"
    assert summary["phase10_development_champion"] == "P10_T1_E1_OPTIMIZED"
    assert (output / "optimization_manifest.json").is_file()
    assert not list((tmp_path / "out").glob("*.tmp"))


def test_plan_check_is_fail_closed_and_returns_fold_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import warranty_analytics_model.catboost_optimization.runner as runner

    settings = load_optimization_settings()
    sentinel_inputs = object()
    train_targets = pd.DataFrame({"warranty_claim_key": [1], "target__high_cost_claim_flag": [0]})
    fold_plan = SimpleNamespace(content_sha256="fold", manifest={}, assignments=pd.DataFrame())
    monkeypatch.setattr(runner, "discover_repository_root", lambda root=None: tmp_path)
    monkeypatch.setattr(runner, "load_optimization_settings", lambda root=None: settings)
    monkeypatch.setattr(
        runner,
        "validate_optimization_contract",
        lambda root: {"valid": True, "errors": [], "warnings": []},
    )
    monkeypatch.setattr(
        runner, "load_locked_phase9_inputs", lambda *args, **kwargs: sentinel_inputs
    )
    monkeypatch.setattr(
        runner,
        "load_train_targets_for_optimization",
        lambda inputs: (train_targets, {"test_target_rows_loaded": 0}),
    )
    monkeypatch.setattr(runner, "_fold_plan", lambda *args, **kwargs: fold_plan)
    result = runner.phase10_plan_check(tmp_path / "phase9", project_root=tmp_path)
    assert result["valid"] is True
    assert result["inner_fold_plan"] is fold_plan

    monkeypatch.setattr(
        runner,
        "load_locked_phase9_inputs",
        lambda *args, **kwargs: (_ for _ in ()).throw(OptimizationError("locked input failed")),
    )
    blocked = runner.phase10_plan_check(tmp_path / "phase9", project_root=tmp_path)
    assert blocked["valid"] is False
    assert "locked input failed" in blocked["errors"]


def test_runner_helpers_cover_fold_plan_contract_and_warning_policy() -> None:
    from dataclasses import replace

    import warranty_analytics_model.catboost_optimization.runner as runner

    metadata, targets = _synthetic_train()
    development = metadata.rename(columns={"claim_date": "claim__claim_date"}).assign(split="TRAIN")
    inputs = Phase10Inputs(
        root=Path.cwd(),
        phase9_dir=Path.cwd(),
        phase9_manifest={},
        phase9_inputs=None,  # type: ignore[arg-type]
        feature_sets={},
        development=development,
        claim_snapshot_path=Path.cwd(),
    )
    settings = replace(
        load_optimization_settings(),
        minimum_train_positive=1,
        minimum_validation_positive=1,
    )
    plan = runner._fold_plan(inputs, targets, settings)
    assert len(plan.folds) == 3
    assert runner.phase10_run_id().endswith("Z")
    assert runner.phase10_contract_check()["valid"] is True

    study = StudyResult(
        track="T1",
        phase9_experiment_id="E1",
        study_name="warnings",
        trial_history=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        baseline_inner_cv_metrics={},
        best_trial_number=0,
        best_params=_valid_parameters(),
        best_inner_metrics={"std_average_precision": 0.2},
        best_param_sha256=parameter_sha256(_valid_parameters()),
        warnings=["TRIAL_WARNING"],
    )
    warnings = runner._optimization_warnings(
        {"T1": study, "T3": replace(study, track="T3", phase9_experiment_id="E3")},
        {
            "T1": {
                "optimized_average_precision": 0.1,
                "baseline_average_precision": 0.2,
                "optimized_beats_baseline": False,
            },
            "T3": {
                "optimized_average_precision": 0.1,
                "baseline_average_precision": 0.2,
                "optimized_beats_baseline": False,
            },
        },
        {"T1": {"roc_auc": 0.99, "average_precision": 0.1}},
        settings,
    )
    assert {"NO_OPTIMIZATION_GAIN", "INNER_CV_INSTABILITY", "TRIAL_WARNING"}.issubset(warnings)


def test_standalone_validator_accepts_recomputed_fake_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from dataclasses import replace

    import warranty_analytics_model.catboost_optimization.validation as validation
    from warranty_analytics_model.catboost_optimization.metrics import metrics_for_predictions
    from warranty_analytics_model.catboost_optimization.provenance import fold_content_sha256
    from warranty_analytics_model.catboost_optimization.study import study_history_sha256
    from warranty_analytics_model.feature_mart.manifest import sha256_file

    def write_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    settings = replace(load_optimization_settings(), trials_per_track=1)
    feature_set = FeatureSetSpec(
        "E1",
        ("feature",),
        ("feature",),
        (),
        (),
        (),
        1,
        0,
        0,
        0,
        "tiny",
    )
    development = pd.DataFrame(
        {
            "warranty_claim_key": [1, 2, 3, 4],
            "split": ["TRAIN", "TRAIN", "VALIDATION", "VALIDATION"],
            "claim__claim_date": pd.date_range("2024-01-01", periods=4),
            "feature": [0.0, 1.0, 2.0, 3.0],
        }
    )
    train_targets = pd.DataFrame(
        {
            "warranty_claim_key": [1, 2],
            "target__high_cost_claim_flag": [0, 1],
        }
    )
    validation_targets = pd.DataFrame(
        {
            "warranty_claim_key": [3, 4],
            "target__high_cost_claim_flag": [0, 1],
        }
    )
    inputs = Phase10Inputs(
        root=Path.cwd(),
        phase9_dir=tmp_path / "phase9",
        phase9_manifest={
            "run_id": REQUIRED_PHASE9_RUN_ID,
            "target_hashes": dict(EXPECTED_PHASE9_TARGET_HASHES),
        },
        phase9_inputs=None,  # type: ignore[arg-type]
        feature_sets={"E1": feature_set, "E3": feature_set},
        development=development,
        claim_snapshot_path=tmp_path / "snapshot.parquet",
    )

    class FakeModel:
        def predict_proba(self, pool: Any) -> np.ndarray:
            probabilities = np.array([0.2, 0.8], dtype="float64")[: pool.num_row()]
            return np.column_stack([1.0 - probabilities, probabilities])

    search_parameters = _valid_parameters()
    model_parameters = {**settings.fixed_parameters, **search_parameters}
    optimized_metrics = metrics_for_predictions(
        np.array([0, 1], dtype="int8"), np.array([0.2, 0.8]), 0.5
    )

    run_dir = tmp_path / "optimization"
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True)
    phase9_dir = tmp_path / "phase9"
    phase9_dir.mkdir()
    baseline_predictions = pd.DataFrame(
        {
            "warranty_claim_key": [3, 4, 3, 4],
            "experiment_id": ["E1", "E1", "E3", "E3"],
            "probability": [0.1, 0.9, 0.1, 0.9],
        }
    )
    baseline_predictions.to_parquet(phase9_dir / "validation_predictions.parquet", index=False)

    optimized_predictions = pd.DataFrame(
        {
            "warranty_claim_key": [3, 4, 3, 4],
            "candidate_id": [
                "P10_T1_E1_OPTIMIZED",
                "P10_T1_E1_OPTIMIZED",
                "P10_T3_E3_OPTIMIZED",
                "P10_T3_E3_OPTIMIZED",
            ],
            "high_cost_probability": [0.2, 0.8, 0.2, 0.8],
        }
    )
    optimized_predictions.to_parquet(run_dir / "validation_predictions.parquet", index=False)
    write_json(
        run_dir / "validation_metrics.json",
        {
            "candidate_metrics": {
                "P10_T1_E1_OPTIMIZED": optimized_metrics,
                "P10_T3_E3_OPTIMIZED": optimized_metrics,
            },
            "phase10_development_champion": "P9_E1_BASELINE",
        },
    )

    fold_frame = pd.DataFrame(
        {
            "warranty_claim_key": [1, 2],
            "claim_date": ["2024-01-01", "2024-01-02"],
            "fold_id": [1, 1],
            "role": ["TRAIN", "VALIDATION"],
        }
    )
    fold_hash = fold_content_sha256(fold_frame)
    fold_frame.to_parquet(run_dir / "inner_cv_folds.parquet", index=False)
    write_json(
        run_dir / "inner_cv_manifest.json",
        {
            "source_outer_split": "TRAIN",
            "fold_count": 3,
            "fold_content_sha256": fold_hash,
            "folds": [
                {
                    "fold_id": 1,
                    "train_max_date": "2024-01-01",
                    "validation_min_date": "2024-01-02",
                }
            ],
        },
    )
    history_rows = []
    best_inner_metrics = {
        "mean_average_precision": 0.5,
        "min_average_precision": 0.5,
        "std_average_precision": 0.0,
        "mean_roc_auc": 0.7,
        "min_roc_auc": 0.7,
        "mean_log_loss": 0.4,
        "mean_brier_score": 0.2,
    }
    for track in ("T1", "T3"):
        history_rows.append(
            {
                "track": track,
                "trial_number": 0,
                "state": "COMPLETE",
                **search_parameters,
                "mean_average_precision": best_inner_metrics["mean_average_precision"],
                "min_average_precision": best_inner_metrics["min_average_precision"],
                "max_average_precision": 0.5,
                "std_average_precision": best_inner_metrics["std_average_precision"],
                "mean_roc_auc": best_inner_metrics["mean_roc_auc"],
                "min_roc_auc": best_inner_metrics["min_roc_auc"],
                "mean_log_loss": best_inner_metrics["mean_log_loss"],
                "mean_brier_score": best_inner_metrics["mean_brier_score"],
                "fold_count": 3,
                "training_seconds": 0.01,
            }
        )
    history = pd.DataFrame(history_rows, columns=TRIAL_HISTORY_COLUMNS)
    history.to_parquet(run_dir / "trial_history.parquet", index=False)
    pd.DataFrame(
        columns=[
            "track",
            "trial_number",
            "fold_id",
            "average_precision",
            "roc_auc",
            "log_loss",
            "brier_score",
            "train_rows",
            "validation_rows",
        ]
    ).to_parquet(run_dir / "trial_fold_metrics.parquet", index=False)
    freeze = {
        "phase": 10,
        "tracks": {
            track: {
                "study_name": f"phase10_{track}",
                "best_trial_number": 0,
                "best_params": search_parameters,
                "best_inner_metrics": best_inner_metrics,
                "baseline_inner_cv_metrics": {},
            }
            for track in ("T1", "T3")
        },
        "outer_validation_accessed": False,
        "inner_fold_content_sha256": fold_hash,
        "trial_history_content_sha256": study_history_sha256(history),
    }
    from warranty_analytics_model.catboost_optimization.manifest import freeze_payload_sha256

    freeze["study_freeze_sha256"] = freeze_payload_sha256(freeze)
    write_json(run_dir / "study_freeze.json", freeze)

    model_entries: dict[str, dict[str, Any]] = {}
    for candidate_id, filename in (
        ("P10_T1_E1_OPTIMIZED", "t1_e1_optimized.cbm"),
        ("P10_T3_E3_OPTIMIZED", "t3_e3_optimized.cbm"),
    ):
        model_path = model_dir / filename
        model_path.write_bytes(b"fake-model")
        model_entries[candidate_id] = {
            "model_file": f"models/{filename}",
            "model_sha256": sha256_file(model_path),
            "best_params": search_parameters,
            "model_parameters": model_parameters,
        }
    write_json(run_dir / "model_manifest.json", {"models": model_entries})
    write_json(
        run_dir / "target_access_audit.json",
        {
            "test_target_rows_loaded": 0,
            "test_predictions_created": 0,
            "test_metrics_computed": False,
            "test_target_access_allowed": False,
            "first_allowed_test_target_phase": 15,
            "outer_validation_accessed_before_study_freeze": False,
        },
    )
    write_json(run_dir / "validation.json", {})

    artifact_names = [
        "inner_cv_folds.parquet",
        "inner_cv_manifest.json",
        "trial_history.parquet",
        "trial_fold_metrics.parquet",
        "study_freeze.json",
        "best_params.json",
        "validation_predictions.parquet",
        "validation_metrics.json",
        "target_access_audit.json",
        "model_manifest.json",
    ]
    write_json(
        run_dir / "best_params.json",
        {track: {"best_params": search_parameters} for track in ("T1", "T3")},
    )
    manifest = {
        "phase": 10,
        "contract_version": load_optimization_contract()[0]["phase10"]["version"],
        "contract_checksum": load_optimization_contract()[1],
        "contract_policy_snapshot": load_optimization_contract()[0]["phase10"],
        "phase9_dir": str(phase9_dir),
        "phase9_run_id": REQUIRED_PHASE9_RUN_ID,
        "phase9_hardened_status": "HARDENED_PASS",
        "phase9_target_hashes": dict(EXPECTED_PHASE9_TARGET_HASHES),
        "phase9_feature_set_hashes": {"T1": "tiny", "T3": "tiny"},
        "outer_validation_accessed": True,
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
        "inner_fold_content_sha256": fold_hash,
        "settings": settings_payload(settings),
        "trials_per_track": 1,
        "objective_metric": "mean_average_precision",
        "warnings": [],
        "artifact_file_sha256": {name: sha256_file(run_dir / name) for name in artifact_names},
    }
    write_json(run_dir / "optimization_manifest.json", manifest)

    monkeypatch.setattr(validation, "discover_repository_root", lambda root=None: Path.cwd())
    monkeypatch.setattr(validation, "load_optimization_settings", lambda root=None: settings)
    monkeypatch.setattr(validation, "load_locked_phase9_inputs", lambda *args, **kwargs: inputs)
    from types import SimpleNamespace

    monkeypatch.setattr(
        validation,
        "build_inner_fold_plan",
        lambda *args, **kwargs: SimpleNamespace(content_sha256=fold_hash, assignments=fold_frame),
    )
    monkeypatch.setattr(
        validation,
        "_validate_trial_fold_evidence",
        lambda *args, **kwargs: {
            "required_fold_ids": [1, 2, 3],
            "successful_trial_count": 2,
            "fold_row_count": 0,
            "aggregate_reproduction": "PASS",
        },
    )
    monkeypatch.setattr(validation, "_reproduce_winning_trials", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        validation,
        "validate_model_directory",
        lambda *args, **kwargs: {"errors": [], "hardening_status": "HARDENED_PASS"},
    )
    monkeypatch.setattr(
        validation,
        "load_train_targets_for_optimization",
        lambda phase10_inputs: (train_targets, {}),
    )
    monkeypatch.setattr(
        validation,
        "load_validation_targets_after_freeze",
        lambda phase10_inputs, study_frozen: (validation_targets, {}),
    )
    monkeypatch.setattr(validation, "load_model", lambda path: FakeModel())
    monkeypatch.setattr(validation, "effective_parameters", lambda model: model_parameters)

    result = validation.validate_optimization_directory(run_dir, project_root=Path.cwd())
    assert result["valid"] is True, result["errors"]
    assert result["hardening_status"] == "HARDENED_PASS"
