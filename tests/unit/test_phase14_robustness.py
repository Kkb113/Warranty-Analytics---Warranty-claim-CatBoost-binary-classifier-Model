"""Unit coverage for the Phase 14 frozen-model diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from warranty_analytics_model.baseline_model.models import FeatureSetSpec
from warranty_analytics_model.robustness_analysis import input as input_module
from warranty_analytics_model.robustness_analysis import runner as runner_module
from warranty_analytics_model.robustness_analysis import validation as validation_module
from warranty_analytics_model.robustness_analysis.bootstrap import (
    material_slice_bootstrap,
    stratified_bootstrap,
)
from warranty_analytics_model.robustness_analysis.checkpoint import (
    checkpoint_sha,
    load_checkpoint,
    write_checkpoint,
)
from warranty_analytics_model.robustness_analysis.config import (
    Phase14ConfigurationError,
    compute_plan,
    load_robustness_settings,
)
from warranty_analytics_model.robustness_analysis.contract import phase14_contract_check
from warranty_analytics_model.robustness_analysis.drift import feature_drift, score_drift
from warranty_analytics_model.robustness_analysis.errors import (
    build_error_context,
    error_cohorts,
    error_profile,
    high_confidence_errors,
)
from warranty_analytics_model.robustness_analysis.invariance import prediction_invariance
from warranty_analytics_model.robustness_analysis.leakage import leakage_recheck
from warranty_analytics_model.robustness_analysis.metrics import (
    overall_metrics,
    safe_metric_dict,
    support_status,
)
from warranty_analytics_model.robustness_analysis.planning import build_analysis_plan
from warranty_analytics_model.robustness_analysis.ranking import risk_decile_metrics, topk_lift
from warranty_analytics_model.robustness_analysis.readiness import readiness_gate
from warranty_analytics_model.robustness_analysis.reporting import write_phase14_reports
from warranty_analytics_model.robustness_analysis.slices import (
    evaluate_slices,
    membership_for_definition,
)
from warranty_analytics_model.robustness_analysis.temporal import temporal_metrics
from warranty_analytics_model.robustness_analysis.threshold_diagnostics import threshold_sensitivity
from warranty_analytics_model.robustness_analysis.validation import validate_existing_phase14

KEY = "warranty_claim_key"
TARGET = "target__high_cost_claim_flag"


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            KEY: np.arange(1, 13),
            "claim__claim_date": pd.date_range("2024-01-01", periods=12, freq="MS"),
            "f_num": np.arange(12, dtype="float64"),
            "f_cat": ["A", "A", "B", "B", "C", None, "A", "B", "C", "A", "B", "C"],
            "f_bool": [True, False] * 6,
        }
    )


def _settings():
    return load_robustness_settings(Path.cwd())


def test_contract_config_and_compute_guards() -> None:
    result = phase14_contract_check(Path.cwd())
    assert result["valid"] is True
    settings = _settings()
    plan = compute_plan(settings, max_workers=1, catboost_inference_threads=1)
    assert plan["seed"] == 20260810
    assert plan["overall_bootstrap_replicates"] == 2000
    with pytest.raises(Phase14ConfigurationError):
        compute_plan(settings, max_workers=1, bootstrap_replicates=10)
    with pytest.raises(Phase14ConfigurationError):
        compute_plan(settings, max_workers=1, catboost_inference_threads=10_000)
    with pytest.raises(Phase14ConfigurationError):
        compute_plan(settings, max_workers=0)
    with pytest.raises(Phase14ConfigurationError):
        compute_plan(settings, catboost_inference_threads=0)


def test_metrics_support_bootstrap_and_thresholds_are_deterministic() -> None:
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1], dtype="int8")
    p = np.array([0.02, 0.10, 0.20, 0.55, 0.70, 0.90, 0.04, 0.60])
    metrics = overall_metrics(y, p, 0.5)
    assert metrics["primary_signal_pass"] is True
    assert (
        support_status(8, 4, 4, min_rows=4, min_positive_ranking=2, min_negative_ranking=2)
        == "SUPPORTED"
    )
    assert support_status(3, 1, 2) == "LOW_SUPPORT"
    assert safe_metric_dict(np.ones(2, dtype="int8"), [0.2, 0.3], 0.5)["status"] == "LOW_SUPPORT"
    summary_a, rows_a = stratified_bootstrap(y, p, 0.5, replicates=12, seed=7, workers=1)
    summary_b, rows_b = stratified_bootstrap(y, p, 0.5, replicates=12, seed=7, workers=2)
    assert rows_a == rows_b
    assert summary_a == summary_b
    table = threshold_sensitivity(y, p, 0.5)
    assert len(table) == 5
    assert table["DO_NOT_USE_FOR_THRESHOLD_SELECTION"].all()


def test_drift_ranking_errors_and_invariance() -> None:
    frame = _frame()
    train = frame.iloc[:8].copy()
    validation = frame.iloc[4:].copy()
    drift = feature_drift(train, validation, ["f_num", "f_bool", "f_cat", "missing"], {"f_cat"})
    assert set(drift["feature"]) == {"f_num", "f_bool", "f_cat", "missing"}
    assert drift.loc[drift["feature"] == "f_bool", "status"].item() == "AVAILABLE"
    summary, shift = score_drift([0.1, 0.2, 0.3], [0.2, 0.3, 0.4])
    assert summary["train_p50"] == 0.2
    assert shift["classification"] in {"LOW_SHIFT", "SCORE_DISTRIBUTION_SHIFT"}

    y = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1], dtype="int8")
    p = np.array([0.8, 0.1, 0.2, 0.7, 0.3, 0.2, 0.6, 0.1, 0.9, 0.2, 0.1, 0.8])
    oof = pd.DataFrame({KEY: frame[KEY], "probability": p})
    deciles = risk_decile_metrics(oof, frame, y, pd.Series(p))
    assert not deciles.empty
    assert topk_lift(frame[KEY], y, p)["rows"][0]["claims_selected"] == 1
    cohorts = error_cohorts(frame[KEY], y, p, 0.5)
    high = high_confidence_errors(cohorts, limit=2)
    profile = error_profile(cohorts, frame[[KEY, "claim__claim_date"]])
    assert {"FALSE_POSITIVE", "FALSE_NEGATIVE"}.issubset(set(cohorts["error_type"]))
    assert len(high) <= 4
    assert not profile.empty

    by_key = dict(zip(frame[KEY].astype(int), p, strict=True))
    invariant = prediction_invariance(
        frame,
        lambda batch: batch[[KEY]].assign(probability=batch[KEY].map(by_key).to_numpy()),
        batch_sizes=(3, 5),
        seed=4,
    )
    assert invariant["valid"] is True
    assert invariant["serialization_max_probability_delta"] == 0.0
    assert invariant["serialization_reload_verified"] is False


def test_phase14_frozen_diagnostics_order_and_classification() -> None:
    frame = _frame()
    y = pd.Series([0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1])
    probabilities = pd.Series(
        [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.09, 0.08, 0.07]
    )
    overall = overall_metrics(y, probabilities, 0.5)
    temporal = temporal_metrics(frame, y, probabilities, 0.5, _settings(), overall=overall)
    assert set(temporal["stability_classification"]) <= {
        "LOW_SUPPORT",
        "STABLE",
        "MODERATE_DEGRADATION",
        "SEVERE_DEGRADATION",
    }
    oof = pd.DataFrame({KEY: frame[KEY], "probability": probabilities})
    deciles = risk_decile_metrics(oof, frame, y, probabilities)
    assert deciles.iloc[0]["decile"] == "D10"
    assert deciles.iloc[0]["cumulative_recall"] == deciles.iloc[0]["recall_contribution"]
    train = pd.DataFrame({"f_cat": ["A"] * 10})
    shifted = pd.DataFrame({"f_cat": ["B"] * 10})
    categorical = feature_drift(train, shifted, ["f_cat"], {"f_cat"}).iloc[0]
    assert categorical["psi"] > 0.25
    assert categorical["psi_classification"] == "HIGH_SHIFT"
    assert "js_distance" in categorical


def test_material_slice_bootstrap_and_error_context() -> None:
    frame = _frame().copy()
    y = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0])
    probabilities = pd.Series(np.linspace(0.01, 0.9, len(frame)))
    definitions = [
        {
            "slice_id": "cat",
            "kind": "categorical",
            "column": "f_cat",
            "categories": ["A", "B", "__MISSING__", "__OTHER__"],
        }
    ]
    slices, _ = evaluate_slices(
        definitions,
        frame,
        y,
        probabilities,
        0.5,
        overall_metrics(y, probabilities, 0.5),
        _settings(),
    )
    slices["status"] = "SUPPORTED"
    summary, rows = material_slice_bootstrap(
        slices,
        definitions,
        frame,
        y,
        probabilities,
        0.5,
        replicates=3,
        seed=7,
        workers=1,
        confidence_level=0.95,
        min_positive=1,
        min_negative=1,
    )
    assert summary["eligible_slice_count"] >= 1
    assert len(rows) == summary["eligible_slice_count"] * 3
    context = build_error_context(frame, definitions, probabilities, ("f_num", "f_cat"))
    profile = error_profile(error_cohorts(frame[KEY], y, probabilities, 0.5), context)
    assert {"aggregate", "numeric_summary", "categorical_cohort"}.issubset(
        set(profile["profile_kind"])
    )


def test_resume_plan_binding_ignores_creation_time() -> None:
    plan = {
        "analysis_plan_sha256": "plan",
        "slice_registry_sha256": "registry",
        "slice_definition_sha256": "definitions",
        "temporal_definition_sha256": "temporal",
        "configuration_sha256": "configuration",
        "created_at_utc": "2024-01-01T00:00:00Z",
    }
    changed = {**plan, "created_at_utc": "2025-01-01T00:00:00Z"}
    assert runner_module._plan_resume_bindings(plan) == runner_module._plan_resume_bindings(changed)
    assert runner_module._plan_resume_bindings(
        {**changed, "slice_definition_sha256": "stale"}
    ) != runner_module._plan_resume_bindings(plan)


def test_train_oof_obeys_frozen_raw_score_space(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            KEY: [1, 2],
            "track": ["T1", "T1"],
            "candidate_id": ["C1", "C1"],
            "raw_probability": [0.10, 0.20],
            "calibrated_probability": [0.60, 0.70],
        }
    ).to_parquet(tmp_path / "selected_calibrated_oof_predictions.parquet", index=False)
    resolved = SimpleNamespace(
        phase13_dir=tmp_path,
        phase13_manifest={"phase12_run_id": "P12"},
        champion_type="SINGLE_TRACK",
        champion_id="C1",
        score_space="RAW_UNCALIBRATED_PROBABILITY",
        components=(SimpleNamespace(track="T1"),),
    )
    assert input_module.train_oof_scores(resolved)["probability"].tolist() == [0.10, 0.20]


def test_train_oof_obeys_calibrated_single_and_ensemble_spaces(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            KEY: [1, 2, 1, 2],
            "track": ["T1", "T1", "T3", "T3"],
            "candidate_id": ["C1"] * 4,
            "raw_probability": [0.10, 0.20, 0.30, 0.40],
            "calibrated_probability": [0.60, 0.70, 0.80, 0.90],
        }
    )
    frame.to_parquet(tmp_path / "selected_calibrated_oof_predictions.parquet", index=False)
    single = SimpleNamespace(
        phase13_dir=tmp_path,
        phase13_manifest={"phase12_run_id": "P12"},
        champion_type="SINGLE_TRACK",
        champion_id="C1",
        score_space="CALIBRATED_PROBABILITY",
        components=(SimpleNamespace(track="T1"),),
    )
    assert input_module.train_oof_scores(single)["probability"].tolist() == [0.60, 0.70]
    ensemble = SimpleNamespace(
        phase13_dir=tmp_path,
        phase13_manifest={"phase12_run_id": "P12"},
        champion_type="ENSEMBLE",
        champion_id="ENSEMBLE",
        score_space="CALIBRATED_ENSEMBLE_PROBABILITY",
        ensemble_t1_weight=0.25,
        components=(SimpleNamespace(track="T1"), SimpleNamespace(track="T3")),
    )
    assert np.allclose(input_module.train_oof_scores(ensemble)["probability"], [0.75, 0.85])


def test_plan_slices_temporal_and_leakage(tmp_path: Path) -> None:
    frame = _frame()
    spec = FeatureSetSpec(
        experiment_id="E1",
        feature_names=("f_num", "f_cat", "f_bool"),
        numeric_features=("f_num", "f_bool"),
        categorical_features=("f_cat",),
        boolean_features=("f_bool",),
        text_features=(),
        phase7_core_count=3,
        phase7_extended_count=0,
        phase8_lexical_count=0,
        phase8_text_count=0,
        feature_set_sha256="sha",
    )
    phase13_dir = tmp_path / "phase13"
    phase13_dir.mkdir()
    pd.DataFrame(
        {KEY: frame[KEY], "effective_probability": np.linspace(0.01, 0.2, len(frame))}
    ).to_parquet(phase13_dir / "selected_calibrated_oof_predictions.parquet", index=False)

    @dataclass
    class FakeResolved:
        train_features: pd.DataFrame
        validation_features: pd.DataFrame
        components: tuple[object, ...]
        phase13_manifest: dict[str, str]
        phase13_dir: Path

        @property
        def feature_names(self) -> tuple[str, ...]:
            return tuple(self.components[0].feature_set.feature_names)  # type: ignore[attr-defined]

    resolved = FakeResolved(
        frame.iloc[:8].assign(split="TRAIN"),
        frame.iloc[4:].assign(split="VALIDATION"),
        (SimpleNamespace(feature_set=spec),),
        {"run_id": "P13"},
        phase13_dir,
    )
    plan = build_analysis_plan(resolved, _settings())
    assert plan["validation_targets_accessed"] is False
    assert plan["constructed_feature_definition_count"] == 3
    definitions = plan["slice_definitions"]
    y = pd.Series([0, 1, 0, 1, 0, 0, 1, 0], index=resolved.validation_features.index)
    scores = pd.Series(np.linspace(0.01, 0.9, len(y)), index=y.index)
    overall = overall_metrics(y, scores, 0.5)
    slices, summary = evaluate_slices(
        definitions, resolved.validation_features, y, scores, 0.5, overall, _settings()
    )
    assert summary["total_slices"] > 0
    assert "target_independent_definition" in slices
    for definition in definitions[:5]:
        membership = membership_for_definition(
            definition, resolved.validation_features, scores=scores
        )
        assert isinstance(membership, pd.Series)
        assert len(membership) == len(y)
    temporal = temporal_metrics(
        resolved.validation_features, y, scores, 0.5, _settings(), overall=overall
    )
    assert not temporal.empty
    assert leakage_recheck(["safe_feature", "target__high_cost_claim_flag"])["valid"] is False
    assert leakage_recheck(["safe_feature"])["valid"] is True


def test_checkpoint_readiness_and_missing_artifact_validation(tmp_path: Path) -> None:
    payload = {"phase": 14, "run_id": "R1"}
    assert checkpoint_sha(payload)
    path = tmp_path / "checkpoint.json"
    written = write_checkpoint(path, payload)
    assert load_checkpoint(path, {"phase": 14}) == written
    assert load_checkpoint(path, {"phase": 13}) is None
    assert (
        readiness_gate({"prevalence": 0.2, "average_precision": 0.1, "roc_auc": 0.4}, [])["status"]
        == "BLOCKED"
    )
    missing = validate_existing_phase14(tmp_path)
    assert missing["valid"] is False
    assert missing["hardening_status"] == "BLOCKED"


def test_input_resolution_guards_and_single_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolve an explicit single-track parent without scoring or fitting anything."""

    spec = FeatureSetSpec(
        experiment_id="E1",
        feature_names=("f_num",),
        numeric_features=("f_num",),
        categorical_features=(),
        boolean_features=(),
        text_features=(),
        phase7_core_count=1,
        phase7_extended_count=0,
        phase8_lexical_count=0,
        phase8_text_count=0,
        feature_set_sha256="sha",
    )
    phase13 = tmp_path / "phase13"
    (phase13 / "calibrators").mkdir(parents=True)
    (phase13 / "models").mkdir()
    (phase13 / "models" / "model.cbm").write_bytes(b"serialized-model")
    (phase13 / "calibrators" / "t1.json").write_text('{"method":"NONE"}\n', encoding="utf-8")
    development = pd.DataFrame(
        {
            KEY: [1, 2],
            "split": ["TRAIN", "VALIDATION"],
            "claim__claim_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "f_num": [1.0, 2.0],
        }
    )
    fake_inputs = SimpleNamespace(
        phase10_inputs=SimpleNamespace(development=development),
        parents={"T1": SimpleNamespace(feature_set=spec)},
    )
    fake_lock = SimpleNamespace(
        phase12_inputs=fake_inputs,
        phase12_dir=tmp_path,
        effective_models={"T1": {"model_file": "models/model.cbm"}},
        train_targets=pd.DataFrame({KEY: [1], "target__high_cost_claim_flag": [0]}),
    )
    (phase13 / "phase13_manifest.json").write_text(
        '{"phase":13,"run_id":"P13","git_commit_sha":"abc","phase12_dir":".","test_target_rows_loaded":0,"test_predictions_created":0,"test_metrics_computed":false,"test_target_access_allowed":false,"first_allowed_test_target_phase":15}',
        encoding="utf-8",
    )
    (phase13 / "phase13_freeze.json").write_text(
        '{"outer_validation_accessed":false,"test_target_accessed":false}', encoding="utf-8"
    )
    (phase13 / "validation.json").write_text('{"valid":true}', encoding="utf-8")
    (phase13 / "validation_metrics.json").write_text(
        '{"phase13_development_champion":"C1"}', encoding="utf-8"
    )
    (phase13 / "effective_model_manifest.json").write_text(
        '{"models":[{"candidate_id":"C1","candidate_type":"SINGLE_TRACK","track":"T1","model_file":"models/model.cbm","score_space":"RAW_UNCALIBRATED_PROBABILITY","technical_threshold":0.5}]}',
        encoding="utf-8",
    )
    (phase13 / "threshold_policy.json").write_text('{"candidates":{}}', encoding="utf-8")
    (phase13 / "target_access_audit.json").write_text(
        '{"test_target_rows_loaded":0,"test_predictions_created":0,"test_metrics_computed":false,"test_target_access_allowed":false,"first_allowed_test_target_phase":15}',
        encoding="utf-8",
    )
    (phase13 / "phase12_parent_resolution.json").write_text("{}", encoding="utf-8")
    pd.DataFrame({KEY: [], "track": [], "effective_probability": []}).to_parquet(
        phase13 / "validation_predictions.parquet", index=False
    )
    monkeypatch.setattr(
        input_module,
        "validate_existing_phase13",
        lambda *args, **kwargs: {"valid": True, "hardening_status": "HARDENED_PASS"},
    )
    monkeypatch.setattr(input_module, "load_phase12_lock", lambda *args, **kwargs: fake_lock)
    resolved = input_module.resolve_phase13_parent(
        phase13, project_root=Path.cwd(), require_main_merge=False
    )
    assert resolved.champion_id == "C1"
    assert resolved.champion_type == "SINGLE_TRACK"
    assert resolved.feature_names == ("f_num",)
    assert input_module._test_seal(resolved.phase13_manifest, "manifest") == []
    assert input_module._read_json(phase13 / "phase13_manifest.json", "manifest")["phase"] == 13
    assert input_module.current_git_commit(Path.cwd())
    monkeypatch.setattr(
        input_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="yes\n"),
    )
    assert input_module.phase13_merged_to_main(Path.cwd(), "abc") is True


def test_runner_helpers_and_reports(tmp_path: Path) -> None:
    """Exercise deterministic runner helpers without invoking model inference."""

    assert runner_module.phase14_run_id().endswith("_PHASE14")
    assert (
        runner_module._json_safe(
            {"x": np.float64(1.0), "t": pd.Timestamp("2024-01-01"), "bad": float("inf")}
        )["bad"]
        is None
    )
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("x", encoding="utf-8")
    assert artifact.name in runner_module._artifact_hashes(tmp_path)
    resolved = SimpleNamespace(
        champion_type="SINGLE_TRACK",
        components=(
            SimpleNamespace(
                track="T1",
                model_sha256="model",
                calibrator_sha256="calibrator",
                feature_list_sha256="features",
            ),
        ),
        phase13_dir=tmp_path,
        phase13_manifest={"run_id": "P13"},
        threshold=0.5,
        score_space="RAW_UNCALIBRATED_PROBABILITY",
        phase13_manifest_sha256="m",
        phase13_validation_sha256="v",
        phase13_freeze_sha256="f",
        effective_manifest_sha256="e",
        champion_id="C1",
        ensemble_t1_weight=None,
    )
    pd.DataFrame(
        {
            KEY: [1, 2],
            "track": ["T1", "T1"],
            "candidate_id": ["C1", "C1"],
            "effective_probability": [0.1, 0.2],
        }
    ).to_parquet(tmp_path / "validation_predictions.parquet", index=False)
    actual = pd.DataFrame({KEY: [1, 2], "probability": [0.1, 0.2]})
    assert runner_module._reproduction(resolved, actual)["valid"] is True
    freeze = runner_module._phase14_freeze(
        resolved,
        {
            "slice_registry_sha256": "s",
            "slice_definition_sha256": "d",
            "temporal_definition_sha256": "t",
            "analysis_plan_sha256": "a",
            "bootstrap_policy": {},
            "drift_policy": {},
            "threshold_diagnostic_policy": {},
            "readiness_policy": {},
        },
        _settings(),
    )
    assert freeze["development_decisions_frozen"] is True
    warnings = runner_module._warning_inventory(
        {"positive_count": 1},
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        {"classification": "LOW_SHIFT"},
        pd.DataFrame({"error_type": []}),
    )
    assert "SMALL_VALIDATION_POSITIVE_COUNT" in warnings
    report = runner_module._reports(
        tmp_path / "reports",
        "R1",
        {
            "hardening_status": "READY",
            "phase15_readiness": {},
            "overall_metrics": {},
            "warnings": [],
        },
    )
    assert (report / "phase_14_summary.md").is_file()


def test_runner_checkpoint_binding_reuses_and_rejects_stale_results(tmp_path: Path) -> None:
    output = tmp_path / "stage.json"
    output.write_text('{"value": 1}\n', encoding="utf-8")
    checkpoint = tmp_path / "checkpoints" / "stage.json"
    bindings = {"phase": 14, "task": "stage", "analysis_plan_sha256": "plan"}
    runner_module._write_stage_checkpoint(checkpoint, [output], bindings)
    assert runner_module._checkpoint_reusable(checkpoint, [output], bindings) is True
    assert (
        runner_module._checkpoint_reusable(
            checkpoint, [output], {**bindings, "analysis_plan_sha256": "changed"}
        )
        is False
    )
    output.write_text('{"value": 2}\n', encoding="utf-8")
    assert runner_module._checkpoint_reusable(checkpoint, [output], bindings) is False


def test_independent_validator_replays_published_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    """The validator checks the immutable local bundle independently of runner status."""

    candidates = sorted(Path("artifacts/robustness_analysis").glob("202*"))
    artifact = candidates[-1].resolve() if candidates else Path("missing")
    if not artifact.is_dir() or any(
        not (artifact / name).is_file() for name in validation_module.REQUIRED_ARTIFACTS
    ):
        pytest.skip("local Phase 14 execution artifact is unavailable")
    fake = SimpleNamespace(
        phase13_manifest={"run_id": "20260817T_PHASE13_FINAL2"},
        phase13_manifest_sha256="e56bd0a38e880e942b6d7eed38f13b3aa98ff283727d2a84c846cc6d616a698a",
        phase13_validation_sha256="34be2ca671176f46befa0248d966228afbe2162035a1b8f615ec82df11da0cbf",
        phase13_freeze_sha256="01055aa87e8b2967f5ae38b20f934cc5c5d0e0d7668556dd1d9514df3b5ea62a",
        effective_manifest_sha256="4745b8526217772d26e070a0e4f1558ceeb13cc0f3bd59d8799bd3f4b8fd3886",
        champion_id="P10_T1_E1_OPTIMIZED",
        score_space="RAW_UNCALIBRATED_PROBABILITY",
        threshold=0.057,
    )
    monkeypatch.setattr(validation_module, "resolve_phase13_parent", lambda *args, **kwargs: fake)
    result = validation_module.validate_existing_phase14(artifact, project_root=Path.cwd())
    assert result["valid"] is True
    assert result["hardening_status"] == "HARDENED_PASS_WITH_WARNINGS"


def test_independent_validator_replays_fixture_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CI can validate a complete fixture without depending on a local run artifact."""

    from warranty_analytics_model.robustness_analysis.config import (
        PHASE14_VERSION,
        configuration_sha256,
    )

    artifact = tmp_path / "phase14"
    artifact.mkdir()
    phase13_parent = tmp_path / "phase13"
    phase13_parent.mkdir()
    accepted_scores = pd.DataFrame(
        {
            KEY: [1, 2],
            "track": ["T1", "T1"],
            "candidate_id": ["C1", "C1"],
            "effective_probability": [0.2, 0.8],
        }
    )
    accepted_scores.to_parquet(phase13_parent / "validation_predictions.parquet", index=False)
    validation_features = pd.DataFrame({KEY: [1, 2], "score": [0.2, 0.8]})
    validation_targets = pd.DataFrame({KEY: [1, 2], TARGET: [0, 1]})
    target_audit = {
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
    }
    fake = SimpleNamespace(
        phase13_manifest={"run_id": "P13"},
        phase13_manifest_sha256="m",
        phase13_validation_sha256="v",
        phase13_freeze_sha256="f",
        effective_manifest_sha256="e",
        champion_id="C1",
        champion_type="SINGLE_TRACK",
        score_space="RAW_UNCALIBRATED_PROBABILITY",
        threshold=0.5,
        phase13_dir=phase13_parent,
        components=(SimpleNamespace(track="T1"),),
        ensemble_t1_weight=None,
        validation_features=validation_features,
        load_validation_targets=lambda: (validation_targets, target_audit),
    )
    monkeypatch.setattr(validation_module, "resolve_phase13_parent", lambda *args, **kwargs: fake)
    monkeypatch.setattr(
        validation_module,
        "prepare_scorer",
        lambda *args, **kwargs: (
            lambda frame: frame[[KEY]].assign(probability=frame["score"].to_numpy())
        ),
    )

    plan = {
        "phase": 14,
        "feature_names": [],
        "risk_score_train_oof_rows": 0,
        "temporal_sort": [],
        "support_policy": {},
        "bootstrap_policy": {},
        "drift_policy": {},
        "threshold_diagnostic_policy": {},
        "readiness_policy": {},
        "constructed_feature_definition_count": 0,
        "validation_targets_accessed": False,
        "test_targets_accessed": False,
        "configuration_sha256": configuration_sha256(),
        "created_at_utc": "2024-01-01T00:00:00Z",
        "analysis_plan_sha256": "plan",
        "slice_registry": [],
        "slice_definitions": [],
        "slice_registry_sha256": validation_module._sha([]),
        "slice_definition_sha256": validation_module._sha([]),
    }
    plan["analysis_plan_sha256"] = validation_module._sha(
        {
            key: value
            for key, value in plan.items()
            if not key.endswith("_sha256") and key != "created_at_utc"
        }
    )
    plan["temporal_definition_sha256"] = validation_module._sha([])
    freeze_body = {
        "analysis_plan_sha256": plan["analysis_plan_sha256"],
        "validation_targets_accessed": False,
        "test_targets_accessed": False,
    }
    freeze = {
        **freeze_body,
        "phase14_analysis_freeze_sha256": validation_module._sha(freeze_body),
    }
    overall = overall_metrics(validation_targets[TARGET], validation_features["score"], 0.5)
    audit = {
        "test_target_rows_loaded": 0,
        "test_feature_rows_scored": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
    }
    invariant = prediction_invariance(
        validation_features,
        lambda frame: frame[[KEY]].assign(probability=frame["score"].to_numpy()),
        fresh_scorer=lambda frame: frame[[KEY]].assign(probability=frame["score"].to_numpy()),
    )
    readiness = readiness_gate(overall, [], test_audit=audit)
    manifest = {
        "phase": 14,
        "contract_version": PHASE14_VERSION,
        "configuration_sha256": configuration_sha256(),
        "phase13_dir": str(phase13_parent),
        "phase13_run_id": "P13",
        "phase13_manifest_sha256": "m",
        "phase13_validation_sha256": "v",
        "phase13_freeze_sha256": "f",
        "phase13_effective_model_manifest_sha256": "e",
        "phase13_development_champion": "C1",
        "frozen_score_space": "RAW_UNCALIBRATED_PROBABILITY",
        "frozen_threshold": 0.5,
        "warning_inventory": [],
        "git_commit_sha": "fixture-commit",
        "artifact_file_sha256": {},
    }
    json_files = {
        "phase13_parent_resolution.json": {},
        "analysis_plan.json": plan,
        "phase14_analysis_freeze.json": freeze,
        "prediction_reproduction.json": {
            "valid": True,
            "row_count": 2,
            "maximum_probability_delta": 0.0,
        },
        "prediction_invariance.json": invariant,
        "overall_metrics.json": overall,
        "temporal_summary.json": {},
        "slice_registry.json": {},
        "slice_definitions.json": {},
        "slice_summary.json": {},
        "feature_drift_summary.json": {},
        "score_distribution.json": {},
        "score_drift.json": {},
        "topk_lift.json": {},
        "error_profile_summary.json": {},
        "leakage_recheck.json": {"valid": True, "prohibited_feature_count": 0},
        "phase15_readiness.json": readiness,
        "target_access_audit.json": audit,
        "compute_manifest.json": {},
        "validation.json": {},
        "phase14_manifest.json": manifest,
        "overall_bootstrap_summary.json": {"replicate_count": 0},
        "material_slice_bootstrap_summary.json": {"eligible_slice_count": 0},
    }
    for name, payload in json_files.items():
        (artifact / name).write_text(json.dumps(payload), encoding="utf-8")
    manifest["artifact_file_sha256"] = {
        "overall_metrics.json": validation_module.sha256_file(artifact / "overall_metrics.json")
    }
    (artifact / "phase14_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in (
        "overall_bootstrap.parquet",
        "temporal_metrics.parquet",
        "slice_metrics.parquet",
        "feature_drift.parquet",
        "risk_decile_metrics.parquet",
        "threshold_sensitivity.parquet",
        "error_cohorts.parquet",
        "high_confidence_errors.parquet",
        "error_profile.parquet",
        "slice_memberships.parquet",
        "material_slice_bootstrap.parquet",
    ):
        pd.DataFrame().to_parquet(artifact / name, index=False)

    assert validation_module._independent_replay(SimpleNamespace(), artifact, {}, {}, {}) == []
    result = validation_module.validate_existing_phase14(artifact, project_root=tmp_path)
    assert result["valid"] is True
    assert result["hardening_status"] == "HARDENED_PASS"


def test_validator_replay_helpers_cover_scalar_and_ensemble_paths(tmp_path: Path) -> None:
    assert validation_module._close(None, None)
    assert not validation_module._close(None, 1)
    assert validation_module._close("same", "same")

    parent = tmp_path / "phase13"
    parent.mkdir()
    pd.DataFrame(
        {
            KEY: [1, 2, 1, 2],
            "track": ["T1", "T1", "T3", "T3"],
            "effective_probability": [0.2, 0.4, 0.1, 0.3],
        }
    ).to_parquet(parent / "validation_predictions.parquet", index=False)
    resolved = SimpleNamespace(
        phase13_dir=parent,
        champion_type="ENSEMBLE",
        ensemble_t1_weight=0.75,
    )
    result = validation_module._accepted_probabilities(resolved)
    assert np.allclose(result["expected_probability"].to_numpy(), [0.175, 0.375])


def test_independent_validator_replays_full_diagnostic_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the standalone validator's full scientific replay orchestration."""

    parent = tmp_path / "phase13"
    parent.mkdir()
    pd.DataFrame(
        {
            KEY: [1, 2],
            "track": ["T1", "T1"],
            "candidate_id": ["C1", "C1"],
            "effective_probability": [0.2, 0.8],
        }
    ).to_parquet(parent / "validation_predictions.parquet", index=False)
    features = pd.DataFrame({KEY: [1, 2], "score": [0.2, 0.8]})
    targets = pd.DataFrame({KEY: [1, 2], TARGET: [0, 1]})
    audit = {
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
    }

    def scorer(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[[KEY]].assign(probability=frame["score"].to_numpy())

    fake = SimpleNamespace(
        root=Path.cwd(),
        phase13_dir=parent,
        validation_features=features,
        train_features=features.copy(),
        components=(
            SimpleNamespace(
                track="T1",
                feature_set=SimpleNamespace(categorical_features=(), text_features=()),
            ),
        ),
        feature_names=("score",),
        champion_type="SINGLE_TRACK",
        champion_id="C1",
        threshold=0.5,
        score_space="RAW_UNCALIBRATED_PROBABILITY",
        load_validation_targets=lambda: (targets, audit),
    )
    persisted_overall = overall_metrics(targets[TARGET], features["score"], 0.5)
    persisted_invariance = {
        "valid": True,
        "serialization_reload_verified": True,
        "serialization_max_probability_delta": 0.0,
        "row_order_max_probability_delta": 0.0,
        "batch_max_probability_delta": 0.0,
    }
    plan = {
        "slice_definitions": [],
    }

    monkeypatch.setattr(validation_module, "prepare_scorer", lambda *args, **kwargs: scorer)
    monkeypatch.setattr(
        validation_module,
        "prediction_invariance",
        lambda *args, **kwargs: dict(persisted_invariance),
    )
    monkeypatch.setattr(
        validation_module,
        "load_robustness_settings",
        lambda *args, **kwargs: SimpleNamespace(
            overall_replicates=2,
            material_slice_replicates=2,
            preferred_bootstrap_workers=3,
            seed=7,
            confidence_level=0.95,
            min_slice_positives_bootstrap=1,
            min_slice_negatives_ranking=1,
            top_k=(0.5,),
            threshold_multipliers=(1.0,),
        ),
    )
    monkeypatch.setattr(
        validation_module,
        "temporal_metrics",
        lambda *args, **kwargs: pd.DataFrame({"stability_classification": ["STABLE"]}),
    )
    monkeypatch.setattr(
        validation_module,
        "evaluate_slices",
        lambda *args, **kwargs: (pd.DataFrame({"status": ["SUPPORTED"]}), {}),
    )
    monkeypatch.setattr(
        validation_module,
        "feature_drift",
        lambda *args, **kwargs: pd.DataFrame({"psi": [0.0]}),
    )
    monkeypatch.setattr(
        validation_module,
        "train_oof_scores",
        lambda *args, **kwargs: pd.DataFrame({KEY: [1, 2], "probability": [0.2, 0.8]}),
    )
    monkeypatch.setattr(
        validation_module,
        "score_drift",
        lambda *args, **kwargs: ({"mean": 0.5}, {"classification": "LOW_SHIFT"}),
    )
    monkeypatch.setattr(
        validation_module,
        "risk_decile_metrics",
        lambda *args, **kwargs: pd.DataFrame({"decile": ["D10"]}),
    )
    monkeypatch.setattr(validation_module, "topk_lift", lambda *args, **kwargs: {"top_k": 0.5})
    monkeypatch.setattr(
        validation_module,
        "threshold_sensitivity",
        lambda *args, **kwargs: pd.DataFrame({"threshold": [0.5]}),
    )
    empty_errors = pd.DataFrame({"error_type": pd.Series(dtype=str)})
    monkeypatch.setattr(validation_module, "error_cohorts", lambda *args, **kwargs: empty_errors)
    monkeypatch.setattr(
        validation_module, "high_confidence_errors", lambda *args, **kwargs: empty_errors
    )
    monkeypatch.setattr(
        validation_module, "build_error_context", lambda *args, **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(validation_module, "error_profile", lambda *args, **kwargs: pd.DataFrame())
    replay_workers: list[int] = []

    def replay_bootstrap(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], list[Any]]:
        replay_workers.append(int(kwargs["workers"]))
        return {"replicate_count": 2}, []

    def replay_material_bootstrap(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], list[Any]]:
        replay_workers.append(int(kwargs["workers"]))
        return {"eligible_slice_count": 0}, []

    monkeypatch.setattr(validation_module, "stratified_bootstrap", replay_bootstrap)
    monkeypatch.setattr(validation_module, "material_slice_bootstrap", replay_material_bootstrap)
    monkeypatch.setattr(
        validation_module,
        "leakage_recheck",
        lambda *args, **kwargs: {"valid": True, "prohibited_feature_count": 0},
    )
    monkeypatch.setattr(validation_module, "_warning_inventory", lambda *args, **kwargs: [])
    monkeypatch.setattr(validation_module, "_compare_frames", lambda *args, **kwargs: [])
    monkeypatch.setattr(validation_module, "_compare_nested", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        validation_module, "readiness_gate", lambda *args, **kwargs: {"status": "READY"}
    )

    for name in (
        "temporal_metrics.parquet",
        "slice_metrics.parquet",
        "slice_memberships.parquet",
        "feature_drift.parquet",
        "risk_decile_metrics.parquet",
        "threshold_sensitivity.parquet",
        "error_cohorts.parquet",
        "high_confidence_errors.parquet",
        "error_profile.parquet",
        "overall_bootstrap.parquet",
        "material_slice_bootstrap.parquet",
    ):
        pd.DataFrame().to_parquet(tmp_path / name, index=False)
    for name, payload in {
        "score_distribution.json": {},
        "score_drift.json": {},
        "topk_lift.json": {},
        "overall_bootstrap_summary.json": {},
        "material_slice_bootstrap_summary.json": {},
        "phase14_manifest.json": {"warning_inventory": []},
        "leakage_recheck.json": {"valid": True, "prohibited_feature_count": 0},
        "phase15_readiness.json": {"status": "READY"},
    }.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    errors = validation_module._independent_replay(
        fake,
        tmp_path,
        {"row_count": 2, "maximum_probability_delta": 0.0},
        persisted_overall,
        persisted_invariance,
        plan,
    )
    assert errors == []
    assert replay_workers == [3, 3]


def test_validator_comparison_membership_and_warning_helpers(tmp_path: Path) -> None:
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        validation_module._read(invalid_json)
    assert validation_module._sha({"b": 2, "a": 1}) == validation_module._sha({"a": 1, "b": 2})

    assert validation_module._compare_nested({"a": 1}, {"a": 1}) == []
    assert validation_module._compare_nested({"a": 1}, {"b": 1})
    assert validation_module._compare_nested({"a": 1}, [])
    assert validation_module._compare_nested([1], [1, 2])
    assert validation_module._compare_nested([1], [2])
    assert (
        validation_module._compare_frames(
            pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]}), "equal"
        )
        == []
    )
    assert validation_module._compare_frames(
        pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}), "columns"
    )
    assert validation_module._compare_frames(
        pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]}), "values"
    )

    frame = _frame().iloc[:3].copy()
    memberships = validation_module._slice_membership_frame(
        [
            {
                "slice_id": "cat",
                "kind": "categorical",
                "column": "f_cat",
                "categories": ["A"],
            }
        ],
        frame,
        pd.Series([0.1, 0.2, 0.3]),
    )
    assert set(memberships["slice_label"]) == {"A", "__OTHER__"}

    warnings = validation_module._warning_inventory(
        {"positive_count": 1},
        pd.DataFrame({"stability_classification": ["MODERATE_DEGRADATION", "SEVERE_DEGRADATION"]}),
        pd.DataFrame(
            {
                "status": ["LOW_SUPPORT"],
                "stability_classification": ["SEVERE_DEGRADATION"],
            }
        ),
        pd.DataFrame(
            {
                "psi": [0.3],
                "missingness_classification": ["MATERIAL_MISSINGNESS_SHIFT"],
            }
        ),
        {"classification": "SCORE_DISTRIBUTION_SHIFT"},
        pd.DataFrame({"error_type": ["FALSE_NEGATIVE"]}),
    )
    assert {
        "SYNTHETIC_POC",
        "BUSINESS_TARGET_UNCONFIRMED",
        "SMALL_VALIDATION_POSITIVE_COUNT",
        "TEMPORAL_DEGRADATION",
        "SEVERE_TEMPORAL_DEGRADATION",
        "LOW_SUPPORT_SLICE",
        "SEVERE_SLICE_PERFORMANCE_DEGRADATION",
        "HIGH_FEATURE_DRIFT",
        "MATERIAL_MISSINGNESS_SHIFT",
        "SCORE_DISTRIBUTION_SHIFT",
        "HIGH_FALSE_NEGATIVE_CONCENTRATION",
    }.issubset(warnings)


def test_runner_branches_plan_check_oof_and_aggregate_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover ensemble, warning, plan-check, OOF, and aggregate-report paths."""

    accepted = pd.DataFrame(
        {
            KEY: [1, 2, 1, 2],
            "track": ["T1", "T1", "T3", "T3"],
            "candidate_id": ["C1"] * 4,
            "effective_probability": [0.2, 0.4, 0.1, 0.3],
        }
    )
    accepted.to_parquet(tmp_path / "validation_predictions.parquet", index=False)
    ensemble = SimpleNamespace(
        champion_type="ENSEMBLE",
        components=(),
        phase13_dir=tmp_path,
        phase13_manifest={"run_id": "P13"},
        ensemble_t1_weight=0.75,
    )
    actual = pd.DataFrame({KEY: [1, 2], "probability": [0.175, 0.375]})
    assert runner_module._reproduction(ensemble, actual)["valid"] is True
    with pytest.raises(ValueError):
        runner_module._reproduction(
            SimpleNamespace(**{**ensemble.__dict__, "ensemble_t1_weight": 0.5}),
            pd.DataFrame({KEY: [1, 1], "probability": [0.1, 0.2]}),
        )
    audit = runner_module._population_audit(
        SimpleNamespace(train_targets=pd.DataFrame({KEY: [1, 2]})), {"target_rows_loaded": 2}, 2
    )
    assert audit["test_target_rows_loaded"] == 0
    temporal = pd.DataFrame(
        {"stability_classification": ["MODERATE_DEGRADATION", "SEVERE_DEGRADATION"]}
    )
    slices = pd.DataFrame(
        {"status": ["LOW_SUPPORT"], "stability_classification": ["SEVERE_DEGRADATION"]}
    )
    drift = pd.DataFrame({"psi": [0.3], "missingness_classification": ["HIGH_MISSINGNESS_SHIFT"]})
    errors = pd.DataFrame({"error_type": ["FALSE_NEGATIVE"] * 4})
    warnings = runner_module._warning_inventory(
        {"positive_count": 1},
        temporal,
        slices,
        drift,
        {"classification": "SCORE_DISTRIBUTION_SHIFT"},
        errors,
    )
    assert {
        "TEMPORAL_DEGRADATION",
        "SEVERE_TEMPORAL_DEGRADATION",
        "HIGH_FEATURE_DRIFT",
        "SCORE_DISTRIBUTION_SHIFT",
    }.issubset(warnings)

    oof_resolved = SimpleNamespace(
        phase13_dir=tmp_path,
        champion_type="SINGLE_TRACK",
        components=(SimpleNamespace(track="T1"),),
        phase13_manifest={"phase12_run_id": "P12"},
    )
    pd.DataFrame(
        {
            KEY: [1, 2],
            "track": ["T1", "T1"],
            "raw_probability": [0.1, 0.2],
            "calibrated_probability": [0.11, 0.22],
        }
    ).to_parquet(tmp_path / "selected_calibrated_oof_predictions.parquet", index=False)
    assert len(input_module.train_oof_scores(oof_resolved)) == 2
    pd.DataFrame({KEY: [1, 2], "effective_probability": [0.3, 0.4]}).to_parquet(
        tmp_path / "selected_calibrated_oof_predictions.parquet", index=False
    )
    assert input_module.train_oof_scores(oof_resolved)["probability"].tolist() == [0.3, 0.4]

    plan_resolved = SimpleNamespace(phase13_manifest={"run_id": "P13"}, champion_id="C1")
    monkeypatch.setattr(runner_module, "load_robustness_settings", lambda root: _settings())
    monkeypatch.setattr(runner_module, "phase14_contract_check", lambda root: {"valid": True})
    monkeypatch.setattr(
        runner_module, "resolve_phase13_parent", lambda *args, **kwargs: plan_resolved
    )
    monkeypatch.setattr(
        runner_module, "build_analysis_plan", lambda resolved, settings: {"ok": True}
    )
    assert runner_module.phase14_plan_check(tmp_path)["valid"] is True
    monkeypatch.setattr(
        runner_module,
        "resolve_phase13_parent",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad")),
    )
    assert runner_module.phase14_plan_check(tmp_path)["valid"] is False

    report = write_phase14_reports(
        tmp_path / "aggregate", "R2", {"phase15_readiness": {"status": "READY"}}
    )
    assert (report / "phase_14_summary.json").is_file()


def test_runner_builds_a_minimal_frozen_bundle_with_mocked_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the orchestration path with every scientific primitive isolated."""

    frame = _frame().iloc[:4].copy()
    frame["split"] = "VALIDATION"
    targets = pd.DataFrame({KEY: frame[KEY], "target__high_cost_claim_flag": [0, 1, 0, 1]})
    component = SimpleNamespace(
        track="T1",
        model_sha256="model",
        calibrator_sha256="calibrator",
        feature_list_sha256="features",
        feature_set=SimpleNamespace(
            feature_names=("f_num",), categorical_features=(), text_features=()
        ),
    )
    fake = SimpleNamespace(
        root=Path.cwd(),
        phase13_dir=tmp_path / "phase13",
        phase13_manifest={"run_id": "P13", "git_commit_sha": "abc"},
        phase13_freeze={},
        phase13_validation={},
        phase13_metrics={},
        effective_manifest={},
        threshold_policy={},
        phase13_audit={},
        parent_resolution={},
        phase13_manifest_sha256="m",
        phase13_validation_sha256="v",
        phase13_freeze_sha256="f",
        effective_manifest_sha256="e",
        champion_id="C1",
        champion_type="SINGLE_TRACK",
        score_space="RAW_UNCALIBRATED_PROBABILITY",
        threshold=0.5,
        components=(component,),
        ensemble_t1_weight=None,
        feature_names=("f_num",),
        train_features=frame.assign(split="TRAIN"),
        validation_features=frame,
        train_targets=targets,
        load_validation_targets=lambda: (targets.copy(), {"test_target_rows_loaded": 0}),
    )
    plan = {
        "phase13_run_id": "P13",
        "slice_registry": [],
        "slice_definitions": [],
        "slice_registry_sha256": runner_module._canonical_sha([]),
        "slice_definition_sha256": runner_module._canonical_sha([]),
        "temporal_definition_sha256": runner_module._canonical_sha([]),
        "analysis_plan_sha256": "plan",
        "bootstrap_policy": {},
        "drift_policy": {},
        "threshold_diagnostic_policy": {},
        "readiness_policy": {},
    }
    monkeypatch.setattr(runner_module, "resolve_phase13_parent", lambda *args, **kwargs: fake)
    monkeypatch.setattr(runner_module, "build_analysis_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        runner_module,
        "prepare_scorer",
        lambda *args, **kwargs: lambda data: data[[KEY]].assign(probability=0.2),
    )
    monkeypatch.setattr(
        runner_module,
        "_reproduction",
        lambda *args, **kwargs: {"valid": True, "maximum_probability_delta": 0.0},
    )
    monkeypatch.setattr(
        runner_module,
        "overall_metrics",
        lambda *args, **kwargs: {
            "row_count": 4,
            "positive_count": 2,
            "negative_count": 2,
            "prevalence": 0.5,
            "average_precision": 0.75,
            "roc_auc": 0.75,
            "primary_signal_pass": True,
        },
    )
    monkeypatch.setattr(
        runner_module,
        "stratified_bootstrap",
        lambda *args, **kwargs: ({"replicate_count": 2000}, [{"replicate": 0}]),
    )
    monkeypatch.setattr(runner_module, "temporal_metrics", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        runner_module,
        "evaluate_slices",
        lambda *args, **kwargs: (
            pd.DataFrame(),
            {
                "total_slices": 0,
                "supported_slices": 0,
                "low_support_slices": 0,
                "severe_degradation_warnings": 0,
            },
        ),
    )
    monkeypatch.setattr(runner_module, "feature_drift", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        runner_module,
        "train_oof_scores",
        lambda *args, **kwargs: pd.DataFrame({KEY: [1, 2], "probability": [0.1, 0.2]}),
    )
    monkeypatch.setattr(
        runner_module,
        "score_drift",
        lambda *args, **kwargs: ({"train_p50": 0.1}, {"classification": "LOW_SHIFT"}),
    )
    monkeypatch.setattr(
        runner_module, "risk_decile_metrics", lambda *args, **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(runner_module, "topk_lift", lambda *args, **kwargs: {"rows": []})
    monkeypatch.setattr(
        runner_module, "threshold_sensitivity", lambda *args, **kwargs: pd.DataFrame()
    )
    empty_errors = pd.DataFrame(
        {"error_type": [], KEY: [], "probability": [], "threshold_margin": []}
    )
    monkeypatch.setattr(runner_module, "error_cohorts", lambda *args, **kwargs: empty_errors.copy())
    monkeypatch.setattr(
        runner_module, "high_confidence_errors", lambda *args, **kwargs: empty_errors.copy()
    )
    monkeypatch.setattr(runner_module, "error_profile", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        runner_module,
        "prediction_invariance",
        lambda *args, **kwargs: {
            "valid": True,
            "serialization_max_probability_delta": 0.0,
            "row_order_max_probability_delta": 0.0,
            "batch_max_probability_delta": 0.0,
        },
    )
    monkeypatch.setattr(
        runner_module,
        "leakage_recheck",
        lambda *args, **kwargs: {"valid": True, "prohibited_feature_count": 0},
    )
    monkeypatch.setattr(
        runner_module,
        "readiness_gate",
        lambda *args, **kwargs: {
            "status": "READY",
            "safe_to_start_phase15": True,
            "hard_blockers": [],
            "warnings": [],
            "development_decisions_frozen": True,
            "model_changes_after_phase14_analysis": "prohibited",
        },
    )
    monkeypatch.setattr(runner_module, "current_git_commit", lambda *args, **kwargs: "abc")
    monkeypatch.setattr(runner_module, "_artifact_hashes", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        validation_module,
        "validate_existing_phase14",
        lambda *args, **kwargs: {
            "valid": True,
            "errors": [],
            "warnings": [],
            "hardening_status": "HARDENED_PASS",
        },
    )
    result = runner_module.build_phase14(
        fake.phase13_dir,
        project_root=Path.cwd(),
        output_dir=tmp_path / "output",
        report_dir=tmp_path / "reports",
        run_id="R-MOCK",
        max_workers=1,
        catboost_inference_threads=1,
    )
    assert result["run_id"] == "R-MOCK"
    assert Path(result["run_directory"]).is_dir()
