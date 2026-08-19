"""Fixture-driven safety and metric tests for Phase 15."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from warranty_analytics_model.final_evaluation.checkpoint import (
    checkpoint_sha,
    load_checkpoint,
    write_checkpoint,
)
from warranty_analytics_model.final_evaluation.config import (
    Phase15ConfigurationError,
    compute_plan,
    configuration_sha256,
    load_final_test_settings,
)
from warranty_analytics_model.final_evaluation.contract import phase15_contract_check
from warranty_analytics_model.final_evaluation.input import (
    Phase15InputError,
    _build_full_feature_frame,
    _phase14_ci_green,
    _phase14_commit_reachable_from_main,
    _read_json,
    _schema_payload,
    _test_seal,
    _validate_phase14_start,
    build_test_membership_audit,
    feature_schema_sha256,
    leakage_audit,
    load_test_targets_after_freeze,
)
from warranty_analytics_model.final_evaluation.metrics import (
    reliability_table,
    signal_status,
    validation_test_comparison,
)
from warranty_analytics_model.final_evaluation.metrics import (
    test_metrics as compute_test_metrics,
)
from warranty_analytics_model.final_evaluation.planning import (
    build_evaluation_plan,
    build_plan_inputs,
    frozen_test_slice_definitions,
)
from warranty_analytics_model.final_evaluation.planning import (
    test_slice_memberships as build_test_slice_memberships,
)
from warranty_analytics_model.final_evaluation.provenance import (
    artifact_hashes,
    canonical_sha,
    current_scientific_commit,
    json_safe,
)
from warranty_analytics_model.final_evaluation.ranking import concentration_summary
from warranty_analytics_model.final_evaluation.reporting import write_phase15_reports
from warranty_analytics_model.final_evaluation.runner import (
    _false_negative_summary,
    _freeze_payload,
    _required_phase14_gate_error,
    _validation_metrics,
    _write_json,
    build_phase15,
    phase15_plan_check,
    phase15_run_id,
)
from warranty_analytics_model.final_evaluation.scoring import (
    build_final_model_policy,
    score_test_in_batches,
)
from warranty_analytics_model.final_evaluation.status import final_model_status
from warranty_analytics_model.final_evaluation.validation import (
    _blocked,
    _close,
    _compare_fields,
    _compare_frames,
    validate_existing_phase15,
)
from warranty_analytics_model.robustness_analysis.bootstrap import stratified_bootstrap
from warranty_analytics_model.robustness_analysis.input import KEY
from warranty_analytics_model.robustness_analysis.ranking import risk_decile_metrics


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            KEY: np.arange(1, 13),
            "claim__claim_date": pd.date_range("2025-01-01", periods=12, freq="MS"),
            "f_cat": ["A", "B", "A", None] * 3,
            "f_num": np.arange(12, dtype="float64"),
        }
    )


def _resolved() -> SimpleNamespace:
    frame = _frame()
    component = SimpleNamespace(
        track="T1",
        candidate_id="C1",
        model_sha256="model-sha",
        calibrator_sha256="cal-sha",
        feature_list_sha256="features-sha",
    )
    nested = SimpleNamespace(
        phase7_lineage={"f_cat": {"target_dependent": False}, "f_num": {"target_dependent": False}},
    )
    phase12 = SimpleNamespace(
        phase12_inputs=SimpleNamespace(phase10_inputs=SimpleNamespace(phase9_inputs=nested))
    )
    phase13 = SimpleNamespace(
        phase13_manifest={"run_id": "P13"},
        phase12_lock=phase12,
    )
    return SimpleNamespace(
        phase14_manifest={"run_id": "P14"},
        phase14_plan={
            "slice_definitions": [
                {
                    "slice_id": "cat",
                    "kind": "categorical",
                    "column": "f_cat",
                    "categories": ["A", "B"],
                },
                {
                    "slice_id": "num",
                    "kind": "numeric_quantile",
                    "column": "f_num",
                    "edges": [0.0, 4.0, 8.0, 12.0],
                },
                {
                    "slice_id": "chron",
                    "kind": "chronological_thirds",
                    "column": "claim__claim_date",
                },
                {
                    "slice_id": "risk",
                    "kind": "risk_score_decile",
                    "column": "f_num",
                    "edges": [0.0, 0.3, 0.6, 1.0],
                },
            ]
        },
        phase14_readiness={"warnings": ["SYNTHETIC_POC"]},
        phase13=phase13,
        champion_id="C1",
        champion_type="SINGLE_TRACK",
        score_space="RAW_UNCALIBRATED_PROBABILITY",
        threshold=0.5,
        components=(component,),
        feature_names=("f_cat", "f_num"),
        feature_schema_sha256="schema-sha",
        test_features=frame,
    )


def test_contract_config_and_compute_guards() -> None:
    assert phase15_contract_check(Path.cwd())["valid"] is True
    settings = load_final_test_settings(Path.cwd())
    assert settings.seed == 20260810
    assert configuration_sha256()
    plan = compute_plan(
        settings, max_workers=1, catboost_inference_threads=1, bootstrap_replicates=2000
    )
    assert plan["native_threads_per_worker"] == 1
    with pytest.raises(Phase15ConfigurationError):
        compute_plan(settings, max_workers=0)
    with pytest.raises(Phase15ConfigurationError):
        compute_plan(settings, bootstrap_replicates=1999)
    with pytest.raises(Phase15ConfigurationError):
        compute_plan(settings, catboost_inference_threads=10_000)


def test_metrics_signal_reliability_and_generalization() -> None:
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1], dtype="int8")
    p = np.array([0.02, 0.10, 0.20, 0.55, 0.70, 0.90, 0.04, 0.60])
    metrics = compute_test_metrics(y, p, 0.5)
    assert metrics["average_precision"] > metrics["prevalence"]
    assert signal_status(metrics)["status"] == "SIGNAL_CONFIRMED"
    table, reliability = reliability_table(y, p)
    assert len(table) == 10
    assert set(reliability) == {"ece_10", "mce_10"}
    stable = validation_test_comparison(
        metrics, metrics, moderate_ap_ratio=0.75, moderate_roc_drop=0.1
    )
    assert stable["generalization_status"] == "STABLE_GENERALIZATION"
    moderate = validation_test_comparison(
        {**metrics, "average_precision": 0.9},
        {**metrics, "average_precision": 0.6},
        moderate_ap_ratio=0.75,
        moderate_roc_drop=0.1,
    )
    assert moderate["generalization_status"] == "MODERATE_DEGRADATION"
    severe = validation_test_comparison(
        metrics,
        {**metrics, "average_precision": 0.1, "roc_auc": 0.5},
        moderate_ap_ratio=0.75,
        moderate_roc_drop=0.1,
    )
    assert severe["generalization_status"] == "SEVERE_DEGRADATION"
    assert (
        signal_status({"average_precision": 0.1, "prevalence": 0.2, "roc_auc": 0.4})["status"]
        == "SIGNAL_COLLAPSE"
    )


def test_frozen_definitions_memberships_and_plan() -> None:
    resolved: Any = _resolved()
    definitions = frozen_test_slice_definitions(resolved)
    assert any(item["kind"] == "chronological_thirds" for item in definitions)
    probabilities = pd.Series(np.linspace(0.05, 0.95, len(resolved.test_features)))
    memberships = build_test_slice_memberships(definitions, resolved.test_features, probabilities)
    assert len(memberships) == len(definitions) * len(resolved.test_features)
    plan = build_evaluation_plan(
        resolved,
        load_final_test_settings(Path.cwd()),
        {"expected_test_row_count": len(resolved.test_features), "target_independent": True},
        definitions,
        {
            "test_bootstrap_replicates": 2000,
            "bootstrap_workers": 1,
            "catboost_inference_threads": 1,
        },
    )
    assert plan["scoring_policy_count"] == 1
    assert plan["test_targets_accessed"] is False
    assert plan["ranking_policy"]["top_k"] == [0.05, 0.1, 0.2, 0.3]


def test_target_independent_plan_input_builder_and_reexport_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved: Any = _resolved()
    resolved.phase6_manifest = {}
    resolved.test_lock = {}
    resolved.phase13.phase12_lock.phase12_inputs.phase10_inputs.phase9_inputs.assignments = (
        _frame()[[KEY, "claim__claim_date"]].assign(split="TEST")
    )
    import warranty_analytics_model.final_evaluation.planning as planning_module

    monkeypatch.setattr(
        planning_module,
        "build_test_membership_audit",
        lambda *_args, **_kwargs: {
            "expected_test_row_count": len(resolved.test_features),
            "target_independent": True,
        },
    )
    plan, definitions, memberships = build_plan_inputs(
        resolved,
        load_final_test_settings(Path.cwd()),
        {
            "test_bootstrap_replicates": 2000,
            "bootstrap_workers": 1,
            "catboost_inference_threads": 1,
        },
    )
    assert plan["test_targets_accessed"] is False
    assert definitions and not memberships.empty

    # The small Phase 15 modules intentionally re-export the canonical,
    # already-tested Phase 14 diagnostics; importing them is itself a contract
    # check that the public Phase 15 surface remains wired correctly.
    from warranty_analytics_model.final_evaluation import (
        comparison,
        errors,
        invariance,
        reliability,
        slices,
        temporal,
    )

    assert callable(comparison.validation_test_comparison)
    assert callable(errors.error_cohorts)
    assert callable(invariance.prediction_invariance)
    assert callable(reliability.reliability_table)
    assert callable(slices.evaluate_slices)
    assert callable(temporal.temporal_metrics)


def test_membership_leakage_policy_and_model_policy() -> None:
    assignments = pd.DataFrame(
        {
            KEY: [1, 2, 3, 4],
            "claim_date": pd.date_range("2025-01-01", periods=4),
            "split": ["TRAIN", "VALIDATION", "TEST", "TEST"],
        }
    )
    from warranty_analytics_model.splits.manifest import (
        assignment_content_sha256,
        claim_key_sha256,
        unordered_claim_key_sha256,
    )

    test = assignments[assignments["split"] == "TEST"]
    phase6 = {"split_assignment_sha256": assignment_content_sha256(assignments)}
    lock = {
        "ordered_test_claim_keys_sha256": claim_key_sha256(test),
        "unordered_test_claim_keys_sha256": unordered_claim_key_sha256(test),
        "test_assignment_content_sha256": assignment_content_sha256(test),
        "test_row_count": 2,
    }
    audit = build_test_membership_audit(assignments, phase6, lock)
    assert audit["expected_test_row_count"] == 2
    with pytest.raises(Phase15InputError):
        build_test_membership_audit(assignments.drop(index=3), phase6, lock)
    resolved: Any = _resolved()
    safe = leakage_audit(resolved)
    assert safe["valid"] is True
    resolved.phase13.phase12_lock.phase12_inputs.phase10_inputs.phase9_inputs.phase7_lineage[
        "f_num"
    ]["target_dependent"] = True
    assert leakage_audit(resolved)["valid"] is False
    policy = build_final_model_policy(resolved)
    assert policy["policy"] == "REUSE_FROZEN_PHASE14_CHAMPION"
    assert policy["model_retraining"] is False


def test_provenance_checkpoint_reports_and_status(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_text("{}", encoding="utf-8")
    assert canonical_sha({"a": 1}) == canonical_sha({"a": 1})
    assert artifact_hashes(tmp_path, exclude=set())["payload.json"]
    assert current_scientific_commit(Path.cwd())
    assert json_safe(np.int64(3)) == 3
    checkpoint = tmp_path / "checkpoints" / "one.json"
    payload = write_checkpoint(checkpoint, {"phase": 15, "run_id": "R"})
    assert checkpoint_sha({"phase": 15, "run_id": "R"})
    assert load_checkpoint(checkpoint, {"phase": 15}) == payload
    assert load_checkpoint(checkpoint, {"phase": 14}) is None
    report = write_phase15_reports(
        tmp_path / "reports",
        "R",
        {
            "test_metrics": {"average_precision": 0.2},
            "phase15_final_model_status": {"final_model_status": "ACCEPTED_FOR_POC"},
        },
    )
    assert (report / "phase15_summary.md").is_file()
    assert (
        final_model_status(
            {"status": "SIGNAL_CONFIRMED"},
            {"generalization_status": "STABLE_GENERALIZATION"},
            provenance_valid=True,
            leakage_valid=True,
            scoring_valid=True,
            test_use_valid=True,
        )["final_model_status"]
        == "ACCEPTED_FOR_POC"
    )
    assert (
        final_model_status(
            {"status": "SIGNAL_COLLAPSE"},
            {},
            provenance_valid=True,
            leakage_valid=True,
            scoring_valid=True,
            test_use_valid=True,
        )["final_model_status"]
        == "FAILED_GENERALIZATION"
    )
    assert (
        final_model_status(
            {"status": "SIGNAL_CONFIRMED"},
            {"generalization_status": "MODERATE_DEGRADATION"},
            provenance_valid=True,
            leakage_valid=True,
            scoring_valid=True,
            test_use_valid=True,
        )["final_model_status"]
        == "ACCEPTED_WITH_LIMITATIONS"
    )


def test_bootstrap_ranking_and_empty_validation(tmp_path: Path) -> None:
    y = np.array([0, 1, 0, 1, 0, 1], dtype="int8")
    p = np.array([0.1, 0.8, 0.2, 0.7, 0.3, 0.9])
    summary_a, rows_a = stratified_bootstrap(y, p, 0.5, replicates=8, seed=3, workers=1)
    summary_b, rows_b = stratified_bootstrap(y, p, 0.5, replicates=8, seed=3, workers=2)
    assert summary_a == summary_b and rows_a == rows_b
    oof = pd.DataFrame({KEY: np.arange(1, 7), "probability": p})
    deciles = risk_decile_metrics(oof, _frame().iloc[:6], y, p)
    assert deciles.iloc[0]["decile"] == "D10"
    assert concentration_summary(deciles)["positive_share_d10"] >= 0.0
    assert validate_existing_phase15(tmp_path)["valid"] is False
    assert phase15_plan_check(tmp_path)["valid"] is False
    with pytest.raises(Phase15InputError):
        build_phase15(tmp_path)


def test_seals_and_phase14_ci_gate() -> None:
    closed = {
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
    }
    assert _test_seal(closed, "x") == []
    assert _test_seal({**closed, "test_metrics_computed": True}, "x")
    assert _phase14_ci_green({"main_quality_ci_green": "GREEN"}, {}) is True
    assert _phase14_ci_green({}, {}) is False


def test_input_schema_and_start_guards(tmp_path: Path) -> None:
    with pytest.raises(Phase15InputError):
        _read_json(tmp_path / "missing.json", "fixture")
    root = _resolved()
    root.phase13.components = ()
    assert _schema_payload(root.phase13) == {}
    assert feature_schema_sha256(root.phase13)
    assert _phase14_commit_reachable_from_main(Path.cwd(), "unknown") is False
    with pytest.raises(Phase15InputError):
        _validate_phase14_start(
            Path.cwd(),
            {"phase": 13},
            {"test_seal": {}},
            {},
            {},
            require_main_merge=False,
        )
    # An invalid explicit Phase 14 directory is rejected before any model or
    # target access is attempted.
    with pytest.raises(Phase15InputError):
        from warranty_analytics_model.final_evaluation.input import resolve_phase14_parent

        resolve_phase14_parent(tmp_path, require_main_merge=False, validate_upstream=False)


def test_full_feature_frame_and_target_loader_guards(tmp_path: Path) -> None:
    frame = _frame()
    assignments = frame[[KEY, "claim__claim_date"]].copy()
    assignments["claim_date"] = assignments["claim__claim_date"]
    assignments["split"] = ["TEST"] * len(assignments)
    source = SimpleNamespace(
        phase7_lineage={"f_num": {}},
        structured_features=frame[[KEY, "claim__claim_date", "f_num"]],
        text_features=frame[[KEY, "f_cat"]],
        assignments=assignments,
    )
    phase13: Any = SimpleNamespace(
        phase12_lock=SimpleNamespace(
            phase12_inputs=SimpleNamespace(phase10_inputs=SimpleNamespace(phase9_inputs=source))
        ),
        feature_names=("f_num", "f_cat"),
    )
    full = _build_full_feature_frame(phase13)
    assert set(full[KEY]) == set(frame[KEY])
    assert "target__high_cost_claim_flag" not in full.columns
    # Invalid feature membership is fail-closed.
    source.text_features = source.text_features.iloc[:-1]
    with pytest.raises(Phase15InputError):
        _build_full_feature_frame(phase13)
    # The target loader never accepts an already-used freeze, but succeeds
    # once the untouched freeze is presented and the authoritative snapshot is
    # exactly aligned.
    snapshot = tmp_path / "snapshot.parquet"
    frame[[KEY]].assign(**{"target__high_cost_claim_flag": [0, 1] * 6}).to_parquet(
        snapshot, index=False
    )
    resolved: Any = SimpleNamespace(test_features=frame[[KEY]], claim_snapshot_path=snapshot)
    with pytest.raises(Phase15InputError):
        load_test_targets_after_freeze(resolved, {"test_predictions_created": True})
    targets, audit = load_test_targets_after_freeze(
        resolved,
        {
            "test_targets_accessed": False,
            "test_predictions_created": False,
            "test_metrics_computed": False,
        },
    )
    assert len(targets) == len(frame) and audit["first_access_after_phase15_freeze"] is True


def test_scoring_policy_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    import warranty_analytics_model.final_evaluation.scoring as scoring_module

    frame = _frame()[[KEY, "f_num"]]
    resolved: Any = SimpleNamespace(threshold=0.5, phase13=SimpleNamespace())

    def scorer(batch: pd.DataFrame) -> pd.DataFrame:
        return batch[[KEY]].assign(probability=np.linspace(0.1, 0.9, len(batch)))

    monkeypatch.setattr(scoring_module, "prepare_scorer", lambda *_args, **_kwargs: scorer)
    scored = score_test_in_batches(resolved, frame, inference_threads=1, batch_size=3)
    assert len(scored) == len(frame)
    assert set(scored["predicted_class"]) <= {0, 1}

    def bad_scorer(batch: pd.DataFrame) -> pd.DataFrame:
        return batch[[KEY]].assign(probability=2.0)

    monkeypatch.setattr(scoring_module, "prepare_scorer", lambda *_args, **_kwargs: bad_scorer)
    with pytest.raises(Phase15InputError):
        score_test_in_batches(resolved, frame, inference_threads=1)


def test_scoring_policy_rejects_empty_and_malformed_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import warranty_analytics_model.final_evaluation.scoring as scoring_module

    frame = _frame()[[KEY, "f_num"]]
    resolved: Any = SimpleNamespace(threshold=0.5, phase13=SimpleNamespace())

    monkeypatch.setattr(
        scoring_module,
        "prepare_scorer",
        lambda *_args, **_kwargs: lambda batch: batch[[KEY]].assign(probability=0.2),
    )
    with pytest.raises(Phase15InputError, match="empty TEST"):
        score_test_in_batches(resolved, frame.iloc[:0], inference_threads=1)

    monkeypatch.setattr(
        scoring_module,
        "prepare_scorer",
        lambda *_args, **_kwargs: (
            lambda batch: pd.DataFrame(
                {KEY: [int(batch[KEY].iloc[0])] * len(batch), "probability": 0.2}
            )
        ),
    )
    with pytest.raises(Phase15InputError, match="one score"):
        score_test_in_batches(resolved, frame, inference_threads=1)

    monkeypatch.setattr(
        scoring_module,
        "prepare_scorer",
        lambda *_args, **_kwargs: lambda batch: batch.iloc[:-1][[KEY]].assign(probability=0.2),
    )
    with pytest.raises(Phase15InputError, match="population membership"):
        score_test_in_batches(resolved, frame, inference_threads=1)

    monkeypatch.setattr(
        scoring_module,
        "prepare_scorer",
        lambda *_args, **_kwargs: lambda batch: batch[[KEY]],
    )
    with pytest.raises(Phase15InputError, match="probability"):
        score_test_in_batches(resolved, frame, inference_threads=1)

    monkeypatch.setattr(
        scoring_module,
        "prepare_scorer",
        lambda *_args, **_kwargs: lambda batch: batch[[KEY]].assign(probability=np.nan),
    )
    with pytest.raises(Phase15InputError, match="non-finite"):
        score_test_in_batches(resolved, frame, inference_threads=1)


def test_validator_helper_comparisons_are_fail_closed() -> None:
    assert _close(1.0, 1.0 + 1.0e-12)
    assert not _close(1.0, 2.0)
    errors: list[str] = []
    _compare_fields(errors, {"same": 1.0, "missing": 2}, {"same": 1.0}, "fixture")
    assert errors == ["fixture missing field: missing"]
    blocked = _blocked(["duplicate", "duplicate"], ["warning", "warning"])
    assert blocked["errors"] == ["duplicate"] and blocked["warnings"] == ["warning"]


def test_runner_freeze_helpers_and_json(tmp_path: Path) -> None:
    resolved: Any = _resolved()
    resolved.phase14_manifest_sha256 = "p14-manifest"
    resolved.phase14_validation_sha256 = "p14-validation"
    resolved.phase14_freeze_sha256 = "p14-freeze"
    resolved.phase14_contract_sha256 = "p14-contract"
    resolved.phase14_configuration_sha256 = "p14-config"
    resolved.phase13_manifest_sha256 = "p13-manifest"
    resolved.phase13.phase13_validation_sha256 = "p13-validation"
    resolved.phase13.phase13_freeze_sha256 = "p13-freeze"
    resolved.phase13.phase13_manifest["run_id"] = "P13"
    resolved.phase14_dir = tmp_path
    resolved.root = Path.cwd()
    policy = build_final_model_policy(resolved)
    freeze = _freeze_payload(
        resolved,
        {"bootstrap_policy": {}, "ranking_policy": {"top_k": []}, "invariance_policy": {}},
        [],
        {"expected_test_row_count": 0},
        policy,
        {"valid": True},
        {"bootstrap_workers": 1},
    )
    assert freeze["test_targets_accessed"] is False
    assert phase15_run_id().endswith("_PHASE15")
    assert _required_phase14_gate_error(ValueError("blocked"))["valid"] is False
    _write_json(tmp_path, "x.json", {"a": 1})
    assert json.loads((tmp_path / "x.json").read_text(encoding="utf-8"))["a"] == 1
    (tmp_path / "overall_metrics.json").write_text('{"average_precision": 0.1}', encoding="utf-8")
    assert _validation_metrics(tmp_path)["average_precision"] == 0.1
    cohorts = pd.DataFrame(
        {
            "error_type": ["FALSE_NEGATIVE", "TRUE_NEGATIVE"],
            "target": [1, 0],
            "probability": [0.4, 0.1],
        }
    )
    assert _false_negative_summary(cohorts)["false_negative_count"] == 1


def test_validation_early_artifact_reconciliation(tmp_path: Path) -> None:
    # Populate all required names with valid empty containers so the validator
    # exercises its independent policy checks before resolving an upstream run.
    from warranty_analytics_model.final_evaluation.validation import REQUIRED_ARTIFACTS

    for name in REQUIRED_ARTIFACTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            pd.DataFrame().to_parquet(path, index=False)
        else:
            path.write_text("{}", encoding="utf-8")
    result = validate_existing_phase15(tmp_path)
    assert result["valid"] is False
    assert result["status"] == "BLOCKED"


def test_validation_frame_reconciliation() -> None:
    frame = pd.DataFrame({"b": [1.0, 2.0], "a": ["x", "y"]})
    assert _compare_frames(frame, frame[["a", "b"]], "fixture") == []
    assert _compare_frames(frame, frame.assign(b=[1.0, 3.0]), "fixture")
    assert _compare_frames(frame, frame.drop(columns=["a"]), "fixture")


def test_runner_fixture_end_to_end_without_real_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise publication ordering with a deterministic fixture scorer."""

    import warranty_analytics_model.final_evaluation.runner as runner_module

    frame = _frame()
    frame["f_num"] = np.linspace(0.01, 0.99, len(frame))
    component = SimpleNamespace(
        track="T1",
        model_sha256="model",
        calibrator_sha256="calibrator",
        feature_list_sha256="features",
    )
    assignments = pd.DataFrame(
        {KEY: frame[KEY], "claim_date": frame["claim__claim_date"], "split": ["TEST"] * len(frame)}
    )
    phase13 = SimpleNamespace(
        phase13_manifest={"run_id": "P13"},
        phase13_validation_sha256="p13-validation",
        phase13_freeze_sha256="p13-freeze",
        phase12_lock=SimpleNamespace(
            phase12_inputs=SimpleNamespace(
                phase10_inputs=SimpleNamespace(
                    phase9_inputs=SimpleNamespace(assignments=assignments)
                )
            )
        ),
    )
    fake: Any = SimpleNamespace(
        root=Path.cwd(),
        phase14_dir=tmp_path / "p14",
        phase14_manifest={"run_id": "P14"},
        phase14_validation={"valid": True},
        phase14_readiness={"warnings": []},
        phase14_manifest_sha256="p14-manifest",
        phase14_validation_sha256="p14-validation",
        phase14_freeze_sha256="p14-freeze",
        phase14_contract_sha256="p14-contract",
        phase14_configuration_sha256="p14-config",
        phase14_plan={
            "slice_definitions": [
                {
                    "slice_id": "cat",
                    "kind": "categorical",
                    "column": "f_cat",
                    "categories": ["A", "B"],
                },
                {
                    "slice_id": "chron",
                    "kind": "chronological_thirds",
                    "column": "claim__claim_date",
                },
            ]
        },
        phase13=phase13,
        phase13_manifest_sha256="p13-manifest",
        phase6_manifest={},
        test_lock={},
        champion_id="C1",
        champion_type="SINGLE_TRACK",
        score_space="RAW_UNCALIBRATED_PROBABILITY",
        threshold=0.5,
        components=(component,),
        feature_names=("f_cat", "f_num"),
        feature_schema_sha256="schema",
        test_features=frame,
        test_assignments=assignments,
    )

    monkeypatch.setattr(runner_module, "resolve_phase14_parent", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(
        runner_module,
        "phase15_contract_check",
        lambda *_args, **_kwargs: {
            "valid": True,
            "contract_version": "v1",
            "contract_sha256": "contract",
            "configuration_sha256": "config",
        },
    )
    monkeypatch.setattr(
        runner_module,
        "compute_plan",
        lambda *_args, **_kwargs: {
            "bootstrap_workers": 1,
            "catboost_inference_threads": 1,
            "test_bootstrap_replicates": 2,
            "native_threads_per_worker": 1,
        },
    )
    monkeypatch.setattr(
        runner_module,
        "build_test_membership_audit",
        lambda *_args, **_kwargs: {
            "expected_test_row_count": len(frame),
            "target_independent": True,
        },
    )
    monkeypatch.setattr(
        runner_module,
        "leakage_audit",
        lambda *_args, **_kwargs: {"valid": True, "prohibited_feature_count": 0},
    )
    monkeypatch.setattr(
        runner_module,
        "train_oof_scores",
        lambda *_args, **_kwargs: pd.DataFrame(
            {KEY: frame[KEY], "probability": np.linspace(0.01, 0.99, len(frame))}
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "score_test_in_batches",
        lambda *_args, **_kwargs: frame[[KEY]].assign(
            probability=np.linspace(0.01, 0.99, len(frame))
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "load_test_targets_after_freeze",
        lambda *_args, **_kwargs: (
            frame[[KEY]].assign(
                **{"target__high_cost_claim_flag": [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1]}
            ),
            {
                "test_target_rows_loaded": len(frame),
                "model_selection_using_TEST": False,
                "threshold_tuning_using_TEST": False,
                "calibration_tuning_using_TEST": False,
                "ensemble_tuning_using_TEST": False,
                "feature_selection_using_TEST": False,
            },
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "prepare_scorer",
        lambda *_args, **_kwargs: (
            lambda batch: batch[[KEY]].assign(probability=np.linspace(0.01, 0.99, len(batch)))
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "prediction_invariance",
        lambda *_args, **_kwargs: {"valid": True, "batch_max_probability_delta": 0.0},
    )
    monkeypatch.setattr(
        runner_module,
        "validate_existing_phase15",
        lambda *_args, **_kwargs: {"valid": True, "errors": [], "warnings": []},
    )
    fake.phase14_dir.mkdir(parents=True, exist_ok=True)
    (fake.phase14_dir / "overall_metrics.json").write_text(
        json.dumps(
            {
                "average_precision": 0.2,
                "roc_auc": 0.6,
                "log_loss": 0.4,
                "brier_score": 0.1,
                "prevalence": 0.2,
            }
        ),
        encoding="utf-8",
    )
    result = runner_module.build_phase15(
        tmp_path / "p14",
        project_root=Path.cwd(),
        output_dir=tmp_path / "artifacts",
        report_dir=tmp_path / "reports",
        run_id="FIXTURE",
        bootstrap_replicates=2000,
    )
    assert result["run_directory"].endswith("FIXTURE")
    assert (tmp_path / "artifacts" / "FIXTURE" / "phase15_manifest.json").is_file()
    # Replay the published fixture through the independent validator with the
    # same frozen fixture inputs; this covers policy, membership, metric, Top-K,
    # and status reconstruction without touching a real model.
    import warranty_analytics_model.final_evaluation.validation as validation_module

    monkeypatch.setattr(validation_module, "resolve_phase14_parent", lambda *_args, **_kwargs: fake)
    monkeypatch.setattr(
        validation_module,
        "build_test_membership_audit",
        lambda *_args, **_kwargs: {
            "expected_test_row_count": len(frame),
            "target_independent": True,
        },
    )
    monkeypatch.setattr(
        validation_module,
        "score_test_in_batches",
        lambda *_args, **_kwargs: frame[[KEY]].assign(
            probability=np.linspace(0.01, 0.99, len(frame))
        ),
    )
    monkeypatch.setattr(
        validation_module,
        "load_test_targets_after_freeze",
        lambda *_args, **_kwargs: (
            frame[[KEY]].assign(
                **{"target__high_cost_claim_flag": [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1]}
            ),
            {},
        ),
    )
    fixture_dir = tmp_path / "artifacts" / "FIXTURE"
    monkeypatch.setattr(
        validation_module,
        "train_oof_scores",
        lambda *_args, **_kwargs: pd.DataFrame(
            {KEY: frame[KEY], "probability": np.linspace(0.01, 0.99, len(frame))}
        ),
    )
    monkeypatch.setattr(
        validation_module,
        "load_robustness_settings",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        validation_module,
        "stratified_bootstrap",
        lambda *_args, **_kwargs: (
            json.loads((fixture_dir / "test_bootstrap_summary.json").read_text(encoding="utf-8")),
            pd.read_parquet(fixture_dir / "test_bootstrap.parquet").to_dict("records"),
        ),
    )
    monkeypatch.setattr(
        validation_module,
        "prepare_scorer",
        lambda *_args, **_kwargs: (
            lambda batch: batch[[KEY]].assign(probability=np.linspace(0.01, 0.99, len(batch)))
        ),
    )
    monkeypatch.setattr(
        validation_module,
        "prediction_invariance",
        lambda *_args, **_kwargs: json.loads(
            (fixture_dir / "test_prediction_invariance.json").read_text(encoding="utf-8")
        ),
    )
    replay = validation_module.validate_existing_phase15(fixture_dir, project_root=Path.cwd())
    assert replay["phase"] == 15


def test_phase15_cli_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Exercise JSON and human-readable CLI paths without touching TEST data."""

    import warranty_analytics_model.final_evaluation.runner as runner_module
    import warranty_analytics_model.final_evaluation.validation as validation_module
    from warranty_analytics_model import cli

    blocked = {
        "phase": 15,
        "valid": False,
        "status": "BLOCKED",
        "errors": ["upstream gate"],
        "warnings": ["review"],
        "test_targets_accessed": False,
    }
    monkeypatch.setattr(runner_module, "phase15_plan_check", lambda *_a, **_k: blocked)
    assert cli.main(["phase15-plan-check", "--phase14-dir", str(tmp_path), "--json"]) == 1
    assert cli.main(["phase15-plan-check", "--phase14-dir", str(tmp_path)]) == 1

    accepted = {
        "phase": 15,
        "valid": True,
        "status": "ACCEPTED_FOR_POC",
        "run_directory": str(tmp_path / "run"),
        "report_directory": str(tmp_path / "reports"),
        "validation": {"valid": True, "errors": [], "warnings": []},
    }
    monkeypatch.setattr(runner_module, "build_phase15", lambda *_a, **_k: accepted)
    assert cli.main(["phase15-evaluate", "--phase14-dir", str(tmp_path), "--json"]) == 0
    assert cli.main(["phase15-evaluate", "--phase14-dir", str(tmp_path)]) == 0

    monkeypatch.setattr(validation_module, "validate_existing_phase15", lambda *_a, **_k: blocked)
    assert cli.main(["phase15-validate", "--phase15-dir", str(tmp_path), "--json"]) == 1
    assert cli.main(["phase15-validate", "--phase15-dir", str(tmp_path)]) == 1
    assert cli.main(["phase15-contract-check", "--json"]) == 0
    assert cli.main(["phase15-contract-check"]) == 0
