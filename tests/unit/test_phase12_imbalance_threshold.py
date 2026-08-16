"""Focused Phase 12 policy, metric, threshold, planner, and checkpoint tests."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from warranty_analytics_model.cli import build_parser
from warranty_analytics_model.imbalance_threshold import runner as phase12_runner
from warranty_analytics_model.imbalance_threshold import validation as phase12_validation
from warranty_analytics_model.imbalance_threshold.checkpoint import (
    load_valid_checkpoint,
    write_checkpoint,
)
from warranty_analytics_model.imbalance_threshold.config import (
    STRATEGY_IDS,
    ImbalanceThresholdError,
    load_imbalance_threshold_settings,
    settings_payload,
)
from warranty_analytics_model.imbalance_threshold.contract import (
    validate_imbalance_threshold_contract,
)
from warranty_analytics_model.imbalance_threshold.metrics import (
    aggregate_strategy_metrics,
    fold_metric_row,
    ranking_metrics,
    strategy_fold_metrics_frame,
    threshold_metrics,
    validate_binary_inputs,
)
from warranty_analytics_model.imbalance_threshold.planner import build_compute_plan
from warranty_analytics_model.imbalance_threshold.selection import (
    compare_phase12_candidates,
    replacement_decision,
    select_phase12_champion,
    select_strategy,
)
from warranty_analytics_model.imbalance_threshold.strategies import (
    StrategyDefinition,
    build_strategy_definitions,
    strategy_parameter_payload,
    strategy_parameters,
    validate_strategy_parameters,
)
from warranty_analytics_model.imbalance_threshold.thresholds import (
    build_threshold_curve,
    pareto_frontier,
    select_mcc_threshold,
    threshold_grid,
    threshold_summary,
)
from warranty_analytics_model.imbalance_threshold.validation import validate_existing_phase12


def test_phase12_contract_and_configuration_are_locked() -> None:
    settings = load_imbalance_threshold_settings()
    assert settings.strategy_ids == STRATEGY_IDS
    assert settings.threshold_start == 0.001
    assert settings.threshold_stop == 0.999
    assert settings.threshold_step == 0.001
    assert validate_imbalance_threshold_contract()["valid"] is True


def test_strategy_inventory_resolves_auto_values_and_one_mechanism() -> None:
    strategies = build_strategy_definitions(3, 97)
    assert len(strategies) == 8
    assert strategies[0].parameter is None
    assert strategies[6].parameter == "SqrtBalanced"
    assert strategies[6].resolved_parameter == (97 / 3) ** 0.5
    assert strategies[7].resolved_parameter == 97 / 3
    base = {"iterations": 3, "thread_count": 2, "scale_pos_weight": 99}
    weighted = strategy_parameters(base, strategies[1])
    assert weighted["scale_pos_weight"] == 2.0
    assert "auto_class_weights" not in weighted


def test_phase12_policy_helpers_and_failure_guards() -> None:
    settings = load_imbalance_threshold_settings()
    payload = settings_payload(settings)
    assert payload["tracks"] == ["T1", "T3"]
    assert payload["threshold"]["grid"] == {"start": 0.001, "stop": 0.999, "step": 0.001}

    with pytest.raises(ImbalanceThresholdError):
        build_strategy_definitions(positive_count=3)
    with pytest.raises(ImbalanceThresholdError):
        build_strategy_definitions(positive_count=0, negative_count=3)

    strategies = build_strategy_definitions()
    assert strategies[0].weighted is False
    assert strategies[1].weighted is True
    assert strategy_parameter_payload(strategies[1], resolved_value=7.0)["parameter"] == 7.0

    base = {"iterations": 3, "thread_count": 2}
    none_parameters = strategy_parameters(base, strategies[0])
    validate_strategy_parameters(none_parameters, strategies[0], base)
    with pytest.raises(ImbalanceThresholdError):
        strategy_parameters(
            base,
            StrategyDefinition("bad", "scale_pos_weight", None, 0, ""),
        )
    with pytest.raises(ImbalanceThresholdError):
        strategy_parameters(
            base,
            StrategyDefinition("bad", "auto_class_weights", 4.0, 0, ""),
        )
    with pytest.raises(ImbalanceThresholdError):
        strategy_parameters(
            base,
            StrategyDefinition("bad", "unsupported", None, 0, ""),
        )
    with pytest.raises(ImbalanceThresholdError):
        validate_strategy_parameters({"iterations": 4}, strategies[0], base)
    with pytest.raises(ImbalanceThresholdError):
        validate_strategy_parameters(
            {"iterations": 3, "scale_pos_weight": 2.0}, strategies[0], base
        )
    with pytest.raises(ImbalanceThresholdError):
        validate_strategy_parameters(
            {"iterations": 3, "scale_pos_weight": 2.0, "auto_class_weights": "Balanced"},
            strategies[1],
            base,
        )
    with pytest.raises(ImbalanceThresholdError):
        validate_strategy_parameters(
            {"iterations": 3, "scale_pos_weight": 2.0}, strategies[6], base
        )

    with pytest.raises(ImbalanceThresholdError):
        from warranty_analytics_model.imbalance_threshold.config import _number, _positive_int

        _number(True, "test")
    with pytest.raises(ImbalanceThresholdError):
        from warranty_analytics_model.imbalance_threshold.config import _number

        _number(float("inf"), "test")
    with pytest.raises(ImbalanceThresholdError):
        from warranty_analytics_model.imbalance_threshold.config import _positive_int

        _positive_int(False, "test")
    with pytest.raises(ImbalanceThresholdError):
        from warranty_analytics_model.imbalance_threshold.config import _positive_int

        _positive_int(0, "test")


def test_compute_plan_downscales_without_oversubscription() -> None:
    settings = load_imbalance_threshold_settings()
    plan = build_compute_plan(settings, logical_processors=3)
    assert plan.effective_cpu_budget == 1
    assert plan.worker_count * plan.threads_per_fit <= plan.effective_cpu_budget
    assert plan.single_fit_threads <= plan.effective_cpu_budget
    with pytest.raises(ImbalanceThresholdError):
        build_compute_plan(settings, logical_processors=-1)
    with pytest.raises(ImbalanceThresholdError):
        build_compute_plan(settings, logical_processors=4, max_workers=-1)
    expanded = build_compute_plan(
        settings,
        logical_processors=4,
        max_workers=100,
        threads_per_fit=100,
        single_fit_threads=100,
    )
    assert expanded.maximum_concurrent_threads <= expanded.effective_cpu_budget
    assert expanded.single_fit_threads == expanded.effective_cpu_budget


def test_phase12_runner_manifest_helpers_are_deterministic(tmp_path: Path) -> None:
    run_id = phase12_runner.phase12_run_id()
    assert len(run_id) == 16
    output_root = tmp_path / "artifacts"
    assert phase12_runner._phase12_root(tmp_path, output_root) == output_root.resolve()
    work_dir, final_dir = phase12_runner._work_dirs(output_root, "fixture")
    assert work_dir.name == ".phase12_fixture.work"
    assert final_dir.name == "fixture"
    targets = pd.DataFrame({"warranty_claim_key": [2, 1], "target__high_cost_claim_flag": [0, 1]})
    assert phase12_runner._target_by_key(targets).to_dict() == {2: 0, 1: 1}
    assert phase12_runner._parameter_hash({"b": 2, "a": 1}) == phase12_runner._parameter_hash(
        {"a": 1, "b": 2}
    )
    assert (
        phase12_runner._base_parameters(SimpleNamespace(statistical_parameters={}), 3)[
            "thread_count"
        ]
        == 3
    )
    output_root.mkdir()
    (output_root / "artifact.txt").write_text("fixture", encoding="utf-8")
    assert "artifact.txt" in phase12_runner._artifact_hashes(output_root)
    work_dir.mkdir()
    (work_dir / "artifact.txt").write_text("fixture", encoding="utf-8")
    assert "artifact.txt" in phase12_runner._artifact_hashes(work_dir)
    assert phase12_runner.phase12_contract_check(tmp_path.parent).get("valid") is True


def test_ranking_and_threshold_metrics_are_finite_for_zero_positive_case() -> None:
    y = np.zeros(4, dtype="int8")
    probabilities = np.array([0.1, 0.2, 0.3, 0.4])
    ranking = ranking_metrics(y, probabilities)
    operating = threshold_metrics(y, probabilities, 0.5)
    assert all(np.isfinite(value) for value in ranking.values())
    assert operating["tp"] == operating["fp"] == operating["fn"] == 0
    assert operating["tn"] == 4
    assert all(
        np.isfinite(float(value)) for value in operating.values() if isinstance(value, (int, float))
    )


def test_phase12_metric_schema_and_input_guards() -> None:
    with pytest.raises(ValueError):
        validate_binary_inputs([0], [0.1, 0.2])
    with pytest.raises(ValueError):
        validate_binary_inputs([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError):
        validate_binary_inputs([0, 1], [0.1, float("nan")])
    with pytest.raises(ValueError):
        threshold_metrics([0, 1], [0.1, 0.9], 0.0)

    row = fold_metric_row(
        [0, 1, 0, 1],
        [0.1, 0.9, 0.2, 0.8],
        track="T1",
        strategy_id="S0_NONE",
        fold_id=1,
        train_positive_count=2,
        train_negative_count=2,
        training_seconds=0.1,
        weighting_parameters={},
    )
    assert row["validation_positive_count"] == 2
    frame = strategy_fold_metrics_frame([{"track": "T1", "strategy_id": "S0_NONE", "fold_id": 1}])
    assert list(frame.columns)[0:3] == ["track", "strategy_id", "fold_id"]
    with pytest.raises(ValueError):
        aggregate_strategy_metrics([])


def test_threshold_grid_and_tie_break_are_deterministic() -> None:
    assert threshold_grid() == tuple(float(f"{index / 1000:.3f}") for index in range(1, 1000))
    y = np.array([0, 1, 0, 1], dtype="int8")
    probabilities = np.array([0.2, 0.8, 0.2, 0.8])
    curve = build_threshold_curve(y, probabilities, track="T1", strategy_id="S0_NONE")
    selected = select_mcc_threshold(curve)
    assert selected["threshold"] == 0.201
    assert len(curve) == 999
    summary = threshold_summary(curve)
    assert set(summary["alternatives"]) == {"F1_MAX", "F2_MAX", "BALANCED_ACCURACY_MAX"}
    assert summary["pareto_frontier"]


def test_strategy_selection_guardrails_and_none_preference() -> None:
    summary = pd.DataFrame(
        [
            {
                "track": "T1",
                "strategy_id": "S0_NONE",
                "mean_average_precision": 0.10,
                "min_average_precision": 0.09,
                "std_average_precision": 0.01,
                "mean_roc_auc": 0.70,
            },
            {
                "track": "T1",
                "strategy_id": "S1_SCALE_POS_WEIGHT_2",
                "mean_average_precision": 0.101,
                "min_average_precision": 0.09,
                "std_average_precision": 0.01,
                "mean_roc_auc": 0.70,
            },
        ]
    )
    thresholds = {
        "S0_NONE": {"technical_default": {"threshold": 0.2, "metrics": {"mcc": 0.20}}},
        "S1_SCALE_POS_WEIGHT_2": {
            "technical_default": {"threshold": 0.2, "metrics": {"mcc": 0.201}}
        },
    }
    decision = select_strategy(summary, thresholds)
    assert decision["selected_strategy_id"] == "S0_NONE"
    assert decision["none_preferred"] is True


def test_replacement_routes_and_champion_tie_break() -> None:
    parent = {"average_precision": 0.10, "roc_auc": 0.70, "mcc": 0.10}
    weighted = {"average_precision": 0.101, "roc_auc": 0.699, "mcc": 0.20}
    decision = replacement_decision(parent, weighted)
    assert decision["replace_parent"] is True
    assert decision["route_a_ranking_improvement"] is True
    route_b = replacement_decision(
        {"average_precision": 0.10, "roc_auc": 0.70, "mcc": 0.10},
        {"average_precision": 0.0998, "roc_auc": 0.699, "mcc": 0.20},
    )
    assert route_b["reason"] == "ROUTE_B_OPERATING_POINT"
    fallback = replacement_decision(
        {"average_precision": 0.10, "roc_auc": 0.70, "mcc": 0.10},
        {"average_precision": 0.09, "roc_auc": 0.60, "mcc": 0.20},
    )
    assert fallback["reason"] == "FALLBACK_PARENT"
    candidates = [
        {
            "candidate_id": "T3",
            "feature_count": 100,
            "complexity_order": 2,
            "validation_metrics": {
                "average_precision": 0.2,
                "mcc": 0.3,
                "roc_auc": 0.7,
                "log_loss": 0.1,
            },
        },
        {
            "candidate_id": "T1",
            "feature_count": 80,
            "complexity_order": 1,
            "validation_metrics": {
                "average_precision": 0.2,
                "mcc": 0.3,
                "roc_auc": 0.7,
                "log_loss": 0.1,
            },
        },
    ]
    assert select_phase12_champion(candidates) == "T1"
    with pytest.raises(ValueError, match="No Phase 12"):
        select_phase12_champion([])


def test_champion_comparator_applies_one_e6_tolerance_before_tie_breaks() -> None:
    def candidate(
        candidate_id: str,
        *,
        average_precision: float,
        mcc: float,
        roc_auc: float,
        log_loss: float,
        feature_count: int = 100,
        complexity_order: int = 1,
    ) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "feature_count": feature_count,
            "complexity_order": complexity_order,
            "validation_metrics": {
                "average_precision": average_precision,
                "mcc": mcc,
                "roc_auc": roc_auc,
                "log_loss": log_loss,
            },
        }

    base = candidate("base", average_precision=0.2, mcc=0.3, roc_auc=0.7, log_loss=0.2)
    ap_tie_mcc_winner = candidate(
        "mcc-winner",
        average_precision=0.2000005,
        mcc=0.31,
        roc_auc=0.7,
        log_loss=0.2,
    )
    assert compare_phase12_candidates(ap_tie_mcc_winner, base) < 0

    ap_winner = candidate(
        "ap-winner",
        average_precision=0.200002,
        mcc=0.1,
        roc_auc=0.1,
        log_loss=0.9,
    )
    assert compare_phase12_candidates(ap_winner, base) < 0

    roc_winner = candidate(
        "roc-winner",
        average_precision=0.2,
        mcc=0.3,
        roc_auc=0.71,
        log_loss=0.2,
    )
    assert compare_phase12_candidates(roc_winner, base) < 0

    log_loss_winner = candidate(
        "logloss-winner",
        average_precision=0.2,
        mcc=0.3,
        roc_auc=0.7,
        log_loss=0.19,
    )
    assert compare_phase12_candidates(log_loss_winner, base) < 0

    assert (
        select_phase12_champion(
            [
                candidate(
                    "higher-ap-outside-tolerance",
                    average_precision=0.200002,
                    mcc=0.0,
                    roc_auc=0.0,
                    log_loss=1.0,
                ),
                base,
            ]
        )
        == "higher-ap-outside-tolerance"
    )


def test_pareto_frontier_returns_true_nondominated_points() -> None:
    curve = pd.DataFrame(
        [
            {"threshold": 0.1, "precision": 0.20, "recall": 0.90},
            {"threshold": 0.2, "precision": 0.30, "recall": 0.80},
            {"threshold": 0.3, "precision": 0.25, "recall": 0.70},
            {"threshold": 0.4, "precision": 0.40, "recall": 0.60},
            {"threshold": 0.5, "precision": 0.40, "recall": 0.50},
            {"threshold": 0.6, "precision": 0.50, "recall": 0.40},
        ]
    )
    frontier = pareto_frontier(curve)
    assert [row["threshold"] for row in frontier] == [0.1, 0.2, 0.4, 0.6]


def test_phase12_validator_fixture_is_fail_closed(tmp_path: Path) -> None:
    fixture_bundle = tmp_path / "phase12_fixture"
    fixture_bundle.mkdir()
    result = validate_existing_phase12(
        fixture_bundle, project_root=Path(__file__).resolve().parents[2]
    )
    assert result["valid"] is False
    assert result["hardening_status"] == "BLOCKED"
    assert any("Phase 12 artifacts missing" in error for error in result["errors"])


def test_phase12_validator_helpers_and_model_parameter_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    valid_json = tmp_path / "valid.json"
    valid_json.write_text('{"ok": true}', encoding="utf-8")
    assert phase12_validation._read_json(valid_json) == {"ok": True}
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ImbalanceThresholdError, match="JSON object"):
        phase12_validation._read_json(invalid_json)
    assert phase12_validation._finite_equal(1.0, 1.0 + 1.0e-12)
    assert not phase12_validation._finite_equal("left", "right")

    expected = pd.DataFrame({"number": [1.0], "label": ["ok"]})
    assert phase12_validation._compare_frame(expected, expected, ["number", "label"]) == []
    assert phase12_validation._compare_frame(
        expected.assign(number=[2.0]), expected, ["number", "label"]
    )
    assert phase12_validation._compare_frame(
        expected.assign(label=["different"]), expected, ["number", "label"]
    )
    assert phase12_validation._compare_frame(expected.iloc[:0], expected, ["number", "label"])
    assert phase12_validation._compare_frame(expected, expected, ["other"])

    payload_errors: list[str] = []
    phase12_validation._compare_payload(
        {"number": 1.0, "label": "ok"}, {}, "fixture", payload_errors
    )
    phase12_validation._compare_payload({"number": 1.0}, {"number": 2.0}, "fixture", payload_errors)
    phase12_validation._compare_payload(
        {"label": "ok"}, {"label": "bad"}, "fixture", payload_errors
    )
    phase12_validation._compare_payload({"label": "ok"}, None, "fixture", payload_errors)
    assert len(payload_errors) == 5

    class FakeModel:
        pass

    model = FakeModel()
    parameters: dict[str, object] = {}
    monkeypatch.setattr(phase12_validation, "load_model", lambda _path: model)
    monkeypatch.setattr(phase12_validation, "effective_parameters", lambda _model: parameters)
    strategies = build_strategy_definitions(3, 97)
    phase12_validation._validate_model_parameters(tmp_path / "none.cbm", {}, strategies[0])
    with pytest.raises(ImbalanceThresholdError, match="parameter mismatch"):
        phase12_validation._validate_model_parameters(
            tmp_path / "missing.cbm", {"iterations": 3}, strategies[0]
        )
    parameters["iterations"] = 4
    with pytest.raises(ImbalanceThresholdError, match="parameter mismatch"):
        phase12_validation._validate_model_parameters(
            tmp_path / "drift.cbm", {"iterations": 3}, strategies[0]
        )
    parameters.clear()
    parameters["task_type"] = "CPU"
    with pytest.raises(ImbalanceThresholdError, match="parameter mismatch"):
        phase12_validation._validate_model_parameters(
            tmp_path / "text-drift.cbm", {"task_type": "GPU"}, strategies[0]
        )
    parameters.clear()

    parameters.update({"scale_pos_weight": 2.0})
    with pytest.raises(ImbalanceThresholdError, match="S0_NONE"):
        phase12_validation._validate_model_parameters(tmp_path / "none.cbm", {}, strategies[0])
    phase12_validation._validate_model_parameters(tmp_path / "scale.cbm", {}, strategies[1])
    parameters["scale_pos_weight"] = 3.0
    with pytest.raises(ImbalanceThresholdError, match="Scale-positive"):
        phase12_validation._validate_model_parameters(tmp_path / "scale.cbm", {}, strategies[1])
    parameters.update({"scale_pos_weight": 2.0, "auto_class_weights": "Balanced"})
    with pytest.raises(ImbalanceThresholdError, match="also contains"):
        phase12_validation._validate_model_parameters(tmp_path / "scale.cbm", {}, strategies[1])

    parameters.clear()
    parameters.update(
        {
            "auto_class_weights": "SqrtBalanced",
            "class_weights": [1.0, (97.0 / 3.0) ** 0.5],
        }
    )
    phase12_validation._validate_model_parameters(tmp_path / "auto.cbm", {}, strategies[6])
    parameters.clear()
    with pytest.raises(ImbalanceThresholdError, match="Auto CatBoost"):
        phase12_validation._validate_model_parameters(tmp_path / "auto.cbm", {}, strategies[6])
    parameters.update({"class_weights": [0.0, 1.0]})
    with pytest.raises(ImbalanceThresholdError, match="Auto CatBoost"):
        phase12_validation._validate_model_parameters(tmp_path / "auto.cbm", {}, strategies[6])
    invalid_scale = StrategyDefinition("invalid", "scale_pos_weight", None, 1, "hash")
    parameters.clear()
    with pytest.raises(ImbalanceThresholdError, match="no value"):
        phase12_validation._validate_model_parameters(tmp_path / "invalid.cbm", {}, invalid_scale)
    invalid_scale_type = StrategyDefinition("invalid", "scale_pos_weight", "2", 1, "hash")
    with pytest.raises(ImbalanceThresholdError, match="no numeric"):
        phase12_validation._validate_model_parameters(
            tmp_path / "invalid-type.cbm", {}, invalid_scale_type
        )


def test_phase12_validator_accepts_a_synthetic_fixture_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the independent validator with a small, self-contained bundle."""

    root = Path(__file__).resolve().parents[2]
    bundle = tmp_path / "phase12_fixture"
    bundle.mkdir()
    required_json = {
        "phase12_manifest.json",
        "phase11_parent_resolution.json",
        "strategy_definitions.json",
        "threshold_summary.json",
        "phase12_freeze.json",
        "validation_metrics.json",
        "effective_model_manifest.json",
        "threshold_policy.json",
        "model_manifest.json",
        "target_access_audit.json",
        "compute_manifest.json",
        "validation.json",
    }
    required_parquet = {
        "strategy_fold_metrics.parquet",
        "strategy_summary.parquet",
        "strategy_oof_predictions.parquet",
        "threshold_curve.parquet",
        "validation_predictions.parquet",
    }
    for name in required_json | required_parquet:
        (bundle / name).write_bytes(b"fixture")

    tracks = ("T1", "T3")
    fold_keys = (1, 2, 3)
    train_targets = pd.DataFrame(
        {
            "warranty_claim_key": [1, 2, 3],
            "target__high_cost_claim_flag": [0, 1, 0],
        }
    )
    validation_targets = pd.DataFrame(
        {
            "warranty_claim_key": [4, 5],
            "target__high_cost_claim_flag": [0, 1],
        }
    )
    development = pd.DataFrame(
        {
            "warranty_claim_key": [1, 2, 3, 4, 5],
            "split": ["TRAIN", "TRAIN", "TRAIN", "VALIDATION", "VALIDATION"],
        }
    )
    feature_set = SimpleNamespace(feature_count=1, feature_set_sha256="fixture-feature")
    parents = {
        track: SimpleNamespace(
            track=track,
            effective_candidate_id=f"P11_{track}",
            statistical_parameters={},
            feature_set=feature_set,
        )
        for track in tracks
    }
    folds = tuple(
        SimpleNamespace(
            fold_id=index,
            train_keys=(1, 2, 3),
            validation_keys=(key,),
        )
        for index, key in enumerate(fold_keys, start=1)
    )
    inputs = SimpleNamespace(
        root=root,
        phase11_manifest={"run_id": "PHASE11_FIXTURE"},
        phase11_manifest_sha256="phase11-manifest",
        phase11_validation_sha256="phase11-validation",
        phase11_model_manifest_sha256="phase11-model-manifest",
        phase10_inputs=SimpleNamespace(),
        fold_plan=SimpleNamespace(folds=folds, content_sha256="fold-plan"),
        parents=parents,
        development=development,
    )
    strategies = build_strategy_definitions(1, 2)
    strategy_ids = tuple(item.strategy_id for item in strategies)

    fold_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    train_by_key = train_targets.set_index("warranty_claim_key")["target__high_cost_claim_flag"]
    for track in tracks:
        for strategy_id in strategy_ids:
            for fold_id, key in enumerate(fold_keys, start=1):
                probability = 0.2 if int(train_by_key.loc[key]) == 0 else 0.8
                metrics = fold_metric_row(
                    [int(train_by_key.loc[key])],
                    [probability],
                    track=track,
                    strategy_id=strategy_id,
                    fold_id=fold_id,
                    train_positive_count=1,
                    train_negative_count=2,
                    training_seconds=0.0,
                    weighting_parameters={},
                )
                fold_rows.append(metrics)
                oof_rows.append(
                    {
                        "warranty_claim_key": key,
                        "track": track,
                        "strategy_id": strategy_id,
                        "fold_id": fold_id,
                        "high_cost_probability": probability,
                    }
                )
    fold_frame = strategy_fold_metrics_frame(fold_rows)
    summary_frame = pd.DataFrame(
        [
            aggregate_strategy_metrics(
                [
                    row
                    for row in fold_rows
                    if row["track"] == track and row["strategy_id"] == strategy_id
                ]
            )
            for track in tracks
            for strategy_id in strategy_ids
        ]
    )
    oof_frame = pd.DataFrame(oof_rows)
    curve_rows = [
        {
            "track": track,
            "strategy_id": strategy_id,
            "threshold": 0.5,
            "precision": 0.5,
            "recall": 0.5,
        }
        for track in tracks
        for strategy_id in strategy_ids
    ]
    curve_frame = pd.DataFrame(curve_rows)
    threshold_payload = {
        track: {
            strategy_id: {
                "technical_default": {"threshold": 0.5, "metrics": {"mcc": 0.0}},
                "alternatives": {},
                "precision_at_recall": {},
                "pareto_frontier": [],
            }
            for strategy_id in strategy_ids
        }
        for track in tracks
    }
    parent_probabilities = np.array([0.8, 0.2])
    weighted_probabilities = np.array([0.1, 0.9])
    validation_y = validation_targets["target__high_cost_claim_flag"].to_numpy(dtype="int8")
    parent_metrics = ranking_metrics(validation_y, parent_probabilities)
    parent_metrics.update(threshold_metrics(validation_y, parent_probabilities, 0.5))
    weighted_metrics = ranking_metrics(validation_y, weighted_probabilities)
    weighted_metrics.update(threshold_metrics(validation_y, weighted_probabilities, 0.5))
    selected_strategy_id = "S1_SCALE_POS_WEIGHT_2"
    model_entries: list[dict[str, object]] = []
    effective_entries: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for track in tracks:
        parent_id = f"P11_{track}"
        weighted_id = f"P12_{track}_{selected_strategy_id}"
        parent_file = f"models/{track.lower()}_parent.cbm"
        weighted_file = f"models/{track.lower()}_weighted.cbm"
        (bundle / "models").mkdir(exist_ok=True)
        (bundle / parent_file).write_bytes(b"parent")
        (bundle / weighted_file).write_bytes(b"weighted")
        parent_entry = {
            "track": track,
            "candidate_id": parent_id,
            "model_file": parent_file,
            "model_sha256": "model-hash",
            "imbalance_strategy": {"strategy_id": "S0_NONE"},
            "validation_metrics": parent_metrics,
        }
        weighted_entry = {
            "track": track,
            "candidate_id": weighted_id,
            "model_file": weighted_file,
            "model_sha256": "model-hash",
            "imbalance_strategy": {"strategy_id": selected_strategy_id},
            "validation_metrics": weighted_metrics,
        }
        model_entries.extend((parent_entry, weighted_entry))
        decision = replacement_decision(parent_metrics, weighted_metrics)
        decision.update(
            {
                "selected_strategy_id": selected_strategy_id,
                "effective_candidate_id": weighted_id,
                "fallback_to_phase11_parent": False,
            }
        )
        effective_entries.append(
            {
                "track": track,
                "candidate_id": weighted_id,
                "feature_count": 1,
                "technical_threshold": 0.5,
                "selected_imbalance_strategy": selected_strategy_id,
                "validation_metrics": weighted_metrics,
                "replacement_decision": decision,
            }
        )
        validation_rows.extend(
            {
                "warranty_claim_key": key,
                "track": track,
                "candidate_id": candidate_id,
                "high_cost_probability": float(probability),
            }
            for candidate_id, probabilities in (
                (parent_id, parent_probabilities),
                (weighted_id, weighted_probabilities),
            )
            for key, probability in zip((4, 5), probabilities, strict=True)
        )
    validation_frame = pd.DataFrame(validation_rows)
    expected_champion = effective_entries[0]["candidate_id"]
    json_payloads: dict[str, dict[str, object]] = {
        "phase12_manifest.json": {
            "phase": 12,
            "run_id": "PHASE12_FIXTURE",
            "artifact_file_sha256": {},
            "contract_sha256": "contract",
            "phase11_run_id": "PHASE11_FIXTURE",
            "phase11_manifest_sha256": "phase11-manifest",
            "phase11_validation_sha256": "phase11-validation",
            "phase11_model_manifest_sha256": "phase11-model-manifest",
            "phase10_inner_fold_sha256": "fold-plan",
            "selected_strategies": {track: selected_strategy_id for track in tracks},
        },
        "phase11_parent_resolution.json": {"phase11_manifest_sha256": "phase11-manifest"},
        "strategy_definitions.json": {"strategies": [item.as_dict() for item in strategies]},
        "threshold_summary.json": threshold_payload,
        "phase12_freeze.json": {
            "phase": 12,
            "outer_validation_accessed": False,
            "test_target_accessed": False,
            "phase12_freeze_sha256": "placeholder",
        },
        "validation_metrics.json": {"development_champion": expected_champion},
        "effective_model_manifest.json": {"models": effective_entries},
        "threshold_policy.json": {},
        "model_manifest.json": {"models": model_entries},
        "target_access_audit.json": {
            "test_target_rows_loaded": 0,
            "test_predictions_created": 0,
            "test_metrics_computed": False,
            "test_target_access_allowed": False,
            "first_allowed_test_target_phase": 15,
        },
        "compute_manifest.json": {},
        "validation.json": {"valid": True, "hardening_status": "HARDENED_PASS"},
    }
    freeze_copy = dict(json_payloads["phase12_freeze.json"])
    freeze_copy.pop("phase12_freeze_sha256")
    json_payloads["phase12_freeze.json"]["phase12_freeze_sha256"] = (
        phase12_validation.canonical_json_sha256(freeze_copy)
    )
    json_payloads["phase12_manifest.json"]["phase12_freeze_sha256"] = json_payloads[
        "phase12_freeze.json"
    ]["phase12_freeze_sha256"]
    parquet_payloads = {
        "strategy_fold_metrics.parquet": fold_frame,
        "strategy_summary.parquet": summary_frame,
        "strategy_oof_predictions.parquet": oof_frame,
        "threshold_curve.parquet": curve_frame,
        "validation_predictions.parquet": validation_frame,
    }

    class FakeModel:
        def __init__(self, weighted: bool) -> None:
            self.weighted = weighted

        def predict_proba(self, _pool: object) -> np.ndarray:
            values = weighted_probabilities if self.weighted else parent_probabilities
            return np.column_stack((1.0 - values, values))

    def fake_load_model(path: Path) -> FakeModel:
        return FakeModel("weighted" in path.name)

    def fake_effective_parameters(model: FakeModel) -> dict[str, object]:
        return {"scale_pos_weight": 2.0} if model.weighted else {}

    monkeypatch.setattr(phase12_validation, "discover_repository_root", lambda _start: root)
    monkeypatch.setattr(
        phase12_validation,
        "validate_imbalance_threshold_contract",
        lambda _root: {"valid": True, "contract_checksum": "contract"},
    )
    monkeypatch.setattr(
        phase12_validation, "load_locked_phase11_inputs", lambda *_args, **_kwargs: inputs
    )
    monkeypatch.setattr(
        phase12_validation,
        "load_train_targets_for_optimization",
        lambda *_args, **_kwargs: (train_targets, {}),
    )
    monkeypatch.setattr(
        phase12_validation,
        "load_validation_targets_after_freeze",
        lambda *_args, **_kwargs: (validation_targets, {}),
    )
    monkeypatch.setattr(phase12_validation, "sha256_file", lambda _path: "model-hash")
    monkeypatch.setattr(
        phase12_validation,
        "select_strategy",
        lambda *_args, **_kwargs: {"selected_strategy_id": selected_strategy_id},
    )
    monkeypatch.setattr(
        phase12_validation,
        "build_threshold_curve",
        lambda _y, _p, *, track, strategy_id: curve_frame.loc[
            (curve_frame["track"] == track) & (curve_frame["strategy_id"] == strategy_id)
        ].copy(),
    )
    monkeypatch.setattr(
        phase12_validation,
        "threshold_summary",
        lambda _curve: threshold_payload["T1"]["S0_NONE"],
    )
    monkeypatch.setattr(phase12_validation, "load_model", fake_load_model)
    monkeypatch.setattr(phase12_validation, "effective_parameters", fake_effective_parameters)
    import warranty_analytics_model.baseline_model.adapters as adapters
    import warranty_analytics_model.baseline_model.catboost_baseline as baseline_module
    import warranty_analytics_model.baseline_model.config as baseline_config

    baseline_settings = baseline_config.load_baseline_settings(root)
    monkeypatch.setattr(baseline_config, "load_baseline_settings", lambda _root: baseline_settings)
    monkeypatch.setattr(adapters, "adapt_matrix", lambda frame, *_args: frame)
    monkeypatch.setattr(baseline_module, "build_pool", lambda frame, *_args: frame)
    monkeypatch.setattr(baseline_module, "load_model", fake_load_model)
    monkeypatch.setattr(baseline_module, "effective_parameters", fake_effective_parameters)
    monkeypatch.setattr(
        phase12_validation.pd,
        "read_parquet",
        lambda path: parquet_payloads[Path(path).name].copy(),
    )
    monkeypatch.setattr(
        phase12_validation,
        "_read_json",
        lambda path: json_payloads[Path(path).name],
    )

    result = validate_existing_phase12(bundle, project_root=root)
    assert result["valid"] is True, result
    assert result["hardening_status"] == "HARDENED_PASS"
    assert result["phase12_development_champion"] == expected_champion
    assert result["test_seal"]["test_target_rows_loaded"] == 0


def test_strategy_aggregation_and_checkpoint_resume(tmp_path) -> None:
    rows = [
        {
            "track": "T1",
            "strategy_id": "S0_NONE",
            "fold_id": index,
            "average_precision": 0.1 + index / 100,
            "roc_auc": 0.7,
            "log_loss": 0.2,
            "brier_score": 0.1,
            "training_seconds": 1.0,
        }
        for index in (1, 2, 3)
    ]
    aggregate = aggregate_strategy_metrics(rows)
    assert aggregate["fold_count"] == 3
    path = write_checkpoint(
        tmp_path,
        track="T1",
        strategy_id="S0_NONE",
        fold_id=1,
        feature_set_sha256="f",
        parent_parameter_sha256="p",
        strategy_parameter_sha256="s",
        fold_membership_sha256="m",
        metrics=rows[0],
        prediction_sha256="pred",
        training_seconds=1.0,
        prediction_keys=[1, 2],
        prediction_values=[0.1, 0.2],
    )
    assert path.is_file()
    checkpoint = load_valid_checkpoint(
        tmp_path,
        track="T1",
        strategy_id="S0_NONE",
        fold_id=1,
        feature_set_sha256="f",
        parent_parameter_sha256="p",
        strategy_parameter_sha256="s",
        fold_membership_sha256="m",
    )
    assert checkpoint is not None
    assert checkpoint["prediction_values"] == [0.1, 0.2]
    assert (
        load_valid_checkpoint(
            tmp_path,
            track="T1",
            strategy_id="S0_NONE",
            fold_id=1,
            feature_set_sha256="tampered",
            parent_parameter_sha256="p",
            strategy_parameter_sha256="s",
            fold_membership_sha256="m",
        )
        is None
    )


def test_phase12_cli_commands_are_registered() -> None:
    parser = build_parser()
    assert parser.parse_args(["phase12-contract-check"]).command == "phase12-contract-check"
    assert (
        parser.parse_args(["phase12-plan-check", "--phase11-dir", "x"]).command
        == "phase12-plan-check"
    )
    assert (
        parser.parse_args(["phase12-optimize", "--phase11-dir", "x"]).command == "phase12-optimize"
    )
    assert (
        parser.parse_args(["phase12-validate", "--phase12-dir", "x"]).command == "phase12-validate"
    )


def test_published_phase12_bundle_passes_independent_validator() -> None:
    acceptance_dir = os.environ.get("PHASE12_ACCEPTANCE_DIR")
    if not acceptance_dir:
        pytest.skip("Set PHASE12_ACCEPTANCE_DIR for local generated-bundle acceptance.")
    result = validate_existing_phase12(Path(acceptance_dir))
    assert result["valid"] is True
    assert result["hardening_status"] == "HARDENED_PASS"
