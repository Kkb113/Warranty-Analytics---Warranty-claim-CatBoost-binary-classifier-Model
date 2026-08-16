"""Focused Phase 12 policy, metric, threshold, planner, and checkpoint tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from warranty_analytics_model.cli import build_parser
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
    result = validate_existing_phase12(Path("artifacts/imbalance_threshold/20260816T_PHASE12"))
    assert result["valid"] is True
    assert result["hardening_status"] == "HARDENED_PASS"
