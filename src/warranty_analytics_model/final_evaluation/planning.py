"""Target-independent TEST definitions and frozen evaluation plan."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ..catboost_optimization.provenance import canonical_json_sha256
from ..robustness_analysis.input import KEY
from ..robustness_analysis.slices import membership_for_definition
from .config import Phase15Settings, configuration_sha256
from .input import Phase15InputError, Phase15Resolved, build_test_membership_audit


def _chronological_definition(frame: pd.DataFrame) -> dict[str, Any]:
    if "claim__claim_date" not in frame.columns:
        raise Phase15InputError("Phase 15 requires claim__claim_date for TEST chronology.")
    ordered = (
        frame[[KEY, "claim__claim_date"]]
        .assign(_date=pd.to_datetime(frame["claim__claim_date"], errors="coerce"))
        .sort_values(["_date", KEY], kind="mergesort")
    )
    if ordered["_date"].isna().any():
        raise Phase15InputError("TEST claim dates contain null or invalid values.")
    n = len(ordered)
    labels = ("EARLY", "MIDDLE", "LATE")
    return {
        "slice_id": "temporal_chronological_thirds",
        "kind": "chronological_thirds",
        "column": "claim__claim_date",
        "membership_keys": {
            label: [
                int(value) for value in ordered.iloc[(n * index) // 3 : (n * (index + 1)) // 3][KEY]
            ]
            for index, label in enumerate(labels)
        },
        "source": "TEST_FEATURES_AND_DATE_ONLY",
        "target_independent": True,
    }


def frozen_test_slice_definitions(resolved: Phase15Resolved) -> list[dict[str, Any]]:
    """Reuse Phase 14 TRAIN rules, but rebuild TEST chronology memberships."""

    result: list[dict[str, Any]] = []
    for original in resolved.phase14_plan.get("slice_definitions", []):
        if not isinstance(original, dict):
            continue
        definition = dict(original)
        if definition.get("kind") == "chronological_thirds":
            definition = _chronological_definition(resolved.test_features)
        column = definition.get("column")
        if column is not None and str(column) not in resolved.test_features.columns:
            raise Phase15InputError(f"Frozen TEST slice column is missing: {column}")
        definition["target_independent"] = True
        definition["source_phase14_definition_sha256"] = canonical_json_sha256(original)
        result.append(definition)
    if not any(item.get("kind") == "chronological_thirds" for item in result):
        result.append(_chronological_definition(resolved.test_features))
    return result


def test_slice_memberships(
    definitions: list[dict[str, Any]],
    frame: pd.DataFrame,
    probabilities: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for definition in definitions:
        membership = membership_for_definition(definition, frame, scores=probabilities)
        for key, label in zip(frame[KEY].astype(int), membership.astype(str), strict=True):
            rows.append(
                {
                    "slice_id": str(definition["slice_id"]),
                    "slice_label": str(label),
                    KEY: int(key),
                }
            )
    return (
        pd.DataFrame(rows, columns=["slice_id", "slice_label", KEY])
        .sort_values(["slice_id", "slice_label", KEY], kind="mergesort")
        .reset_index(drop=True)
    )


def build_evaluation_plan(
    resolved: Phase15Resolved,
    settings: Phase15Settings,
    membership: dict[str, Any],
    definitions: list[dict[str, Any]],
    execution: dict[str, Any],
) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "phase": 15,
        "phase14_run_id": resolved.phase14_manifest["run_id"],
        "phase13_run_id": resolved.phase13.phase13_manifest["run_id"],
        "final_scoring_policy": "REUSE_FROZEN_PHASE14_CHAMPION",
        "scoring_policy_count": 1,
        "candidate_type": resolved.champion_type,
        "champion_id": resolved.champion_id,
        "effective_score_space": resolved.score_space,
        "frozen_technical_threshold": resolved.threshold,
        "metric_list": [
            "average_precision",
            "roc_auc",
            "log_loss",
            "brier_score",
            "ece_10",
            "mce_10",
            "ap_lift_over_prevalence",
        ],
        "threshold_metric_list": [
            "tp",
            "fp",
            "tn",
            "fn",
            "precision",
            "recall",
            "specificity",
            "negative_predictive_value",
            "false_positive_rate",
            "false_negative_rate",
            "f1",
            "f2",
            "balanced_accuracy",
            "mcc",
            "predicted_positive_rate",
        ],
        "ranking_policy": {
            "sort": ["probability DESC", f"{KEY} ASC"],
            "top_k": list(settings.top_k),
            "risk_deciles": "TRAIN_OOF_FROZEN_D10_HIGH_TO_D1_LOW",
        },
        "bootstrap_policy": {
            "seed": settings.seed,
            "replicates": execution["test_bootstrap_replicates"],
            "confidence_level": settings.confidence_level,
            "method": "STRATIFIED_PERCENTILE",
        },
        "temporal_policy": "PHASE14_RULES_WITH_TEST_DATE_ONLY_CHRONOLOGICAL_THIRDS",
        "slice_policy": "PHASE14_TRAIN_DERIVED_DEFINITIONS_REAPPLIED_TO_TEST",
        "generalization_policy": {
            "moderate_ap_ratio": settings.moderate_ap_ratio,
            "moderate_roc_drop": settings.moderate_roc_drop,
            "random_roc": settings.random_roc,
        },
        "invariance_policy": {
            "tolerance": settings.probability_tolerance,
            "batch_sizes": list(settings.batch_sizes),
        },
        "compute_policy": execution,
        "test_membership": membership,
        "slice_definition_sha256": canonical_json_sha256(definitions),
        "test_targets_accessed": False,
        "test_predictions_created": False,
        "test_metrics_computed": False,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "configuration_sha256": configuration_sha256(),
    }
    stable = {
        key: value
        for key, value in plan.items()
        if key not in {"created_at_utc"} and not key.endswith("_sha256")
    }
    plan["evaluation_plan_sha256"] = canonical_json_sha256(stable)
    return plan


def build_plan_inputs(
    resolved: Phase15Resolved,
    settings: Phase15Settings,
    execution: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    assignments = (
        resolved.phase13.phase12_lock.phase12_inputs.phase10_inputs.phase9_inputs.assignments
    )
    membership = build_test_membership_audit(
        assignments, resolved.phase6_manifest, resolved.test_lock
    )
    definitions = frozen_test_slice_definitions(resolved)
    # Risk-score memberships require frozen scores and are therefore created
    # after the persisted pre-TEST freeze, immediately before target loading.
    # Feature/date/categorical memberships can be materialized during planning.
    non_score_definitions = [
        item
        for item in definitions
        if item.get("kind") not in {"risk_score_band", "risk_score_decile"}
    ]
    memberships = (
        test_slice_memberships(non_score_definitions, resolved.test_features)
        if non_score_definitions
        else pd.DataFrame(columns=["slice_id", "slice_label", KEY])
    )
    plan = build_evaluation_plan(resolved, settings, membership, definitions, execution)
    return plan, definitions, memberships


__all__ = [
    "build_evaluation_plan",
    "build_plan_inputs",
    "frozen_test_slice_definitions",
    "test_slice_memberships",
]
