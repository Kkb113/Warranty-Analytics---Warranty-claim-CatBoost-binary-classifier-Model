"""TRAIN-only CatBoost fold objective and deterministic baseline replay."""

from __future__ import annotations

import gc
import time
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from ..baseline_model.adapters import adapt_matrix
from ..baseline_model.catboost_baseline import build_pool
from ..baseline_model.config import load_baseline_settings
from ..baseline_model.models import BaselineSettings, FeatureSetSpec
from .inner_folds import KEY
from .metrics import aggregate_fold_metrics, metrics_for_predictions
from .models import InnerFoldPlan, OptimizationError, TrialEvaluation
from .search_space import validate_trial_parameters

TARGET = "target__high_cost_claim_flag"


def _model_settings(parameters: dict[str, Any], project_root: Any = None) -> BaselineSettings:
    baseline = load_baseline_settings(project_root)
    return replace(baseline, catboost_parameters=parameters)


def build_model_parameters(
    fixed_parameters: dict[str, Any], search_parameters: dict[str, Any]
) -> dict[str, Any]:
    trial_parameters = validate_trial_parameters(search_parameters)
    if set(fixed_parameters) & set(trial_parameters):
        raise OptimizationError("Fixed and search CatBoost parameters overlap.")
    return {**fixed_parameters, **trial_parameters}


def fit_model(
    matrix: pd.DataFrame,
    target: np.ndarray,
    feature_set: FeatureSetSpec,
    parameters: dict[str, Any],
    *,
    project_root: Any = None,
) -> tuple[CatBoostClassifier, float]:
    if target.ndim != 1 or len(matrix) != len(target):
        raise OptimizationError("CatBoost fit matrix and target lengths differ.")
    settings = _model_settings(parameters, project_root)
    adapted = adapt_matrix(matrix, feature_set, settings)
    model = CatBoostClassifier(**parameters)
    started = time.perf_counter()
    model.fit(build_pool(adapted, feature_set, target))
    return model, time.perf_counter() - started


def evaluate_parameters(
    train_matrix: pd.DataFrame,
    train_targets: pd.DataFrame,
    feature_set: FeatureSetSpec,
    folds: InnerFoldPlan,
    fixed_parameters: dict[str, Any],
    search_parameters: dict[str, Any],
    *,
    threshold: float = 0.5,
    project_root: Any = None,
) -> TrialEvaluation:
    """Fit and score one parameter set on every TRAIN-only inner fold."""

    parameters = validate_trial_parameters(search_parameters)
    model_parameters = build_model_parameters(fixed_parameters, parameters)
    target_by_key = train_targets.set_index(KEY)[TARGET].astype("int8")
    keyed_matrix = train_matrix.copy()
    keyed_matrix[KEY] = keyed_matrix[KEY].astype(int)
    fold_metrics: list[dict[str, Any]] = []
    total_training_seconds = 0.0
    for fold in folds.folds:
        train_frame = keyed_matrix[keyed_matrix[KEY].isin(fold.train_keys)].sort_values(KEY)
        validation_frame = keyed_matrix[keyed_matrix[KEY].isin(fold.validation_keys)].sort_values(
            KEY
        )
        if len(train_frame) != fold.train_rows or len(validation_frame) != fold.validation_rows:
            raise OptimizationError(f"Inner fold {fold.fold_id} matrix membership changed.")
        y_train = target_by_key.loc[train_frame[KEY].tolist()].to_numpy(dtype="int8")
        y_validation = target_by_key.loc[validation_frame[KEY].tolist()].to_numpy(dtype="int8")
        model, elapsed = fit_model(
            train_frame.drop(columns=[KEY]),
            y_train,
            feature_set,
            model_parameters,
            project_root=project_root,
        )
        total_training_seconds += elapsed
        validation_settings = _model_settings(model_parameters, project_root)
        validation_matrix = adapt_matrix(
            validation_frame.drop(columns=[KEY]), feature_set, validation_settings
        )
        probabilities = np.asarray(
            model.predict_proba(build_pool(validation_matrix, feature_set))[:, 1], dtype="float64"
        )
        metrics = metrics_for_predictions(y_validation, probabilities, threshold)
        fold_metrics.append(
            {
                "fold_id": fold.fold_id,
                "average_precision": metrics["average_precision"],
                "roc_auc": metrics["roc_auc"],
                "log_loss": metrics["log_loss"],
                "brier_score": metrics["brier_score"],
                "train_rows": fold.train_rows,
                "validation_rows": fold.validation_rows,
            }
        )
        del model
        gc.collect()
    return TrialEvaluation(
        params=parameters,
        fold_metrics=tuple(fold_metrics),
        aggregate=aggregate_fold_metrics(fold_metrics),
        training_seconds=total_training_seconds,
    )


def baseline_search_parameters(project_root: Any = None) -> dict[str, Any]:
    """Materialize the locked Phase 9 CatBoost settings in the Phase 10 schema."""

    settings = load_baseline_settings(project_root)
    raw = settings.catboost_parameters
    return validate_trial_parameters(
        {
            "iterations": int(raw["iterations"]),
            "learning_rate": float(raw["learning_rate"]),
            "depth": int(raw["depth"]),
            "l2_leaf_reg": float(raw["l2_leaf_reg"]),
            "random_strength": float(raw["random_strength"]),
            "bagging_temperature": float(raw["bagging_temperature"]),
            "border_count": int(raw.get("border_count", 254)),
            "rsm": float(raw.get("rsm", 1.0)),
        }
    )
