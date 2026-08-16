"""Fit the two frozen Phase 10 finalists and create outer VALIDATION predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..baseline_model.catboost_baseline import effective_parameters, save_model
from ..baseline_model.config import load_baseline_settings
from ..baseline_model.models import FeatureSetSpec
from ..feature_mart.manifest import sha256_file
from .config import TRACK_TO_EXPERIMENT
from .input import KEY, TARGET
from .metrics import metrics_for_predictions
from .models import OptimizationError, Phase10Inputs, StudyResult
from .objective import build_model_parameters, fit_model
from .search_space import parameter_sha256


def _ordered_target(targets: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        frame[[KEY]]
        .merge(targets, on=KEY, how="left", validate="one_to_one")[TARGET]
        .to_numpy(dtype="int8"),
        dtype="int8",
    )


def _predict(
    model: Any,
    frame: pd.DataFrame,
    feature_set: FeatureSetSpec,
    project_root: Path,
) -> np.ndarray:
    from ..baseline_model.adapters import adapt_matrix
    from ..baseline_model.catboost_baseline import build_pool

    settings = load_baseline_settings(project_root)
    matrix = adapt_matrix(frame.drop(columns=[KEY]), feature_set, settings)
    return np.asarray(model.predict_proba(build_pool(matrix, feature_set))[:, 1], dtype="float64")


def fit_phase10_finalists(
    phase10_inputs: Phase10Inputs,
    train_targets: pd.DataFrame,
    validation_targets: pd.DataFrame,
    studies: dict[str, StudyResult],
    fixed_parameters: dict[str, Any],
    model_directory: Path,
    *,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fit T1/T3 on all outer TRAIN rows and score only outer VALIDATION."""

    model_directory.mkdir(parents=True, exist_ok=True)
    development = phase10_inputs.development
    train_frame = (
        development.loc[development["split"] == "TRAIN"].sort_values(KEY).reset_index(drop=True)
    )
    validation_frame = (
        development.loc[development["split"] == "VALIDATION"]
        .sort_values(KEY)
        .reset_index(drop=True)
    )
    y_train = _ordered_target(train_targets, train_frame)
    y_validation = _ordered_target(validation_targets, validation_frame)
    prediction_parts: list[pd.DataFrame] = []
    model_manifest: dict[str, Any] = {"models": {}}
    metrics: dict[str, dict[str, Any]] = {}
    for track in ("T1", "T3"):
        study = studies[track]
        experiment_id = TRACK_TO_EXPERIMENT[track]
        feature_set = phase10_inputs.feature_sets[experiment_id]
        model_parameters = build_model_parameters(fixed_parameters, study.best_params)
        model, training_seconds = fit_model(
            train_frame.drop(columns=["split"]),
            y_train,
            feature_set,
            model_parameters,
            project_root=phase10_inputs.root,
        )
        filename = f"{track.lower()}_{experiment_id.lower()}_optimized.cbm"
        model_path = model_directory / filename
        save_model(model, model_path)
        probabilities = _predict(
            model, validation_frame.drop(columns=["split"]), feature_set, phase10_inputs.root
        )
        candidate_id = f"P10_{track}_{experiment_id}_OPTIMIZED"
        prediction_parts.append(
            pd.DataFrame(
                {
                    KEY: validation_frame[KEY].to_numpy(dtype="int64"),
                    "candidate_id": candidate_id,
                    "high_cost_probability": probabilities,
                }
            )
        )
        metric = metrics_for_predictions(y_validation, probabilities, threshold)
        metrics[candidate_id] = metric
        model_manifest["models"][candidate_id] = {
            "track": track,
            "phase9_experiment_id": experiment_id,
            "model_file": f"models/{filename}",
            "model_sha256": sha256_file(model_path),
            "feature_count": feature_set.feature_count,
            "feature_set_sha256": feature_set.feature_set_sha256,
            "best_trial_number": study.best_trial_number,
            "best_params": study.best_params,
            "parameter_sha256": parameter_sha256(study.best_params),
            "model_parameters": model_parameters,
            "effective_parameters": effective_parameters(model),
            "training_seconds": training_seconds,
            "validation_rows": int(len(validation_frame)),
        }
    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        .sort_values(["candidate_id", KEY], kind="mergesort")
        .reset_index(drop=True)
    )
    if len(predictions) != 2 * len(validation_frame):
        raise OptimizationError(
            "Phase 10 finalist prediction population is not exactly two candidates."
        )
    baseline_predictions = pd.read_parquet(
        phase10_inputs.phase9_dir / "validation_predictions.parquet"
    )
    if set(baseline_predictions["experiment_id"].astype(str)) & {"E1", "E3"} != {"E1", "E3"}:
        raise OptimizationError("Phase 9 E1/E3 baseline predictions are missing.")
    baseline_metrics: dict[str, dict[str, Any]] = {}
    for experiment_id, _track in (("E1", "T1"), ("E3", "T3")):
        baseline = baseline_predictions.loc[baseline_predictions["experiment_id"] == experiment_id]
        baseline = validation_frame[[KEY]].merge(
            baseline[[KEY, "probability"]], on=KEY, validate="one_to_one"
        )
        baseline_metrics[f"P9_{experiment_id}_BASELINE"] = metrics_for_predictions(
            y_validation,
            baseline["probability"].to_numpy(dtype="float64"),
            threshold,
        )
    comparisons = {
        "T1": _comparison(baseline_metrics["P9_E1_BASELINE"], metrics["P10_T1_E1_OPTIMIZED"]),
        "T3": _comparison(baseline_metrics["P9_E3_BASELINE"], metrics["P10_T3_E3_OPTIMIZED"]),
    }
    return predictions, metrics, baseline_metrics, {**model_manifest, "comparisons": comparisons}


def _comparison(baseline: dict[str, Any], optimized: dict[str, Any]) -> dict[str, Any]:
    baseline_ap = float(baseline["average_precision"])
    optimized_ap = float(optimized["average_precision"])
    return {
        "baseline_average_precision": baseline_ap,
        "optimized_average_precision": optimized_ap,
        "average_precision_delta": optimized_ap - baseline_ap,
        "average_precision_relative_lift": (
            (optimized_ap - baseline_ap) / baseline_ap if baseline_ap else None
        ),
        "roc_auc_delta": float(optimized["roc_auc"]) - float(baseline["roc_auc"]),
        "log_loss_delta": float(optimized["log_loss"]) - float(baseline["log_loss"]),
        "brier_score_delta": float(optimized["brier_score"]) - float(baseline["brier_score"]),
        "optimized_beats_baseline": optimized_ap > baseline_ap + 1.0e-6,
        "fallback_to_baseline": optimized_ap <= baseline_ap + 1.0e-6,
    }
