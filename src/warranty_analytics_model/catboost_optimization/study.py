"""Independent deterministic Optuna studies for T1 and T3."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .config import TRACK_TO_EXPERIMENT
from .models import InnerFoldPlan, OptimizationError, StudyResult
from .objective import evaluate_parameters
from .provenance import canonical_json_sha256
from .search_space import parameter_sha256, suggest_trial_parameters
from .selection import select_best_trial

TRIAL_HISTORY_COLUMNS = (
    "track",
    "trial_number",
    "state",
    "iterations",
    "learning_rate",
    "depth",
    "l2_leaf_reg",
    "random_strength",
    "bagging_temperature",
    "border_count",
    "rsm",
    "mean_average_precision",
    "min_average_precision",
    "max_average_precision",
    "std_average_precision",
    "mean_roc_auc",
    "min_roc_auc",
    "mean_log_loss",
    "mean_brier_score",
    "fold_count",
    "training_seconds",
)


def require_trial_history_schema(history: pd.DataFrame) -> None:
    """Fail before finalist fitting when persisted trial history would be invalid."""

    actual = tuple(str(column) for column in history.columns)
    if actual != TRIAL_HISTORY_COLUMNS:
        raise OptimizationError(
            "Phase 10 trial_history schema differs before publication: "
            f"expected {TRIAL_HISTORY_COLUMNS}, got {actual}."
        )


def _history_row(
    track: str,
    trial: Any,
    evaluation: Any | None,
) -> dict[str, Any]:
    params = {
        name: trial.params.get(name)
        for name in (
            "iterations",
            "learning_rate",
            "depth",
            "l2_leaf_reg",
            "random_strength",
            "bagging_temperature",
            "border_count",
            "rsm",
        )
    }
    row: dict[str, Any] = {
        "track": track,
        "trial_number": int(trial.number),
        "state": str(trial.state.name),
        **params,
        "mean_average_precision": None,
        "min_average_precision": None,
        "max_average_precision": None,
        "std_average_precision": None,
        "mean_roc_auc": None,
        "min_roc_auc": None,
        "mean_log_loss": None,
        "mean_brier_score": None,
        "fold_count": None,
        "training_seconds": None,
    }
    if evaluation is not None:
        row.update(evaluation.aggregate)
        row["training_seconds"] = evaluation.training_seconds
    return row


def run_track_study(
    track: str,
    train_matrix: pd.DataFrame,
    train_targets: pd.DataFrame,
    feature_set: Any,
    fold_plan: InnerFoldPlan,
    fixed_parameters: dict[str, Any],
    *,
    trials: int,
    seed: int,
    n_startup_trials: int,
    threshold: float,
    project_root: Any = None,
    baseline_inner_cv_metrics: dict[str, Any] | None = None,
) -> StudyResult:
    """Run one independent sequential study and retain all trial outcomes."""

    if track not in TRACK_TO_EXPERIMENT:
        raise OptimizationError(f"Unsupported Phase 10 track: {track}")
    try:
        import optuna
    except ImportError as exc:
        raise OptimizationError(
            "Optuna is required for phase10-optimize; install the optimization extra."
        ) from exc
    study_name = (
        f"phase10_{track}_{TRACK_TO_EXPERIMENT[track].lower()}_"
        f"{('core' if track == 'T1' else 'lexical')}"
    )
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
    )
    evaluations: dict[int, Any] = {}

    def objective(trial: Any) -> float:
        params = suggest_trial_parameters(trial)
        try:
            evaluation = evaluate_parameters(
                train_matrix,
                train_targets,
                feature_set,
                fold_plan,
                fixed_parameters,
                params,
                threshold=threshold,
                project_root=project_root,
            )
        except Exception as exc:
            trial.set_user_attr("failure_message", str(exc))
            raise
        evaluations[trial.number] = evaluation
        return float(evaluation.aggregate["mean_average_precision"])

    study.optimize(
        objective,
        n_trials=trials,
        n_jobs=1,
        catch=(Exception,),
        show_progress_bar=False,
    )
    history_rows = [
        _history_row(track, trial, evaluations.get(trial.number)) for trial in study.trials
    ]
    history = pd.DataFrame(history_rows, columns=TRIAL_HISTORY_COLUMNS)
    completed = int((history["state"] == "COMPLETE").sum())
    if completed < max(1, math.ceil(trials * 0.80)):
        raise OptimizationError(
            f"Phase 10 {track} completed only {completed}/{trials} trials; at least 80% is required."
        )
    if trials >= 50 and completed < 40:
        raise OptimizationError(f"Phase 10 {track} must complete at least 40 trials.")
    warnings: list[str] = []
    failures = trials - completed
    if trials and failures / trials > 0.10:
        warnings.append(f"{track}_TRIAL_FAILURE_RATE_ABOVE_10_PERCENT")
    best = select_best_trial(history)
    best_number = int(best["trial_number"])
    best_evaluation = evaluations.get(best_number)
    if best_evaluation is None:
        raise OptimizationError(f"Phase 10 {track} best trial lacks evaluation metrics.")
    fold_rows: list[dict[str, Any]] = []
    for trial_number, evaluation in evaluations.items():
        for item in evaluation.fold_metrics:
            fold_rows.append({"track": track, "trial_number": trial_number, **item})
    fold_metrics = pd.DataFrame(
        fold_rows,
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
        ],
    )
    return StudyResult(
        track=track,
        phase9_experiment_id=TRACK_TO_EXPERIMENT[track],
        study_name=study_name,
        trial_history=history,
        fold_metrics=fold_metrics,
        baseline_inner_cv_metrics=baseline_inner_cv_metrics or {},
        best_trial_number=best_number,
        best_params={
            name: best[name]
            for name in (
                "iterations",
                "learning_rate",
                "depth",
                "l2_leaf_reg",
                "random_strength",
                "bagging_temperature",
                "border_count",
                "rsm",
            )
        },
        best_inner_metrics={
            key: best[key]
            for key in (
                "mean_average_precision",
                "min_average_precision",
                "std_average_precision",
                "mean_roc_auc",
                "min_roc_auc",
                "mean_log_loss",
                "mean_brier_score",
            )
        },
        best_param_sha256=parameter_sha256(
            {
                name: best[name]
                for name in (
                    "iterations",
                    "learning_rate",
                    "depth",
                    "l2_leaf_reg",
                    "random_strength",
                    "bagging_temperature",
                    "border_count",
                    "rsm",
                )
            }
        ),
        warnings=warnings,
    )


def study_history_sha256(history: pd.DataFrame) -> str:
    """Hash canonical aggregate trial history, including failures but no claims/targets."""

    canonical = history.sort_values(["track", "trial_number"], kind="mergesort").to_dict(
        orient="records"
    )
    return canonical_json_sha256(canonical)
