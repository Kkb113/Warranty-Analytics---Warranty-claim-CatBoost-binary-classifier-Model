"""Phase 12 two-stage optimization runner and immutable publication."""

from __future__ import annotations

import gc
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from ..baseline_model.adapters import adapt_matrix
from ..baseline_model.catboost_baseline import (
    build_pool,
    effective_parameters,
    load_model,
    save_model,
)
from ..baseline_model.config import load_baseline_settings
from ..catboost_optimization.provenance import canonical_json_sha256, sha256_file
from ..feature_mart.manifest import git_commit_sha, write_json, write_parquet
from ..paths import discover_repository_root
from .checkpoint import load_valid_checkpoint, write_checkpoint
from .config import (
    PHASE12_VERSION,
    TRACKS,
    ImbalanceThresholdError,
    load_imbalance_threshold_settings,
)
from .contract import validate_imbalance_threshold_contract
from .input import Phase12Inputs, load_locked_phase11_inputs, write_parent_resolution
from .metrics import (
    aggregate_strategy_metrics,
    fold_metric_row,
    ranking_metrics,
    strategy_fold_metrics_frame,
    threshold_metrics,
)
from .planner import build_compute_plan
from .selection import replacement_decision, select_phase12_champion, select_strategy
from .strategies import (
    StrategyDefinition,
    build_strategy_definitions,
    strategy_parameter_payload,
    strategy_parameters,
    validate_strategy_parameters,
)
from .thresholds import build_threshold_curve, threshold_summary

TARGET = "target__high_cost_claim_flag"
KEY = "warranty_claim_key"


def phase12_run_id() -> str:  # pragma: no cover
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:  # pragma: no cover
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ImbalanceThresholdError(f"Expected a JSON object: {path}")
    return payload


def _phase12_root(root: Path, output_dir: Path | None) -> Path:  # pragma: no cover
    value = output_dir or (root / "artifacts" / "imbalance_threshold")
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _work_dirs(output_root: Path, run_id: str) -> tuple[Path, Path]:  # pragma: no cover
    return output_root / f".phase12_{run_id}.work", output_root / run_id


def _target_by_key(targets: pd.DataFrame) -> pd.Series:  # pragma: no cover
    indexed = targets.set_index(KEY)[TARGET]
    if indexed.index.duplicated().any():
        raise ImbalanceThresholdError("Target keys are duplicated.")
    return indexed


def _fold_membership_hash(fold: Any) -> str:  # pragma: no cover
    return canonical_json_sha256(
        {
            "fold_id": int(fold.fold_id),
            "train": list(fold.train_keys),
            "validation": list(fold.validation_keys),
        }
    )


def _parameter_hash(parameters: dict[str, Any]) -> str:  # pragma: no cover
    return canonical_json_sha256(parameters)


def _base_parameters(parent: Any, threads: int) -> dict[str, Any]:  # pragma: no cover
    params = dict(parent.statistical_parameters)
    params["thread_count"] = int(threads)
    params["allow_writing_files"] = False
    params["verbose"] = False
    params["use_best_model"] = False
    return params


def _fit_fold(  # pragma: no cover
    inputs: Phase12Inputs,
    parent: Any,
    strategy: StrategyDefinition,
    fold: Any,
    train_targets: pd.DataFrame,
    threads: int,
) -> tuple[dict[str, Any], pd.DataFrame, str]:
    matrix = inputs.development
    train = matrix.loc[matrix[KEY].isin(fold.train_keys)].sort_values(KEY, kind="mergesort")
    validation = matrix.loc[matrix[KEY].isin(fold.validation_keys)].sort_values(
        KEY, kind="mergesort"
    )
    if len(train) != fold.train_rows or len(validation) != fold.validation_rows:
        raise ImbalanceThresholdError(f"Inner fold {fold.fold_id} membership changed.")
    target_by_key = _target_by_key(train_targets)
    y_train = target_by_key.loc[train[KEY].tolist()].to_numpy(dtype="int8")
    y_validation = target_by_key.loc[validation[KEY].tolist()].to_numpy(dtype="int8")
    base = _base_parameters(parent, threads)
    parameters = strategy_parameters(base, strategy)
    validate_strategy_parameters(parameters, strategy, base)
    settings = load_baseline_settings(inputs.root)
    adapted_train = adapt_matrix(
        train.drop(columns=[KEY]),
        parent.feature_set,
        replace(settings, catboost_parameters=parameters),
    )
    adapted_validation = adapt_matrix(
        validation.drop(columns=[KEY]),
        parent.feature_set,
        replace(settings, catboost_parameters=parameters),
    )
    model = CatBoostClassifier(**parameters)
    started = time.perf_counter()
    model.fit(build_pool(adapted_train, parent.feature_set, y_train))
    elapsed = time.perf_counter() - started
    probabilities = np.asarray(
        model.predict_proba(build_pool(adapted_validation, parent.feature_set))[:, 1],
        dtype="float64",
    )
    weighting_payload = strategy_parameter_payload(strategy)
    metrics = fold_metric_row(
        y_validation,
        probabilities,
        track=parent.track,
        strategy_id=strategy.strategy_id,
        fold_id=int(fold.fold_id),
        train_positive_count=int(y_train.sum()),
        train_negative_count=int((y_train == 0).sum()),
        training_seconds=elapsed,
        weighting_parameters=weighting_payload,
    )
    prediction = pd.DataFrame(
        {
            KEY: validation[KEY].astype("int64").to_numpy(),
            "track": parent.track,
            "strategy_id": strategy.strategy_id,
            "fold_id": int(fold.fold_id),
            "high_cost_probability": probabilities,
        }
    )
    prediction_hash = canonical_json_sha256(
        {
            "keys": [int(value) for value in prediction[KEY].tolist()],
            "probabilities": [format(float(value), ".17g") for value in probabilities],
        }
    )
    return metrics, prediction, prediction_hash


def _fit_fold_from_checkpoint(  # pragma: no cover
    inputs: Phase12Inputs,
    parent: Any,
    strategy: StrategyDefinition,
    fold: Any,
    train_targets: pd.DataFrame,
    threads: int,
    work_dir: Path,
    resume: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    expected_params = strategy_parameters(_base_parameters(parent, threads), strategy)
    expected_strategy_hash = _parameter_hash(expected_params)
    checkpoint = None
    if resume:
        checkpoint = load_valid_checkpoint(
            work_dir,
            track=parent.track,
            strategy_id=strategy.strategy_id,
            fold_id=fold.fold_id,
            feature_set_sha256=parent.feature_set.feature_set_sha256,
            parent_parameter_sha256=parent.parent_parameter_sha256,
            strategy_parameter_sha256=expected_strategy_hash,
            fold_membership_sha256=_fold_membership_hash(fold),
        )
    if (
        checkpoint is not None
        and "prediction_keys" in checkpoint
        and "prediction_values" in checkpoint
    ):
        prediction = pd.DataFrame(
            {
                KEY: [int(value) for value in checkpoint["prediction_keys"]],
                "track": parent.track,
                "strategy_id": strategy.strategy_id,
                "fold_id": int(fold.fold_id),
                "high_cost_probability": [
                    float(value) for value in checkpoint["prediction_values"]
                ],
            }
        )
        return dict(checkpoint["metrics"]), prediction
    metrics, prediction, prediction_hash = _fit_fold(
        inputs, parent, strategy, fold, train_targets, threads
    )
    write_checkpoint(
        work_dir,
        track=parent.track,
        strategy_id=strategy.strategy_id,
        fold_id=fold.fold_id,
        feature_set_sha256=parent.feature_set.feature_set_sha256,
        parent_parameter_sha256=parent.parent_parameter_sha256,
        strategy_parameter_sha256=expected_strategy_hash,
        fold_membership_sha256=_fold_membership_hash(fold),
        metrics=metrics,
        prediction_sha256=prediction_hash,
        training_seconds=float(metrics["training_seconds"]),
        prediction_keys=[int(value) for value in prediction[KEY].tolist()],
        prediction_values=[float(value) for value in prediction["high_cost_probability"].tolist()],
    )
    return metrics, prediction


def _write_fold_summary(  # pragma: no cover
    work_dir: Path,
    fold_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    oof: pd.DataFrame,
) -> None:
    # JSON-encode nested weighting parameters for a portable, deterministic Parquet schema.
    fold_frame = strategy_fold_metrics_frame(fold_rows)
    fold_frame["weighting_parameters"] = fold_frame["weighting_parameters"].map(
        lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
    )
    write_parquet(fold_frame, work_dir / "strategy_fold_metrics.parquet", compression="zstd")
    summary_frame = (
        pd.DataFrame(summary_rows)
        .sort_values(["track", "strategy_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    write_parquet(summary_frame, work_dir / "strategy_summary.parquet", compression="zstd")
    oof = (
        oof.loc[:, [KEY, "track", "strategy_id", "fold_id", "high_cost_probability"]]
        .sort_values(["track", "strategy_id", "fold_id", KEY], kind="mergesort")
        .reset_index(drop=True)
    )
    if oof.duplicated([KEY, "track", "strategy_id"]).any():
        raise ImbalanceThresholdError("Phase 12 OOF predictions contain duplicates.")
    write_parquet(oof, work_dir / "strategy_oof_predictions.parquet", compression="zstd")


def _write_freeze(  # pragma: no cover
    work_dir: Path,
    inputs: Phase12Inputs,
    strategies_by_track: dict[str, tuple[StrategyDefinition, ...]],
    selections: dict[str, dict[str, Any]],
    summary_frame: pd.DataFrame,
    threshold_frame: pd.DataFrame,
    threshold_summaries: dict[str, dict[str, dict[str, Any]]],
    oof: pd.DataFrame,
) -> tuple[dict[str, Any], str]:
    tracks: dict[str, Any] = {}
    for track in TRACKS:
        parent = inputs.parents[track]
        selected_id = str(selections[track]["selected_strategy_id"])
        selected = next(
            item for item in strategies_by_track[track] if item.strategy_id == selected_id
        )
        track_oof = oof.loc[(oof["track"] == track) & (oof["strategy_id"] == selected_id)]
        oof_hash = canonical_json_sha256(
            {
                "rows": [
                    [int(row[KEY]), format(float(row["high_cost_probability"]), ".17g")]
                    for _, row in track_oof.sort_values(KEY, kind="mergesort").iterrows()
                ]
            }
        )
        track_summary = summary_frame.loc[
            (summary_frame["track"] == track) & (summary_frame["strategy_id"] == selected_id)
        ]
        track_curve = threshold_frame.loc[
            (threshold_frame["track"] == track) & (threshold_frame["strategy_id"] == selected_id)
        ]
        policy = threshold_summaries[track][selected_id]
        payload = {
            "phase11_effective_candidate_id": parent.effective_candidate_id,
            "feature_count": parent.feature_set.feature_count,
            "feature_set_sha256": parent.feature_set.feature_set_sha256,
            "feature_list_sha256": parent.feature_list_sha256,
            "frozen_parent_parameter_sha256": parent.parent_parameter_sha256,
            "selected_imbalance_strategy": selected_id,
            "selected_strategy_parameter_sha256": selected.parameter_sha256,
            "inner_cv_metrics": track_summary.iloc[0].to_dict() if not track_summary.empty else {},
            "oof_prediction_sha256": oof_hash,
            "technical_threshold": policy["technical_default"]["threshold"],
            "threshold_objective": "MCC",
            "threshold_metrics": policy["technical_default"]["metrics"],
            "alternative_thresholds": policy["alternatives"],
            "strategy_summary_sha256": canonical_json_sha256(track_summary.to_dict("records")),
            "threshold_curve_sha256": canonical_json_sha256(track_curve.to_dict("records")),
        }
        tracks[track] = payload
    freeze = {
        "phase": 12,
        "phase11_run_id": inputs.phase11_manifest.get("run_id"),
        "phase10_inner_fold_sha256": inputs.fold_plan.content_sha256,
        "tracks": tracks,
        "outer_validation_accessed": False,
        "test_target_accessed": False,
        "test_predictions_created": False,
        "test_metrics_computed": False,
    }
    freeze_hash = canonical_json_sha256(freeze)
    freeze["phase12_freeze_sha256"] = freeze_hash
    write_json(work_dir / "phase12_freeze.json", freeze)
    return freeze, freeze_hash


def _full_train_model(  # pragma: no cover
    inputs: Phase12Inputs,
    parent: Any,
    strategy: StrategyDefinition,
    train_targets: pd.DataFrame,
    threads: int,
    model_path: Path,
) -> CatBoostClassifier:
    train = inputs.development.loc[inputs.development["split"] == "TRAIN"].sort_values(
        KEY, kind="mergesort"
    )
    y = _target_by_key(train_targets).loc[train[KEY].tolist()].to_numpy(dtype="int8")
    base = _base_parameters(parent, threads)
    parameters = strategy_parameters(base, strategy)
    validate_strategy_parameters(parameters, strategy, base)
    settings = load_baseline_settings(inputs.root)
    adapted = adapt_matrix(
        train.drop(columns=[KEY]),
        parent.feature_set,
        replace(settings, catboost_parameters=parameters),
    )
    model = CatBoostClassifier(**parameters)
    model.fit(build_pool(adapted, parent.feature_set, y))
    save_model(model, model_path)
    return model


def _score_model(  # pragma: no cover
    model: CatBoostClassifier, frame: pd.DataFrame, parent: Any, settings: Any
) -> np.ndarray:
    adapted = adapt_matrix(frame.drop(columns=[KEY]), parent.feature_set, settings)
    return np.asarray(
        model.predict_proba(build_pool(adapted, parent.feature_set))[:, 1], dtype="float64"
    )


def _metric_at_threshold(  # pragma: no cover
    y: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    result = ranking_metrics(y, probabilities)
    result.update(threshold_metrics(y, probabilities, threshold))
    return result


def _build_threshold_artifacts(  # pragma: no cover
    work_dir: Path,
    inputs: Phase12Inputs,
    strategies_by_track: dict[str, tuple[StrategyDefinition, ...]],
    oof: pd.DataFrame,
    train_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, Any]]]]:
    target_by_key = _target_by_key(train_targets)
    curve_frames: list[pd.DataFrame] = []
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for track in TRACKS:
        summaries[track] = {}
        for strategy in strategies_by_track[track]:
            subset = oof.loc[
                (oof["track"] == track) & (oof["strategy_id"] == strategy.strategy_id)
            ].sort_values(KEY, kind="mergesort")
            y = target_by_key.loc[subset[KEY].tolist()].to_numpy(dtype="int8")
            curve = build_threshold_curve(
                y,
                subset["high_cost_probability"].to_numpy(dtype="float64"),
                track=track,
                strategy_id=strategy.strategy_id,
            )
            curve_frames.append(curve)
            summaries[track][strategy.strategy_id] = threshold_summary(curve)
    all_curves = (
        pd.concat(curve_frames, ignore_index=True)
        .sort_values(["track", "strategy_id", "threshold"], kind="mergesort")
        .reset_index(drop=True)
    )
    write_parquet(all_curves, work_dir / "threshold_curve.parquet", compression="zstd")
    write_json(work_dir / "threshold_summary.json", summaries)
    return all_curves, summaries


def _artifact_hashes(directory: Path) -> dict[str, str]:  # pragma: no cover
    hashes: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in {"phase12_manifest.json"}:
            hashes[str(path.relative_to(directory)).replace("\\", "/")] = sha256_file(path)
    return hashes


def phase12_contract_check(project_root: Path | None = None) -> dict[str, Any]:  # pragma: no cover
    return validate_imbalance_threshold_contract(project_root)


def phase12_plan_check(  # pragma: no cover
    phase11_dir: Path,
    *,
    max_workers: int | None = None,
    threads_per_fit: int | None = None,
    single_fit_threads: int | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = discover_repository_root(project_root)
    contract = validate_imbalance_threshold_contract(root)
    if not contract.get("valid"):
        return {"status": "BLOCKED", "valid": False, "errors": contract.get("errors", [])}
    try:
        settings = load_imbalance_threshold_settings(root)
        inputs = load_locked_phase11_inputs(phase11_dir, project_root=root)
        train_targets, audit = __import__(
            "warranty_analytics_model.catboost_optimization.input",
            fromlist=["load_train_targets_for_optimization"],
        ).load_train_targets_for_optimization(inputs.phase10_inputs)
        plan = build_compute_plan(
            settings,
            max_workers=max_workers,
            threads_per_fit=threads_per_fit,
            single_fit_threads=single_fit_threads,
        )
        strategies = build_strategy_definitions(
            int(train_targets[TARGET].sum()), int((train_targets[TARGET] == 0).sum())
        )
        return {
            "status": "PASS",
            "valid": True,
            "inputs": inputs,
            "compute_plan": plan,
            "strategy_definitions": strategies,
            "train_target_audit": audit,
            "inner_fold_plan": inputs.fold_plan,
        }
    except Exception as exc:
        return {"status": "BLOCKED", "valid": False, "errors": [str(exc)]}


def build_phase12(  # pragma: no cover
    phase11_dir: Path,
    *,
    output_dir: Path | None = None,
    report_dir: Path | None = None,
    run_id: str | None = None,
    resume: bool = False,
    max_workers: int | None = None,
    threads_per_fit: int | None = None,
    single_fit_threads: int | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    root = discover_repository_root(project_root)
    contract_check = validate_imbalance_threshold_contract(root)
    if not contract_check.get("valid"):
        raise ImbalanceThresholdError(
            "Phase 12 contract is not valid: " + "; ".join(contract_check.get("errors", []))
        )
    settings = load_imbalance_threshold_settings(root)
    inputs = load_locked_phase11_inputs(phase11_dir, project_root=root)
    compute = build_compute_plan(
        settings,
        max_workers=max_workers,
        threads_per_fit=threads_per_fit,
        single_fit_threads=single_fit_threads,
    )
    train_targets, train_audit = __import__(
        "warranty_analytics_model.catboost_optimization.input",
        fromlist=["load_train_targets_for_optimization"],
    ).load_train_targets_for_optimization(inputs.phase10_inputs)
    positive_count = int(train_targets[TARGET].sum())
    negative_count = int((train_targets[TARGET] == 0).sum())
    strategies = build_strategy_definitions(positive_count, negative_count)
    run = run_id or phase12_run_id()
    output_root = _phase12_root(root, output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir, final_dir = _work_dirs(output_root, run)
    if final_dir.exists():
        raise ImbalanceThresholdError(
            f"Phase 12 run is already published and immutable: {final_dir}"
        )
    if work_dir.exists() and not resume:
        raise ImbalanceThresholdError(f"Incomplete Phase 12 work exists; use --resume: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    write_parent_resolution(work_dir / "phase11_parent_resolution.json", inputs)
    strategy_payload = {
        "phase": 12,
        "train_positive_count": positive_count,
        "train_negative_count": negative_count,
        "strategies": [strategy.as_dict() for strategy in strategies],
    }
    write_json(work_dir / "strategy_definitions.json", strategy_payload)
    compute_payload = {"phase": 12, **compute.as_dict()}
    write_json(work_dir / "compute_manifest.json", compute_payload)
    strategies_by_track = {track: strategies for track in TRACKS}
    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    tasks = [
        (track, strategy, fold)
        for track in TRACKS
        for strategy in strategies
        for fold in inputs.fold_plan.folds
    ]
    with ThreadPoolExecutor(
        max_workers=compute.worker_count, thread_name_prefix="phase12"
    ) as executor:
        futures = {
            executor.submit(
                _fit_fold_from_checkpoint,
                inputs,
                inputs.parents[track],
                strategy,
                fold,
                train_targets,
                compute.threads_per_fit,
                work_dir,
                resume,
            ): (track, strategy.strategy_id, fold.fold_id)
            for track, strategy, fold in tasks
        }
        for future in as_completed(futures):
            metrics, prediction = future.result()
            fold_rows.append(metrics)
            prediction_frames.append(prediction)
    fold_rows.sort(
        key=lambda row: (str(row["track"]), str(row["strategy_id"]), int(row["fold_id"]))
    )
    oof = (
        pd.concat(prediction_frames, ignore_index=True)
        .sort_values(["track", "strategy_id", "fold_id", KEY], kind="mergesort")
        .reset_index(drop=True)
    )
    summary_rows: list[dict[str, Any]] = []
    for track in TRACKS:
        for strategy in strategies:
            rows = [
                row
                for row in fold_rows
                if row["track"] == track and row["strategy_id"] == strategy.strategy_id
            ]
            if len(rows) != 3:
                raise ImbalanceThresholdError(
                    f"{track}/{strategy.strategy_id} did not complete all three folds."
                )
            summary_rows.append(aggregate_strategy_metrics(rows))
    _write_fold_summary(work_dir, fold_rows, summary_rows, oof)
    summary_frame = (
        pd.DataFrame(summary_rows)
        .sort_values(["track", "strategy_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    threshold_frame, threshold_summaries = _build_threshold_artifacts(
        work_dir, inputs, strategies_by_track, oof, train_targets
    )
    selections: dict[str, dict[str, Any]] = {}
    for track in TRACKS:
        track_summary = summary_frame.loc[summary_frame["track"] == track].copy()
        selections[track] = select_strategy(
            track_summary,
            threshold_summaries[track],
            max_ap_tolerance=settings.max_ap_tolerance,
            max_min_ap_drop=settings.max_min_ap_drop,
            max_roc_auc_drop=settings.max_roc_auc_drop,
            prefer_none_mcc_tolerance=settings.prefer_none_mcc_tolerance,
        )
    freeze, freeze_hash = _write_freeze(
        work_dir,
        inputs,
        strategies_by_track,
        selections,
        summary_frame,
        threshold_frame,
        threshold_summaries,
        oof,
    )
    # Stage B begins only after the immutable Stage-A freeze is present.
    from ..catboost_optimization.input import load_validation_targets_after_freeze

    validation_targets, validation_audit = load_validation_targets_after_freeze(
        inputs.phase10_inputs, study_frozen=True
    )
    validation_by_key = _target_by_key(validation_targets)
    baseline_settings = load_baseline_settings(root)
    validation_rows: list[dict[str, Any]] = []
    effective_entries: list[dict[str, Any]] = []
    model_entries: list[dict[str, Any]] = []
    threshold_policy: dict[str, Any] = {}
    outer_candidates: list[dict[str, Any]] = []
    weighted_count = 0
    models_dir = work_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for track in TRACKS:
        parent = inputs.parents[track]
        selected_id = selections[track]["selected_strategy_id"]
        selected = next(item for item in strategies if item.strategy_id == selected_id)
        none_policy = threshold_summaries[track]["S0_NONE"]["technical_default"]
        selected_policy = threshold_summaries[track][selected_id]["technical_default"]
        validation_frame = inputs.development.loc[
            inputs.development["split"] == "VALIDATION"
        ].sort_values(KEY, kind="mergesort")
        parent_model = load_model(parent.source_model)
        parent_probabilities = _score_model(
            parent_model, validation_frame, parent, baseline_settings
        )
        parent_y = validation_by_key.loc[validation_frame[KEY].tolist()].to_numpy(dtype="int8")
        parent_validation_metrics = _metric_at_threshold(
            parent_y, parent_probabilities, float(none_policy["threshold"])
        )
        parent_candidate_id = parent.effective_candidate_id
        parent_copy_path = models_dir / f"{track.lower()}_phase11_parent.cbm"
        shutil.copy2(parent.source_model, parent_copy_path)
        parent_model_digest = sha256_file(parent_copy_path)
        parent_params = effective_parameters(parent_model)
        parent_model_entries = {
            "track": track,
            "candidate_id": parent_candidate_id,
            "source_candidate_id": parent_candidate_id,
            "source_phase": parent.source_phase,
            "feature_count": parent.feature_set.feature_count,
            "feature_set_sha256": parent.feature_set.feature_set_sha256,
            "feature_list_sha256": parent.feature_list_sha256,
            "model_file": str(parent_copy_path.relative_to(work_dir)).replace("\\", "/"),
            "model_sha256": parent_model_digest,
            "source_model_sha256": parent.source_model_sha256,
            "copied_model_sha256": parent_model_digest,
            "statistical_parameters": {
                key: value
                for key, value in parent_params.items()
                if key not in {"thread_count", "allow_writing_files", "verbose", "use_best_model"}
            },
            "imbalance_strategy": {
                "strategy_id": "S0_NONE",
                "strategy_type": "none",
                "parameter": None,
                "parameter_sha256": next(
                    item.parameter_sha256 for item in strategies if item.strategy_id == "S0_NONE"
                ),
            },
            "execution_parameters": {"thread_count": compute.single_fit_threads, "worker_count": 1},
            "train_rows": int(len(train_targets)),
            "validation_rows": int(len(validation_targets)),
            "validation_metrics": parent_validation_metrics,
        }
        model_entries.append(parent_model_entries)
        validation_rows.extend(
            {
                KEY: int(key),
                "track": track,
                "candidate_id": parent_candidate_id,
                "high_cost_probability": float(probability),
            }
            for key, probability in zip(
                validation_frame[KEY].tolist(), parent_probabilities, strict=True
            )
        )
        weighted_entry: dict[str, Any] | None = None
        weighted_metrics: dict[str, Any] | None = None
        weighted_model: CatBoostClassifier | None = None
        if selected.weighted:
            weighted_count += 1
            if weighted_count > 2:
                raise ImbalanceThresholdError(
                    "Phase 12 weighted validation candidate cap exceeded."
                )
            weighted_path = models_dir / f"{track.lower()}_{selected.strategy_id.lower()}.cbm"
            weighted_model = _full_train_model(
                inputs, parent, selected, train_targets, compute.single_fit_threads, weighted_path
            )
            weighted_probabilities = _score_model(
                weighted_model, validation_frame, parent, baseline_settings
            )
            weighted_metrics = _metric_at_threshold(
                parent_y, weighted_probabilities, float(selected_policy["threshold"])
            )
            weighted_entry = {
                "track": track,
                "candidate_id": f"P12_{track}_{selected.strategy_id}",
                "source_candidate_id": parent_candidate_id,
                "source_phase": 12,
                "feature_count": parent.feature_set.feature_count,
                "feature_set_sha256": parent.feature_set.feature_set_sha256,
                "feature_list_sha256": parent.feature_list_sha256,
                "model_file": str(weighted_path.relative_to(work_dir)).replace("\\", "/"),
                "model_sha256": sha256_file(weighted_path),
                "statistical_parameters": {
                    key: value
                    for key, value in effective_parameters(weighted_model).items()
                    if key
                    not in {"thread_count", "allow_writing_files", "verbose", "use_best_model"}
                },
                "imbalance_strategy": selected.as_dict(),
                "execution_parameters": {
                    "thread_count": compute.single_fit_threads,
                    "worker_count": 1,
                },
                "train_rows": int(len(train_targets)),
                "validation_rows": int(len(validation_targets)),
                "validation_metrics": weighted_metrics,
            }
            model_entries.append(weighted_entry)
            validation_rows.extend(
                {
                    KEY: int(key),
                    "track": track,
                    "candidate_id": weighted_entry["candidate_id"],
                    "high_cost_probability": float(probability),
                }
                for key, probability in zip(
                    validation_frame[KEY].tolist(), weighted_probabilities, strict=True
                )
            )
            decision = replacement_decision(
                parent_validation_metrics,
                weighted_metrics,
                ap_improvement_tolerance=settings.ap_improvement_tolerance,
                max_ap_regression_for_mcc_gain=settings.max_ap_regression_for_mcc_gain,
                max_roc_auc_regression=settings.max_roc_auc_regression,
                required_mcc_gain=settings.required_mcc_gain,
            )
        else:
            decision = {
                "replace_parent": False,
                "reason": "S0_NONE_PARENT_RETAINED",
                "route_a_ranking_improvement": False,
                "route_b_operating_point_improvement": False,
            }
        replaced = bool(decision.get("replace_parent"))
        effective_candidate_id = (
            str(weighted_entry["candidate_id"])
            if replaced and weighted_entry
            else parent_candidate_id
        )
        effective_model_entry = (
            weighted_entry if replaced and weighted_entry else parent_model_entries
        )
        effective_metrics = (
            weighted_metrics if replaced and weighted_metrics else parent_validation_metrics
        )
        effective_threshold = float(
            selected_policy["threshold"] if replaced else none_policy["threshold"]
        )
        effective_strategy = selected_id if replaced else "S0_NONE"
        decision["selected_strategy_id"] = selected_id
        decision["effective_candidate_id"] = effective_candidate_id
        decision["fallback_to_phase11_parent"] = not replaced
        effective_entries.append(
            {
                "track": track,
                "candidate_id": effective_candidate_id,
                "source_candidate_id": parent_candidate_id,
                "source_phase": 12 if replaced else parent.source_phase,
                "feature_count": parent.feature_set.feature_count,
                "feature_set_sha256": parent.feature_set.feature_set_sha256,
                "feature_list_sha256": parent.feature_list_sha256,
                "model_file": effective_model_entry["model_file"],
                "model_sha256": effective_model_entry["model_sha256"],
                "parameter_sha256": parent.parent_parameter_sha256,
                "selected_imbalance_strategy": effective_strategy,
                "technical_threshold": effective_threshold,
                "validation_metrics": effective_metrics,
                "replacement_decision": decision,
                "fallback_reason": None if replaced else decision.get("reason"),
            }
        )
        outer_candidates.append(
            {
                "candidate_id": effective_candidate_id,
                "feature_count": parent.feature_set.feature_count,
                "complexity_order": next(
                    item.complexity_order
                    for item in strategies
                    if item.strategy_id == effective_strategy
                ),
                "validation_metrics": effective_metrics,
            }
        )
        threshold_policy[track] = {
            "effective_model_candidate_id": effective_candidate_id,
            "score_space": "RAW_UNCALIBRATED_PROBABILITY",
            "technical_default": threshold_summaries[track][effective_strategy][
                "technical_default"
            ],
            "alternatives": threshold_summaries[track][effective_strategy]["alternatives"],
            "business_approved": False,
            "calibration_status": "UNCALIBRATED",
            "next_review_phase": 13,
        }
    validation_predictions = (
        pd.DataFrame(validation_rows)
        .loc[:, [KEY, "track", "candidate_id", "high_cost_probability"]]
        .sort_values(["track", "candidate_id", KEY], kind="mergesort")
        .reset_index(drop=True)
    )
    write_parquet(
        validation_predictions, work_dir / "validation_predictions.parquet", compression="zstd"
    )
    write_json(
        work_dir / "effective_model_manifest.json", {"phase": 12, "models": effective_entries}
    )
    write_json(work_dir / "model_manifest.json", {"phase": 12, "models": model_entries})
    write_json(work_dir / "threshold_policy.json", {"phase": 12, "tracks": threshold_policy})
    validation_metrics_payload = {
        "phase": 12,
        "tracks": {entry["track"]: entry for entry in effective_entries},
        "development_champion": select_phase12_champion(outer_candidates),
    }
    write_json(work_dir / "validation_metrics.json", validation_metrics_payload)
    target_audit = {
        "phase": 12,
        "train_target_rows_loaded": int(len(train_targets)),
        "train_positive_rows_loaded": positive_count,
        "train_negative_rows_loaded": negative_count,
        "validation_target_rows_loaded_before_phase12_freeze": 0,
        "validation_target_rows_loaded_after_phase12_freeze": int(len(validation_targets)),
        "validation_target_access_allowed": True,
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
    }
    write_json(work_dir / "target_access_audit.json", target_audit)
    pending_validation = {
        "phase": 12,
        "status": "PENDING_INDEPENDENT_VALIDATION",
        "valid": False,
        "hardening_status": "PENDING",
        "errors": [],
        "warnings": ["TECHNICAL_THRESHOLD_ONLY", "BUSINESS_TARGET_UNCONFIRMED"],
        "test_seal": {
            key: target_audit[key]
            for key in (
                "test_target_rows_loaded",
                "test_predictions_created",
                "test_metrics_computed",
                "test_target_access_allowed",
                "first_allowed_test_target_phase",
            )
        },
        "outer_validation_accessed": True,
        "phase12_freeze_sha256": freeze_hash,
    }
    write_json(work_dir / "validation.json", pending_validation)
    manifest = {
        "phase": 12,
        "run_id": run,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit_sha": git_commit_sha(root),
        "contract_version": PHASE12_VERSION,
        "contract_sha256": contract_check["contract_checksum"],
        "phase11_run_id": inputs.phase11_manifest.get("run_id"),
        "phase11_manifest_sha256": inputs.phase11_manifest_sha256,
        "phase11_validation_sha256": inputs.phase11_validation_sha256,
        "phase11_model_manifest_sha256": inputs.phase11_model_manifest_sha256,
        "phase10_inner_fold_sha256": inputs.fold_plan.content_sha256,
        "parents": {track: inputs.parents[track].as_dict() for track in TRACKS},
        "feature_set_hashes": {
            track: inputs.parents[track].feature_set.feature_set_sha256 for track in TRACKS
        },
        "feature_counts": {
            track: inputs.parents[track].feature_set.feature_count for track in TRACKS
        },
        "frozen_parent_parameter_hashes": {
            track: inputs.parents[track].parent_parameter_sha256 for track in TRACKS
        },
        "strategy_inventory": [strategy.as_dict() for strategy in strategies],
        "selected_strategies": {
            track: selections[track]["selected_strategy_id"] for track in TRACKS
        },
        "selected_raw_thresholds": {
            track: threshold_policy[track]["technical_default"]["threshold"] for track in TRACKS
        },
        "phase12_freeze_sha256": freeze_hash,
        "compute_plan": compute_payload,
        "detected_logical_cpus": compute.detected_logical_processors,
        "reserved_logical_cpus": compute.reserved_logical_processors,
        "worker_count": compute.worker_count,
        "threads_per_worker": compute.threads_per_fit,
        "single_fit_threads": compute.single_fit_threads,
        "effective_candidates": {
            track: effective_entries[index]["candidate_id"] for index, track in enumerate(TRACKS)
        },
        "phase12_development_champion": validation_metrics_payload["development_champion"],
        "outer_validation_accessed": True,
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "first_allowed_test_target_phase": 15,
        "warnings": pending_validation["warnings"],
    }
    manifest["artifact_file_sha256"] = _artifact_hashes(work_dir)
    write_json(work_dir / "phase12_manifest.json", manifest)

    # Certify the complete temporary bundle independently before publishing it.
    # The validator reconstructs the statistical evidence, replacement decisions,
    # model reload predictions, and provenance bindings from the bundle itself.
    from .validation import validate_existing_phase12

    independent_validation = validate_existing_phase12(work_dir, project_root=root)
    if (
        not independent_validation.get("valid")
        or independent_validation.get("hardening_status") != "HARDENED_PASS"
    ):
        details = "; ".join(str(item) for item in independent_validation.get("errors", []))
        raise ImbalanceThresholdError(
            "Independent Phase 12 validation failed before publication"
            + (f": {details}" if details else ".")
        )
    pending_warnings = [str(item) for item in cast(list[Any], pending_validation["warnings"])]
    independent_warnings = [
        str(item) for item in cast(list[Any], independent_validation.get("warnings", []))
    ]
    validation_warnings = list(dict.fromkeys(pending_warnings + independent_warnings))
    validation = {
        **independent_validation,
        "phase": 12,
        "status": "PASS WITH WARNINGS" if validation_warnings else "PASS",
        "valid": True,
        "hardening_status": "HARDENED_PASS",
        "warnings": validation_warnings,
        "outer_validation_accessed": True,
        "phase12_freeze_sha256": freeze_hash,
        "test_seal": pending_validation["test_seal"],
    }
    write_json(work_dir / "validation.json", validation)
    manifest["validation_status"] = validation["status"]
    manifest["validation_hardening_status"] = validation["hardening_status"]
    manifest["warnings"] = validation["warnings"]
    manifest["artifact_file_sha256"] = _artifact_hashes(work_dir)
    write_json(work_dir / "phase12_manifest.json", manifest)
    if report_dir is not None:
        report_root = (
            report_dir.resolve() if report_dir.is_absolute() else (root / report_dir).resolve()
        )
        report_root.mkdir(parents=True, exist_ok=True)
        write_json(
            report_root / f"{run}.json",
            {
                "phase12_manifest": manifest,
                "validation": validation,
                "validation_metrics": validation_metrics_payload,
            },
        )
    os.replace(work_dir, final_dir)
    gc.collect()
    return {
        "status": validation["status"],
        "valid": True,
        "run_id": run,
        "run_directory": str(final_dir),
        "report_directory": str(report_dir) if report_dir else None,
        "phase12_development_champion": validation_metrics_payload["development_champion"],
        "compute_plan": compute.as_dict(),
        "validation": validation,
    }


__all__ = [
    "build_phase12",
    "phase12_contract_check",
    "phase12_plan_check",
    "phase12_run_id",
]
