"""Standalone Phase 10 artifact validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..baseline_model.catboost_baseline import effective_parameters, load_model
from ..baseline_model.validation import validate_model_directory
from ..feature_mart.manifest import sha256_file
from ..paths import discover_repository_root
from .config import TRACK_TO_EXPERIMENT, load_optimization_settings
from .input import (
    KEY,
    TARGET,
    load_locked_phase9_inputs,
    load_train_targets_for_optimization,
    load_validation_targets_after_freeze,
)
from .metrics import metrics_for_predictions, validate_prediction_frame
from .models import OptimizationError
from .provenance import fold_content_sha256
from .search_space import validate_trial_parameters
from .selection import select_best_trial, select_development_champion
from .study import TRIAL_HISTORY_COLUMNS, study_history_sha256

REQUIRED_ARTIFACTS = (
    "inner_cv_folds.parquet",
    "inner_cv_manifest.json",
    "trial_history.parquet",
    "trial_fold_metrics.parquet",
    "study_freeze.json",
    "best_params.json",
    "validation_predictions.parquet",
    "validation_metrics.json",
    "target_access_audit.json",
    "optimization_manifest.json",
    "model_manifest.json",
    "validation.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OptimizationError(f"Expected a JSON object: {path}")
    return payload


def _close(left: Any, right: Any, tolerance: float = 1.0e-10) -> bool:
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=tolerance))
    return bool(left == right)


def _policy_errors(
    effective: dict[str, Any],
    fixed: dict[str, Any],
    best: dict[str, Any],
    requested: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    requested_parameters = requested or effective
    for key, value in fixed.items():
        if key not in requested_parameters or not _close(
            requested_parameters[key], value, tolerance=1.0e-6
        ):
            errors.append(f"Persisted optimized model fixed parameter differs: {key}")
    for key in ("class_weights", "auto_class_weights", "scale_pos_weight"):
        if key in effective and effective[key] not in (None, False, 0, 1, "none", "None"):
            errors.append(f"Persisted optimized model uses prohibited class weighting: {key}")
    for key in ("eval_set", "early_stopping_rounds", "od_wait"):
        if key in effective and effective[key] not in (None, False, 0, "none", "None"):
            errors.append(f"Persisted optimized model uses prohibited early stopping: {key}")
    if "od_type" in effective and str(effective["od_type"]).casefold() not in {"none", ""}:
        errors.append("Persisted optimized model uses non-disabled od_type.")
    for key, value in best.items():
        if key not in requested_parameters or not _close(
            requested_parameters[key], value, tolerance=1.0e-6
        ):
            errors.append(f"Persisted optimized model search parameter differs: {key}")
        if key in effective and not _close(effective[key], value, tolerance=1.0e-6):
            errors.append(f"Persisted effective CatBoost parameter differs: {key}")
    return errors


def validate_optimization_directory(
    optimization_dir: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Recompute Phase 10 invariants from persisted artifacts and locked inputs."""

    directory = optimization_dir.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    missing = [name for name in REQUIRED_ARTIFACTS if not (directory / name).is_file()]
    if missing:
        return {
            "status": "BLOCKED",
            "valid": False,
            "errors": ["Missing Phase 10 artifacts: " + ", ".join(missing)],
            "warnings": [],
            "hardening_status": "BLOCKED",
        }
    try:
        root = discover_repository_root(project_root)
        settings = load_optimization_settings(root)
        manifest = _read_json(directory / "optimization_manifest.json")
        if manifest.get("phase") != 10:
            errors.append("Phase 10 optimization_manifest phase is not 10.")
        phase9_dir = Path(str(manifest.get("phase9_dir", ""))).expanduser().resolve()
        phase10_inputs = load_locked_phase9_inputs(phase9_dir, project_root=root)
        phase9_validation = validate_model_directory(phase9_dir, project_root=root)
        errors.extend(
            "Phase 9 validation: " + str(item) for item in phase9_validation.get("errors", [])
        )
        if phase9_validation.get("hardening_status") != "HARDENED_PASS":
            errors.append("Phase 9 is not independently HARDENED_PASS.")
        for key, value in {
            "phase9_hardened_status": "HARDENED_PASS",
            "outer_validation_accessed": True,
            "test_target_rows_loaded": 0,
            "test_predictions_created": 0,
            "test_metrics_computed": False,
            "test_target_access_allowed": False,
            "first_allowed_test_target_phase": 15,
        }.items():
            if manifest.get(key) != value:
                errors.append(f"Phase 10 manifest safety field changed: {key}")
        fold_manifest = _read_json(directory / "inner_cv_manifest.json")
        fold_frame = pd.read_parquet(directory / "inner_cv_folds.parquet")
        expected_fold_columns = [KEY, "claim_date", "fold_id", "role"]
        if list(fold_frame.columns) != expected_fold_columns:
            errors.append("Phase 10 inner_cv_folds schema differs.")
        else:
            actual_fold_hash = fold_content_sha256(fold_frame)
            if actual_fold_hash != fold_manifest.get("fold_content_sha256"):
                errors.append("Phase 10 inner fold content hash differs.")
            if actual_fold_hash != manifest.get("inner_fold_content_sha256"):
                errors.append("Phase 10 manifest inner fold hash differs.")
        if (
            fold_manifest.get("source_outer_split") != "TRAIN"
            or fold_manifest.get("fold_count") != 3
        ):
            errors.append("Phase 10 inner fold manifest is not the required three-fold TRAIN plan.")
        for item in fold_manifest.get("folds", []):
            if not str(item.get("train_max_date")) < str(item.get("validation_min_date")):
                errors.append(f"Inner fold {item.get('fold_id')} violates strict chronology.")
        history = pd.read_parquet(directory / "trial_history.parquet")
        if tuple(history.columns) != TRIAL_HISTORY_COLUMNS:
            errors.append("Phase 10 trial_history schema differs.")
        for track in settings.tracks:
            track_history = history.loc[history["track"] == track].sort_values("trial_number")
            if len(track_history) != settings.trials_per_track:
                errors.append(f"Phase 10 {track} trial count differs from configuration.")
            for row in track_history.to_dict(orient="records"):
                params = {
                    key: row[key]
                    for key in (
                        "iterations",
                        "learning_rate",
                        "depth",
                        "l2_leaf_reg",
                        "random_strength",
                        "bagging_temperature",
                        "border_count",
                        "rsm",
                    )
                    if row[key] is not None
                }
                if params:
                    try:
                        validate_trial_parameters(params)
                    except OptimizationError as exc:
                        errors.append(f"Phase 10 {track} trial parameter range: {exc}")
            completed = track_history.loc[track_history["state"] == "COMPLETE"]
            if len(completed) < max(1, int(np.ceil(settings.trials_per_track * 0.80))):
                errors.append(f"Phase 10 {track} completed trial fraction is below 80%.")
        freeze = _read_json(directory / "study_freeze.json")
        from .manifest import freeze_payload_sha256

        if freeze.get("study_freeze_sha256") != freeze_payload_sha256(freeze):
            errors.append("Phase 10 study_freeze_sha256 does not recompute.")
        if freeze.get("outer_validation_accessed") is not False:
            errors.append("Phase 10 study freeze claims outer validation access.")
        if freeze.get("inner_fold_content_sha256") != fold_manifest.get("fold_content_sha256"):
            errors.append("Phase 10 study freeze fold hash differs.")
        if freeze.get("trial_history_content_sha256") != study_history_sha256(history):
            errors.append("Phase 10 trial history content hash differs.")
        for track in settings.tracks:
            rows = history.loc[history["track"] == track]
            try:
                best = select_best_trial(rows)
            except OptimizationError as exc:
                errors.append(str(exc))
                continue
            persisted = freeze.get("tracks", {}).get(track, {})
            if int(persisted.get("best_trial_number", -1)) != int(best["trial_number"]):
                errors.append(
                    f"Phase 10 {track} best trial differs from deterministic recomputation."
                )
            for key in (
                "mean_average_precision",
                "min_average_precision",
                "std_average_precision",
                "mean_roc_auc",
                "min_roc_auc",
                "mean_log_loss",
                "mean_brier_score",
            ):
                if not _close(persisted.get("best_inner_metrics", {}).get(key), best.get(key)):
                    errors.append(f"Phase 10 {track} best metric differs: {key}")
        train_targets, _ = load_train_targets_for_optimization(phase10_inputs)
        validation_targets, _ = load_validation_targets_after_freeze(
            phase10_inputs, study_frozen=True
        )
        prediction_frame = pd.read_parquet(directory / "validation_predictions.parquet")
        validate_prediction_frame(prediction_frame, {"P10_T1_E1_OPTIMIZED", "P10_T3_E3_OPTIMIZED"})
        validation_keys = set(
            phase10_inputs.development.loc[
                phase10_inputs.development["split"] == "VALIDATION", KEY
            ].astype(int)
        )
        if set(prediction_frame[KEY].astype(int)) != validation_keys:
            errors.append("Phase 10 validation predictions contain non-validation claim keys.")
        if len(prediction_frame) != 2 * len(validation_keys):
            errors.append("Phase 10 validation prediction row count is not two finalists only.")
        metrics_payload = _read_json(directory / "validation_metrics.json")
        persisted_metrics = metrics_payload.get("candidate_metrics", {})
        model_manifest = _read_json(directory / "model_manifest.json")
        model_entries = model_manifest.get("models", {})
        for candidate_id in ("P10_T1_E1_OPTIMIZED", "P10_T3_E3_OPTIMIZED"):
            entry = model_entries.get(candidate_id)
            if not isinstance(entry, dict):
                errors.append(f"Missing finalist model manifest entry: {candidate_id}")
                continue
            model_path = directory / str(entry.get("model_file", "")).replace("/", "\\")
            if not model_path.is_file() or sha256_file(model_path) != entry.get("model_sha256"):
                errors.append(f"Finalist model hash differs: {candidate_id}")
                continue
            model = load_model(model_path)
            errors.extend(
                _policy_errors(
                    effective_parameters(model),
                    settings.fixed_parameters,
                    entry.get("best_params", {}),
                    entry.get("model_parameters", {}),
                )
            )
            track = "T1" if candidate_id.startswith("P10_T1") else "T3"
            experiment_id = TRACK_TO_EXPERIMENT[track]
            validation_frame = phase10_inputs.development.loc[
                phase10_inputs.development["split"] == "VALIDATION"
            ].sort_values(KEY)
            from ..baseline_model.adapters import adapt_matrix
            from ..baseline_model.catboost_baseline import build_pool
            from ..baseline_model.config import load_baseline_settings

            baseline_settings = load_baseline_settings(root)
            matrix = adapt_matrix(
                validation_frame.drop(columns=[KEY, "split"]),
                phase10_inputs.feature_sets[experiment_id],
                baseline_settings,
            )
            probabilities = np.asarray(
                model.predict_proba(build_pool(matrix, phase10_inputs.feature_sets[experiment_id]))[
                    :, 1
                ],
                dtype="float64",
            )
            stored = prediction_frame.loc[
                prediction_frame["candidate_id"] == candidate_id
            ].sort_values(KEY)
            if not np.allclose(
                probabilities,
                stored["high_cost_probability"].to_numpy(dtype="float64"),
                rtol=0.0,
                atol=1.0e-12,
            ):
                errors.append(f"Finalist model reload probabilities differ: {candidate_id}")
            calculated = metrics_for_predictions(
                validation_targets.set_index(KEY)
                .loc[validation_frame[KEY].tolist()][TARGET]
                .to_numpy(dtype="int8"),
                probabilities,
                settings.threshold,
            )
            if candidate_id not in persisted_metrics:
                errors.append(f"Persisted finalist metrics missing: {candidate_id}")
            else:
                for key in (
                    "average_precision",
                    "pr_auc_trapezoidal",
                    "roc_auc",
                    "log_loss",
                    "brier_score",
                ):
                    if not _close(persisted_metrics[candidate_id].get(key), calculated[key]):
                        errors.append(f"Persisted finalist metric differs: {candidate_id}.{key}")
        baseline_predictions = pd.read_parquet(phase9_dir / "validation_predictions.parquet")
        y_validation = (
            validation_targets.set_index(KEY)
            .loc[sorted(validation_keys), TARGET]
            .to_numpy(dtype="int8")
        )
        validation_metrics = dict(persisted_metrics)
        candidates = []
        for experiment_id, _track in (("E1", "T1"), ("E3", "T3")):
            candidate_id = f"P9_{experiment_id}_BASELINE"
            frame = baseline_predictions.loc[
                baseline_predictions["experiment_id"] == experiment_id
            ].sort_values(KEY)
            validation_metrics[candidate_id] = metrics_for_predictions(
                y_validation,
                frame["probability"].to_numpy(dtype="float64"),
                settings.threshold,
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "metrics": validation_metrics[candidate_id],
                    "feature_count": phase10_inputs.feature_sets[experiment_id].feature_count,
                }
            )
        for candidate_id, experiment_id in (
            ("P10_T1_E1_OPTIMIZED", "E1"),
            ("P10_T3_E3_OPTIMIZED", "E3"),
        ):
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "metrics": validation_metrics[candidate_id],
                    "feature_count": phase10_inputs.feature_sets[experiment_id].feature_count,
                }
            )
        champion = select_development_champion(candidates)
        if metrics_payload.get("phase10_development_champion") != champion["candidate_id"]:
            errors.append("Phase 10 champion differs from deterministic selection.")
        audit = _read_json(directory / "target_access_audit.json")
        for key, value in {
            "test_target_rows_loaded": 0,
            "test_predictions_created": 0,
            "test_metrics_computed": False,
            "test_target_access_allowed": False,
            "first_allowed_test_target_phase": 15,
            "outer_validation_accessed_before_study_freeze": False,
        }.items():
            if audit.get(key) != value:
                errors.append(f"Phase 10 target audit TEST/timing field changed: {key}")
        artifact_hashes = manifest.get("artifact_file_sha256", {})
        if not isinstance(artifact_hashes, dict):
            errors.append("Phase 10 artifact_file_sha256 is missing.")
        else:
            for name, digest in artifact_hashes.items():
                path = directory / str(name).replace("/", "\\")
                if not path.is_file() or sha256_file(path) != digest:
                    errors.append(f"Phase 10 artifact file hash differs: {name}")
        warnings.extend(str(item) for item in manifest.get("warnings", []))
    except Exception as exc:
        errors.append(str(exc))
    return {
        "status": "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS"),
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "hardening_status": "HARDENED_PASS" if not errors else "BLOCKED",
    }
