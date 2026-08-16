"""Phase 11 feature-family ablation, TRAIN-only selection, and publication."""

from __future__ import annotations

import gc
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from ..baseline_model.adapters import adapt_matrix
from ..baseline_model.catboost_baseline import build_pool, save_model
from ..baseline_model.config import load_baseline_settings
from ..baseline_model.models import FeatureSetSpec
from ..catboost_optimization.inner_folds import DATE, KEY, build_inner_fold_plan
from ..catboost_optimization.input import (
    CLAIM_DATE,
    load_locked_phase9_inputs,
    load_train_targets_for_optimization,
    load_validation_targets_after_freeze,
)
from ..catboost_optimization.metrics import aggregate_fold_metrics, metrics_for_predictions
from ..catboost_optimization.models import InnerFold, InnerFoldPlan
from ..catboost_optimization.provenance import (
    canonical_json_sha256,
    fold_content_sha256,
    sha256_file,
)
from ..feature_mart.manifest import git_commit_sha, write_json, write_parquet
from ..paths import discover_repository_root
from .checkpoint import load_valid_checkpoint, write_checkpoint
from .config import (
    TRACK_TO_EXPERIMENT,
    TRACKS,
    FeatureSelectionError,
    FeatureSelectionSettings,
    load_feature_selection_settings,
)
from .contract import (
    CONTRACT_VERSION,
    REQUIRED_FEATURE_HASHES,
    REQUIRED_PHASE9_RUN_ID,
    REQUIRED_PHASE10_RUN_ID,
    validate_feature_selection_contract,
)
from .grouping import build_feature_group_manifest, validate_group_membership
from .planner import ComputePlan, build_compute_plan
from .selection import (
    CandidateDefinition,
    feature_list_sha256,
    feature_set_sha256,
    generate_candidates,
    replacement_decision,
    select_candidate,
    stable_importance_ranking,
    subset_feature_set,
)

TARGET = "target__high_cost_claim_flag"
_MODEL_FORBIDDEN = {
    "class_weights",
    "auto_class_weights",
    "scale_pos_weight",
    "early_stopping_rounds",
    "eval_set",
    "od_type",
    "od_wait",
}


def phase11_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _resolve(root: Path, path: Path | None, default: str) -> Path:
    value = path if path is not None else Path(default)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureSelectionError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise FeatureSelectionError(f"JSON artifact must be an object: {path}")
    return payload


def _train_rows(inputs: Any) -> pd.DataFrame:
    frame = inputs.development.loc[inputs.development["split"] == "TRAIN"].copy()
    return frame.sort_values(KEY, kind="mergesort").reset_index(drop=True)


def _phase10_artifact_hashes(directory: Path, manifest: dict[str, Any]) -> None:  # pragma: no cover
    declared = manifest.get("artifact_file_sha256")
    if not isinstance(declared, dict):
        raise FeatureSelectionError("Phase 10 artifact_file_sha256 is missing.")
    for name, digest in declared.items():
        path = directory / str(name)
        if not path.is_file() or sha256_file(path) != str(digest):
            raise FeatureSelectionError(f"Phase 10 artifact hash changed: {name}")


def _validate_upstream(phase10_dir: Path, root: Path) -> dict[str, Any]:  # pragma: no cover
    phase10_dir = phase10_dir.expanduser().resolve()
    required = (
        "optimization_manifest.json",
        "model_manifest.json",
        "validation_metrics.json",
        "validation.json",
        "best_params.json",
        "study_freeze.json",
        "inner_cv_folds.parquet",
        "inner_cv_manifest.json",
        "phase10_acceptance_overlay.json",
        "target_access_audit.json",
    )
    missing = [name for name in required if not (phase10_dir / name).is_file()]
    if missing:
        raise FeatureSelectionError("Phase 10 artifacts missing: " + ", ".join(missing))
    manifest = _read_json(phase10_dir / "optimization_manifest.json")
    if manifest.get("phase") != 10 or manifest.get("run_id") != REQUIRED_PHASE10_RUN_ID:
        raise FeatureSelectionError("Phase 11 requires the locked Phase 10 run id.")
    if manifest.get("contract_version") != "phase10_catboost_optimization_v2":
        raise FeatureSelectionError("Phase 10 contract version drifted.")
    contract_check = __import__(
        "warranty_analytics_model.catboost_optimization.contract",
        fromlist=["validate_optimization_contract"],
    ).validate_optimization_contract(root)
    if not contract_check.get("valid"):
        raise FeatureSelectionError(
            "Phase 10 contract is not valid: " + "; ".join(contract_check.get("errors", []))
        )
    if manifest.get("contract_checksum") != contract_check.get("contract_checksum"):
        raise FeatureSelectionError("Phase 10 contract checksum differs from current repository.")
    _phase10_artifact_hashes(phase10_dir, manifest)
    validation = _read_json(phase10_dir / "validation.json")
    if validation.get("hardening_status") != "HARDENED_PASS" or validation.get("valid") is not True:
        raise FeatureSelectionError("Phase 10 validation is not HARDENED_PASS.")
    overlay = _read_json(phase10_dir / "phase10_acceptance_overlay.json")
    if (
        overlay.get("run_id") != REQUIRED_PHASE10_RUN_ID
        or overlay.get("overlay_version") != "phase10_acceptance_overlay_v1"
    ):
        raise FeatureSelectionError("Phase 10 acceptance overlay is invalid.")
    source = overlay.get("source_manifest", {})
    if source.get("sha256") != sha256_file(
        phase10_dir / "optimization_manifest.json"
    ) or source.get("contract_checksum") != manifest.get("contract_checksum"):
        raise FeatureSelectionError("Phase 10 acceptance overlay source manifest is stale.")
    seal = overlay.get("test_seal", {})
    expected_seal = {
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
    }
    if seal != expected_seal:
        raise FeatureSelectionError("Phase 10 TEST seal changed.")
    audit = _read_json(phase10_dir / "target_access_audit.json")
    if any(audit.get(key) != value for key, value in expected_seal.items()):
        raise FeatureSelectionError("Phase 10 target-access audit TEST seal changed.")
    phase9_dir = root / "artifacts" / "baseline_models" / REQUIRED_PHASE9_RUN_ID
    return {
        "phase10_dir": phase10_dir,
        "phase10_manifest": manifest,
        "phase10_validation": validation,
        "overlay": overlay,
        "phase9_dir": phase9_dir,
    }


def _frozen_fold_plan(
    phase10_dir: Path, inputs: Any, train_targets: pd.DataFrame, settings: FeatureSelectionSettings
) -> InnerFoldPlan:  # pragma: no cover
    table = pd.read_parquet(phase10_dir / "inner_cv_folds.parquet")
    required = [KEY, DATE, "fold_id", "role"]
    if list(table.columns) != required:
        raise FeatureSelectionError("Phase 10 inner fold schema changed.")
    manifest = _read_json(phase10_dir / "inner_cv_manifest.json")
    digest = fold_content_sha256(table)
    if digest != manifest.get("fold_content_sha256"):
        raise FeatureSelectionError("Phase 10 inner fold content hash changed.")
    persisted = _frozen_from_assignments(table, manifest)
    train = _train_rows(inputs)
    reconstructed = build_inner_fold_plan(
        train[[KEY, CLAIM_DATE]].rename(columns={CLAIM_DATE: DATE}),
        train_targets,
        fractions=(0.55, 0.70, 0.85, 1.0),
        minimum_train_positive=40,
        minimum_validation_positive=10,
    )
    if reconstructed.content_sha256 != digest:
        raise FeatureSelectionError("Phase 11 could not reproduce the exact Phase 10 inner folds.")
    return persisted


def _frozen_from_assignments(
    assignments: pd.DataFrame, manifest: dict[str, Any]
) -> InnerFoldPlan:  # pragma: no cover
    folds: list[InnerFold] = []
    for item in manifest.get("folds", []):
        fold_id = int(item["fold_id"])
        part = assignments.loc[assignments["fold_id"] == fold_id]
        train = tuple(int(value) for value in part.loc[part["role"] == "TRAIN", KEY].tolist())
        validation = tuple(
            int(value) for value in part.loc[part["role"] == "VALIDATION", KEY].tolist()
        )
        folds.append(
            InnerFold(
                fold_id,
                train,
                validation,
                str(item["train_max_date"]),
                str(item["validation_min_date"]),
                str(item["validation_max_date"]),
                int(item["train_rows"]),
                int(item["validation_rows"]),
                int(item["train_positive_count"]),
                int(item["validation_positive_count"]),
                str(item["train_membership_sha256"]),
                str(item["validation_membership_sha256"]),
            )
        )
    if len(folds) != 3:
        raise FeatureSelectionError("Phase 10 inner fold count is not exactly three.")
    return InnerFoldPlan(
        assignments=assignments,
        folds=tuple(sorted(folds, key=lambda fold: fold.fold_id)),
        manifest=manifest,
        content_sha256=str(manifest["fold_content_sha256"]),
    )


def _model_params(
    parent_id: str,
    track: str,
    upstream: dict[str, Any],
    root: Path,
    threads: int,
) -> tuple[dict[str, Any], dict[str, Any]]:  # pragma: no cover
    phase10_manifest = upstream["phase10_manifest"]
    phase10_dir: Path = upstream["phase10_dir"]
    fixed = dict(phase10_manifest.get("settings", {}).get("fixed_parameters", {}))
    if parent_id.startswith("P10_"):
        best = _read_json(phase10_dir / "best_params.json")[track]["best_params"]
        params = {**fixed, **best}
    else:
        baseline = load_baseline_settings(root)
        params = dict(baseline.catboost_parameters)
    params["thread_count"] = int(threads)
    params["allow_writing_files"] = False
    params["verbose"] = False
    params["use_best_model"] = False
    if set(params) & _MODEL_FORBIDDEN:
        raise FeatureSelectionError(
            "Frozen parent parameters contain a prohibited training option."
        )
    if (
        params.get("loss_function") != "Logloss"
        or params.get("bootstrap_type") != "Bayesian"
        or params.get("random_seed") != 20260810
        or params.get("task_type") != "CPU"
    ):
        raise FeatureSelectionError("Frozen parent model semantics drifted.")
    statistical = {key: value for key, value in params.items() if key != "thread_count"}
    return params, statistical


def _parent_info(
    track: str, inputs: Any, upstream: dict[str, Any], root: Path, compute: ComputePlan
) -> tuple[dict[str, Any], dict[str, Any], FeatureSetSpec]:  # pragma: no cover
    metrics = _read_json(upstream["phase10_dir"] / "validation_metrics.json")
    model_manifest = _read_json(upstream["phase10_dir"] / "model_manifest.json")
    best_params = _read_json(upstream["phase10_dir"] / "best_params.json")
    optimized_id = f"P10_{track}_{TRACK_TO_EXPERIMENT[track]}_OPTIMIZED"
    baseline_id = f"P9_{TRACK_TO_EXPERIMENT[track]}_BASELINE"
    comparison = metrics.get("replacement_decisions", {}).get(track, {})
    use_optimized = (
        comparison.get("fallback_to_baseline") is False
        and comparison.get("optimized_beats_baseline") is True
    )
    parent_id = optimized_id if use_optimized else baseline_id
    parent_entry = model_manifest.get("models", {}).get(parent_id, {})
    experiment = TRACK_TO_EXPERIMENT[track]
    feature_set = inputs.feature_sets[experiment]
    if feature_set.feature_set_sha256 != REQUIRED_FEATURE_HASHES[track]:
        raise FeatureSelectionError(f"{track} parent feature hash differs from the Phase 10 lock.")
    if parent_id.startswith("P10_"):
        source = upstream["phase10_dir"] / str(parent_entry.get("model_file"))
        inner = best_params[track]["best_inner_metrics"]
        parameter_source = best_params[track]["best_params"]
        parameter_hash = str(best_params[track]["best_param_sha256"])
    else:
        p9_manifest = _read_json(upstream["phase9_dir"] / "model_manifest.json")
        p9_entry = p9_manifest.get("models", {}).get(experiment, {})
        source = upstream["phase9_dir"] / str(p9_entry.get("model_file"))
        inner = best_params[track]["baseline_inner_cv_metrics"]
        parameter_source = load_baseline_settings(root).catboost_parameters
        parameter_hash = canonical_json_sha256(parameter_source)
    if not source.is_file():
        raise FeatureSelectionError(f"Effective parent model is missing: {source}")
    params, statistical = _model_params(
        parent_id, track, upstream, root, compute.threads_per_worker
    )
    parent = {
        "track": track,
        "phase9_experiment": experiment,
        "effective_parent_candidate_id": parent_id,
        "parent_feature_count": feature_set.feature_count,
        "parent_feature_set_sha256": feature_set.feature_set_sha256,
        "parent_parameter_sha256": parameter_hash,
        "parent_validation_metrics": metrics.get("candidate_metrics", {}).get(parent_id, {}),
        "parent_inner_metrics": inner,
        "parent_model_source": str(source),
        "fallback_from_phase10_optimization": not use_optimized,
        "statistical_parameters": statistical,
        "execution_parameters": {
            "thread_count": compute.threads_per_worker,
            "worker_count": compute.worker_count,
        },
    }
    if not parent["parent_validation_metrics"]:
        raise FeatureSelectionError(
            f"Phase 10 validation metrics lack effective parent {parent_id}."
        )
    return parent, params, feature_set


def _fit_fold(
    matrix: pd.DataFrame,
    target_by_key: pd.Series,
    fold: InnerFold,
    feature_set: FeatureSetSpec,
    parameters: dict[str, Any],
    root: Path,
    *,
    importance: bool = False,
) -> tuple[dict[str, Any], CatBoostClassifier | None, dict[str, Any] | None]:  # pragma: no cover
    train = matrix.loc[matrix[KEY].isin(fold.train_keys)].sort_values(KEY, kind="mergesort")
    validation = matrix.loc[matrix[KEY].isin(fold.validation_keys)].sort_values(
        KEY, kind="mergesort"
    )
    if len(train) != fold.train_rows or len(validation) != fold.validation_rows:
        raise FeatureSelectionError(f"Inner fold {fold.fold_id} membership changed.")
    y_train = target_by_key.loc[train[KEY].tolist()].to_numpy(dtype="int8")
    y_val = target_by_key.loc[validation[KEY].tolist()].to_numpy(dtype="int8")
    baseline = load_baseline_settings(root)
    # BaselineSettings is slots/frozen in current releases; replace keeps the
    # categorical/text policy unchanged while overriding execution parameters.
    from dataclasses import replace

    adapted_train = adapt_matrix(
        train.drop(columns=[KEY]), feature_set, replace(baseline, catboost_parameters=parameters)
    )
    adapted_val = adapt_matrix(
        validation.drop(columns=[KEY]),
        feature_set,
        replace(baseline, catboost_parameters=parameters),
    )
    model = CatBoostClassifier(**parameters)
    started = time.perf_counter()
    model.fit(build_pool(adapted_train, feature_set, y_train))
    elapsed = time.perf_counter() - started
    probabilities = np.asarray(
        model.predict_proba(build_pool(adapted_val, feature_set))[:, 1], dtype="float64"
    )
    metrics = metrics_for_predictions(y_val, probabilities, threshold=0.5)
    metrics.update(
        {
            "fold_id": fold.fold_id,
            "train_rows": fold.train_rows,
            "validation_rows": fold.validation_rows,
            "training_seconds": elapsed,
        }
    )
    evidence: dict[str, Any] | None = None
    if importance:
        importance_pool = build_pool(adapted_val, feature_set, y_val)
        loss = np.asarray(
            model.get_feature_importance(data=importance_pool, type="LossFunctionChange"),
            dtype="float64",
        )
        shap_values = np.asarray(
            model.get_feature_importance(data=importance_pool, type="ShapValues"), dtype="float64"
        )
        if shap_values.ndim != 2 or shap_values.shape[1] != feature_set.feature_count + 1:
            raise FeatureSelectionError("CatBoost SHAP output schema changed.")
        evidence = {
            "loss": loss.tolist(),
            "shap": np.mean(np.abs(shap_values[:, :-1]), axis=0).tolist(),
        }
    return metrics, model, evidence


def _aggregate(
    rows: list[dict[str, Any]], feature_count: int, reduction_fraction: float
) -> dict[str, Any]:
    aggregate = aggregate_fold_metrics(rows)
    aggregate["max_average_precision"] = max(float(row["average_precision"]) for row in rows)
    aggregate["feature_count"] = int(feature_count)
    aggregate["reduction_fraction"] = float(reduction_fraction)
    aggregate["training_seconds"] = float(
        sum(float(row.get("training_seconds", 0.0)) for row in rows)
    )
    return aggregate


def _evaluate_experiment(
    experiment: CandidateDefinition,
    matrix: pd.DataFrame,
    target_by_key: pd.Series,
    folds: InnerFoldPlan,
    parent_spec: FeatureSetSpec,
    parameters: dict[str, Any],
    root: Path,
    work_dir: Path,
    *,
    workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:  # pragma: no cover
    spec = subset_feature_set(parent_spec, experiment.feature_list, experiment.candidate_id)
    parameter_hash = canonical_json_sha256(
        {key: value for key, value in parameters.items() if key != "thread_count"}
    )
    spec_hash = canonical_json_sha256(experiment.as_dict())
    rows: list[dict[str, Any]] = []
    for fold in folds.folds:
        cached = load_valid_checkpoint(
            work_dir,
            experiment_id=experiment.candidate_id,
            experiment_spec_sha256=spec_hash,
            track=experiment.track,
            feature_set_sha256=spec.feature_set_sha256,
            parameter_sha256=parameter_hash,
            fold_id=fold.fold_id,
        )
        if cached is not None:
            rows.append(dict(cached["metrics"]))
            continue
        metrics, _, _ = _fit_fold(matrix, target_by_key, fold, spec, parameters, root)
        write_checkpoint(
            work_dir,
            {
                "experiment_id": experiment.candidate_id,
                "experiment_spec_sha256": spec_hash,
                "track": experiment.track,
                "feature_set_sha256": spec.feature_set_sha256,
                "parameter_sha256": parameter_hash,
                "fold_id": fold.fold_id,
                "metrics": metrics,
                "training_seconds": metrics["training_seconds"],
                "completed_at": datetime.now(UTC).isoformat(),
            },
        )
        rows.append(metrics)
        gc.collect()
    rows.sort(key=lambda row: int(row["fold_id"]))
    return _aggregate(rows, experiment.feature_count, experiment.reduction_fraction), rows


def phase11_plan_check(
    phase10_dir: Path,
    *,
    project_root: Path | None = None,
    max_workers: int | None = None,
    threads_per_fit: int | None = None,
    single_fit_threads: int | None = None,
) -> dict[str, Any]:  # pragma: no cover
    root = discover_repository_root(project_root)
    errors: list[str] = []
    contract = validate_feature_selection_contract(root)
    errors.extend(contract.get("errors", []))
    inputs = None
    upstream = None
    folds = None
    train_targets = None
    compute = None
    try:
        settings = load_feature_selection_settings(root)
        upstream = _validate_upstream(phase10_dir, root)
        inputs = load_locked_phase9_inputs(upstream["phase9_dir"], project_root=root)
        train_targets, _ = load_train_targets_for_optimization(inputs)
        compute = build_compute_plan(
            settings,
            max_workers=max_workers,
            threads_per_fit=threads_per_fit,
            single_fit_threads=single_fit_threads,
        )
        folds = _frozen_fold_plan(upstream["phase10_dir"], inputs, train_targets, settings)
        if (
            len(train_targets) != 5952
            or int((inputs.phase9_inputs.assignments["split"] == "VALIDATION").sum()) != 1275
        ):
            raise FeatureSelectionError("Phase 11 outer population counts drifted.")
    except Exception as exc:
        errors.append(str(exc))
    return {
        "status": "BLOCKED" if errors else "PASS",
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": [],
        "contract": contract,
        "inputs": inputs,
        "upstream": upstream,
        "train_targets": train_targets,
        "inner_fold_plan": folds,
        "compute_plan": compute,
    }


def _importance_rows(
    track: str,
    parent_spec: FeatureSetSpec,
    matrix: pd.DataFrame,
    targets: pd.DataFrame,
    folds: InnerFoldPlan,
    parameters: dict[str, Any],
    root: Path,
    family_by_feature: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:  # pragma: no cover
    target_by_key = targets.set_index(KEY)[TARGET].astype("int8")
    by_fold: list[dict[str, Any]] = []
    loss_values: dict[str, list[float]] = {name: [] for name in parent_spec.feature_names}
    shap_values: dict[str, list[float]] = {name: [] for name in parent_spec.feature_names}
    for fold in folds.folds:
        metrics, model, evidence = _fit_fold(
            matrix, target_by_key, fold, parent_spec, parameters, root, importance=True
        )
        if evidence is None or model is None:
            raise FeatureSelectionError("Importance evidence is missing.")
        loss = np.asarray(evidence["loss"], dtype="float64")
        shap = np.asarray(evidence["shap"], dtype="float64")
        loss_order = np.argsort(-loss, kind="mergesort")
        shap_order = np.argsort(-shap, kind="mergesort")
        loss_rank = {int(index): rank + 1 for rank, index in enumerate(loss_order)}
        shap_rank = {int(index): rank + 1 for rank, index in enumerate(shap_order)}
        for index, feature in enumerate(parent_spec.feature_names):
            loss_values[feature].append(float(loss[index]))
            shap_values[feature].append(float(shap[index]))
            by_fold.append(
                {
                    "track": track,
                    "feature": feature,
                    "family": family_by_feature[feature],
                    "fold_id": fold.fold_id,
                    "loss_function_change": float(loss[index]),
                    "loss_rank": loss_rank[index],
                    "loss_percentile_rank": 1.0
                    - (loss_rank[index] - 1) / max(1, len(parent_spec.feature_names) - 1),
                    "mean_abs_shap": float(shap[index]),
                    "shap_rank": shap_rank[index],
                    "shap_percentile_rank": 1.0
                    - (shap_rank[index] - 1) / max(1, len(parent_spec.feature_names) - 1),
                }
            )
        del model
        gc.collect()
    stability: list[dict[str, Any]] = []
    for feature in parent_spec.feature_names:
        rows = [row for row in by_fold if row["feature"] == feature]
        loss_percentiles = [float(row["loss_percentile_rank"]) for row in rows]
        shap_percentiles = [float(row["shap_percentile_rank"]) for row in rows]
        loss_ranks = [float(row["loss_rank"]) for row in rows]
        shap_ranks = [float(row["shap_rank"]) for row in rows]
        stability.append(
            {
                "track": track,
                "feature": feature,
                "family": family_by_feature[feature],
                "median_loss_percentile": float(np.median(loss_percentiles)),
                "median_shap_percentile": float(np.median(shap_percentiles)),
                "top_25_percent_fold_count": int(sum(value >= 0.75 for value in loss_percentiles)),
                "top_50_percent_fold_count": int(sum(value >= 0.50 for value in loss_percentiles)),
                "loss_rank_std": float(np.std(loss_ranks, ddof=0)),
                "shap_rank_std": float(np.std(shap_ranks, ddof=0)),
                "stable_score": float(
                    0.60 * np.median(loss_percentiles) + 0.40 * np.median(shap_percentiles)
                ),
            }
        )
    ranking = stable_importance_ranking(stability)
    return by_fold, stability, {"ranking": ranking}


def _family_ablation(
    track: str,
    parent: FeatureSetSpec,
    family_by_feature: dict[str, str],
    matrix: pd.DataFrame,
    targets: pd.DataFrame,
    folds: InnerFoldPlan,
    parameters: dict[str, Any],
    root: Path,
    work_dir: Path,
) -> list[dict[str, Any]]:  # pragma: no cover
    target_by_key = targets.set_index(KEY)[TARGET].astype("int8")
    families = sorted(set(family_by_feature.values()))
    results: list[dict[str, Any]] = []
    # Parent metrics are obtained from the deterministic replay before this call.
    for family in families:
        candidate_id = f"P11_{track}_ABLATE_{family}"
        features = tuple(name for name in parent.feature_names if family_by_feature[name] != family)
        candidate = CandidateDefinition(
            candidate_id,
            track,
            len(features),
            parent.feature_count - len(features),
            (parent.feature_count - len(features)) / parent.feature_count,
            feature_set_sha256(candidate_id, features),
            features,
            "leave_one_family_out",
            {},
        )
        spec = subset_feature_set(parent, features, candidate_id)
        rows: list[dict[str, Any]] = []
        parameter_hash = canonical_json_sha256(
            {key: value for key, value in parameters.items() if key != "thread_count"}
        )
        spec_hash = canonical_json_sha256(candidate.as_dict())
        for fold in folds.folds:
            cached = load_valid_checkpoint(
                work_dir,
                experiment_id=candidate_id,
                experiment_spec_sha256=spec_hash,
                track=track,
                feature_set_sha256=spec.feature_set_sha256,
                parameter_sha256=parameter_hash,
                fold_id=fold.fold_id,
            )
            if cached is not None:
                row = dict(cached["metrics"])
            else:
                row, _, _ = _fit_fold(matrix, target_by_key, fold, spec, parameters, root)
                write_checkpoint(
                    work_dir,
                    {
                        "experiment_id": candidate_id,
                        "experiment_spec_sha256": spec_hash,
                        "track": track,
                        "feature_set_sha256": spec.feature_set_sha256,
                        "parameter_sha256": parameter_hash,
                        "fold_id": fold.fold_id,
                        "metrics": row,
                        "training_seconds": row["training_seconds"],
                        "completed_at": datetime.now(UTC).isoformat(),
                    },
                )
            rows.append(row)
        agg = _aggregate(rows, len(features), candidate.reduction_fraction)
        results.append(
            {
                "track": track,
                "family": family,
                "removed_feature_count": parent.feature_count - len(features),
                "remaining_feature_count": len(features),
                **{
                    key: agg[key]
                    for key in (
                        "mean_average_precision",
                        "min_average_precision",
                        "std_average_precision",
                        "mean_roc_auc",
                        "mean_log_loss",
                        "mean_brier_score",
                    )
                },
            }
        )
    return results


def build_phase11(
    phase10_dir: Path,
    *,
    output_dir: Path | None = None,
    report_dir: Path | None = None,
    run_id: str | None = None,
    overwrite: bool = False,
    resume: bool = False,
    max_workers: int | None = None,
    threads_per_fit: int | None = None,
    single_fit_threads: int | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:  # pragma: no cover
    root = discover_repository_root(project_root)
    settings = load_feature_selection_settings(root)
    plan = phase11_plan_check(
        phase10_dir,
        project_root=root,
        max_workers=max_workers,
        threads_per_fit=threads_per_fit,
        single_fit_threads=single_fit_threads,
    )
    if (
        not plan["valid"]
        or plan["inputs"] is None
        or plan["train_targets"] is None
        or plan["inner_fold_plan"] is None
        or plan["upstream"] is None
        or plan["compute_plan"] is None
    ):
        raise FeatureSelectionError("Phase 11 plan blocks selection: " + "; ".join(plan["errors"]))
    inputs = plan["inputs"]
    train_targets = plan["train_targets"]
    folds = plan["inner_fold_plan"]
    upstream = plan["upstream"]
    compute: ComputePlan = plan["compute_plan"]
    selected_run_id = run_id or phase11_run_id()
    output_root = _resolve(root, output_dir, settings.output_directory)
    report_root = _resolve(root, report_dir, settings.report_directory)
    final_dir = output_root / selected_run_id
    if final_dir.exists() and not resume and not overwrite:
        raise FeatureSelectionError(f"Completed Phase 11 run is immutable: {final_dir}")
    if final_dir.exists() and overwrite and not resume:
        shutil.rmtree(final_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir = output_root / f".phase11_{selected_run_id}.work"
    work_dir.mkdir(parents=True, exist_ok=True)
    settings_compute = compute.as_dict()
    parents: dict[str, dict[str, Any]] = {}
    parameters: dict[str, dict[str, Any]] = {}
    parent_specs: dict[str, FeatureSetSpec] = {}
    for track in TRACKS:
        parents[track], parameters[track], parent_specs[track] = _parent_info(
            track, inputs, upstream, root, compute
        )
    parent_features = {track: parent_specs[track].feature_names for track in TRACKS}
    group_manifest, membership = build_feature_group_manifest(
        parent_features, inputs.phase9_inputs.phase7_lineage, inputs.phase9_inputs.phase8_lineage
    )
    expected_features = set(name for names in parent_features.values() for name in names)
    validate_group_membership(membership, expected_features)
    write_json(work_dir / "feature_group_manifest.json", group_manifest)
    write_parquet(membership, work_dir / "feature_group_membership.parquet", compression="snappy")
    family_by_feature = {str(row.feature): str(row.family) for row in membership.itertuples()}
    train_matrix = _train_rows(inputs)
    target_by_key = train_targets.set_index(KEY)[TARGET].astype("int8")
    parent_replay: dict[str, Any] = {}
    importance_by_fold: list[dict[str, Any]] = []
    importance_stability: list[dict[str, Any]] = []
    rankings: dict[str, list[str]] = {}
    family_results: list[dict[str, Any]] = []
    # Replay and importance intentionally run before any candidate definitions.
    for track in TRACKS:
        parent = parent_specs[track]
        rows: list[dict[str, Any]] = []
        for fold in folds.folds:
            experiment_id = f"P11_{track}_PARENT_REPLAY"
            spec_hash = canonical_json_sha256(
                {
                    "track": track,
                    "feature_set_sha256": parent.feature_set_sha256,
                    "kind": "parent_replay",
                }
            )
            parameter_hash = canonical_json_sha256(
                {key: value for key, value in parameters[track].items() if key != "thread_count"}
            )
            cached = load_valid_checkpoint(
                work_dir,
                experiment_id=experiment_id,
                experiment_spec_sha256=spec_hash,
                track=track,
                feature_set_sha256=parent.feature_set_sha256,
                parameter_sha256=parameter_hash,
                fold_id=fold.fold_id,
            )
            if cached is not None:
                row = dict(cached["metrics"])
            else:
                row, _, _ = _fit_fold(
                    train_matrix, target_by_key, fold, parent, parameters[track], root
                )
                write_checkpoint(
                    work_dir,
                    {
                        "experiment_id": experiment_id,
                        "experiment_spec_sha256": spec_hash,
                        "track": track,
                        "feature_set_sha256": parent.feature_set_sha256,
                        "parameter_sha256": parameter_hash,
                        "fold_id": fold.fold_id,
                        "metrics": row,
                        "training_seconds": row["training_seconds"],
                        "completed_at": datetime.now(UTC).isoformat(),
                    },
                )
            rows.append(row)
        aggregate = _aggregate(rows, parent.feature_count, 0.0)
        expected = parents[track]["parent_inner_metrics"]
        replay_fields = (
            "mean_average_precision",
            "min_average_precision",
            "std_average_precision",
            "mean_roc_auc",
            "min_roc_auc",
            "mean_log_loss",
            "mean_brier_score",
        )
        diffs = {field: float(aggregate[field]) - float(expected[field]) for field in replay_fields}
        if any(abs(value) > 1e-10 for value in diffs.values()):
            raise FeatureSelectionError(
                f"{track} effective parent inner-CV replay differs by more than 1e-10: {diffs}"
            )
        parent_replay[track] = {
            "expected": expected,
            "reproduced": aggregate,
            "absolute_differences": {key: abs(value) for key, value in diffs.items()},
            "status": "PASS",
        }
        fold_rows, stability_rows, ranking = _importance_rows(
            track,
            parent,
            train_matrix,
            train_targets,
            folds,
            parameters[track],
            root,
            family_by_feature,
        )
        importance_by_fold.extend(fold_rows)
        importance_stability.extend(stability_rows)
        rankings[track] = ranking["ranking"]
    # Family ablation runs use the same frozen parent parameters and folds.
    for track in TRACKS:
        family_results.extend(
            _family_ablation(
                track,
                parent_specs[track],
                family_by_feature,
                train_matrix,
                train_targets,
                folds,
                parameters[track],
                root,
                work_dir,
            )
        )
    family_parent_ap = {
        track: float(parent_replay[track]["reproduced"]["mean_average_precision"])
        for track in TRACKS
    }
    for row in family_results:
        parent_ap = family_parent_ap[str(row["track"])]
        row["delta_ap_vs_parent"] = parent_ap - float(row["mean_average_precision"])
        row["delta_roc_vs_parent"] = float(
            parent_replay[str(row["track"])]["reproduced"]["mean_roc_auc"]
        ) - float(row["mean_roc_auc"])
        row["delta_logloss_vs_parent"] = float(row["mean_log_loss"]) - float(
            parent_replay[str(row["track"])]["reproduced"]["mean_log_loss"]
        )
    # Definitions are frozen before expensive candidate evaluation begins.
    candidate_definitions: dict[str, list[CandidateDefinition]] = {}
    for track in TRACKS:
        candidate_definitions[track] = generate_candidates(
            track,
            parent_specs[track],
            rankings[track],
            [row for row in family_results if row["track"] == track],
            [row for row in importance_stability if row["track"] == track],
            family_by_feature,
            settings,
        )
    candidate_payload = {
        track: [candidate.as_dict() for candidate in candidates]
        for track, candidates in candidate_definitions.items()
    }
    write_json(work_dir / "candidate_feature_sets.json", candidate_payload)
    candidate_results: list[dict[str, Any]] = []
    candidate_fold_metrics: list[dict[str, Any]] = []
    jobs: list[tuple[CandidateDefinition, str]] = [
        (candidate, track) for track in TRACKS for candidate in candidate_definitions[track]
    ]
    with ThreadPoolExecutor(max_workers=compute.worker_count) as executor:
        future_map = {
            executor.submit(
                _evaluate_experiment,
                candidate,
                train_matrix,
                target_by_key,
                folds,
                parent_specs[track],
                parameters[track],
                root,
                work_dir,
                workers=compute.worker_count,
            ): (candidate, track)
            for candidate, track in jobs
        }
        for future in as_completed(future_map):
            candidate, track = future_map[future]
            aggregate, fold_rows = future.result()
            candidate_results.append({"track": track, **candidate.as_dict(), **aggregate})
            candidate_fold_metrics.extend(
                {"track": track, "candidate_id": candidate.candidate_id, **row} for row in fold_rows
            )
    candidate_results.sort(key=lambda row: (str(row["track"]), str(row["candidate_id"])))
    candidate_fold_metrics.sort(
        key=lambda row: (str(row["track"]), str(row["candidate_id"]), int(row["fold_id"]))
    )
    selected: dict[str, dict[str, Any]] = {}
    decision_trace: dict[str, Any] = {}
    for track in TRACKS:
        chosen, trace = select_candidate(
            [row for row in candidate_results if row["track"] == track], settings
        )
        selected[track] = chosen
        decision_trace[track] = trace
    # TRAIN-only freeze gate. Nothing below this point may change selected sets.
    freeze = {
        "phase": 11,
        "phase10_run_id": REQUIRED_PHASE10_RUN_ID,
        "phase10_acceptance_overlay_sha256": sha256_file(
            upstream["phase10_dir"] / "phase10_acceptance_overlay.json"
        ),
        "inner_fold_sha256": folds.content_sha256,
        "tracks": {},
        "candidate_results_sha256": canonical_json_sha256(candidate_results),
        "feature_importance_sha256": canonical_json_sha256(importance_stability),
        "family_ablation_sha256": canonical_json_sha256(family_results),
        "outer_validation_accessed": False,
        "test_target_accessed": False,
    }
    for track in TRACKS:
        chosen = selected[track]
        freeze["tracks"][track] = {
            "effective_parent": parents[track]["effective_parent_candidate_id"],
            "parent_feature_count": parents[track]["parent_feature_count"],
            "selected_candidate_id": chosen["candidate_id"],
            "selected_feature_count": chosen["feature_count"],
            "selected_feature_sha256": chosen["feature_set_sha256"],
            "selected_features": chosen["feature_list"],
            "parent_parameter_sha256": parents[track]["parent_parameter_sha256"],
            "selection_metrics": {
                key: chosen[key]
                for key in (
                    "mean_average_precision",
                    "min_average_precision",
                    "std_average_precision",
                    "mean_roc_auc",
                    "mean_log_loss",
                    "mean_brier_score",
                )
            },
            "selection_decision_trace": decision_trace[track],
        }
    freeze["selection_freeze_sha256"] = canonical_json_sha256(freeze)
    write_json(work_dir / "selection_freeze.json", freeze)
    # Explicitly load VALIDATION only after the freeze has been persisted.
    validation_targets, validation_audit = load_validation_targets_after_freeze(
        inputs, study_frozen=True
    )
    selected_predictions: list[pd.DataFrame] = []
    selected_model_entries: dict[str, Any] = {}
    validation_metrics: dict[str, Any] = {}
    models_dir = work_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for track in TRACKS:
        chosen = selected[track]
        spec = subset_feature_set(
            parent_specs[track], chosen["feature_list"], str(chosen["candidate_id"])
        )
        baseline = load_baseline_settings(root)
        from dataclasses import replace

        final_params = dict(parameters[track])
        final_params["thread_count"] = compute.single_fit_threads
        train_frame = train_matrix.drop(columns=[KEY])
        validation_frame = inputs.development.loc[
            inputs.development["split"] == "VALIDATION"
        ].sort_values(KEY, kind="mergesort")
        adapted_train = adapt_matrix(
            train_frame, spec, replace(baseline, catboost_parameters=final_params)
        )
        adapted_validation = adapt_matrix(
            validation_frame.drop(columns=[KEY]),
            spec,
            replace(baseline, catboost_parameters=final_params),
        )
        y_train = target_by_key.loc[train_matrix[KEY].tolist()].to_numpy(dtype="int8")
        y_validation = (
            validation_targets.set_index(KEY)[TARGET]
            .loc[validation_frame[KEY].tolist()]
            .to_numpy(dtype="int8")
        )
        model = CatBoostClassifier(**final_params)
        started = time.perf_counter()
        model.fit(build_pool(adapted_train, spec, y_train))
        training_seconds = time.perf_counter() - started
        probabilities = np.asarray(
            model.predict_proba(build_pool(adapted_validation, spec))[:, 1], dtype="float64"
        )
        candidate_id = str(chosen["candidate_id"])
        selected_predictions.append(
            pd.DataFrame(
                {
                    KEY: validation_frame[KEY].astype(int).to_numpy(),
                    "candidate_id": candidate_id,
                    "high_cost_probability": probabilities,
                }
            )
        )
        metric = metrics_for_predictions(y_validation, probabilities, 0.5)
        validation_metrics[candidate_id] = metric
        model_file = models_dir / f"selected_{track.lower()}.cbm"
        save_model(model, model_file)
        reloaded = CatBoostClassifier()
        reloaded.load_model(str(model_file), format="cbm")
        reproduced = np.asarray(
            reloaded.predict_proba(build_pool(adapted_validation, spec))[:, 1], dtype="float64"
        )
        if not np.allclose(probabilities, reproduced, rtol=0.0, atol=1e-10):
            raise FeatureSelectionError(f"{track} selected model reload probabilities differ.")
        selected_model_entries[track] = {
            "candidate_id": candidate_id,
            "track": track,
            "parent_candidate_id": parents[track]["effective_parent_candidate_id"],
            "feature_count": spec.feature_count,
            "feature_set_sha256": spec.feature_set_sha256,
            "feature_list_sha256": feature_list_sha256(spec.feature_names),
            "model_file": str(model_file.relative_to(work_dir)),
            "model_sha256": sha256_file(model_file),
            "statistical_parameters": {
                key: value for key, value in final_params.items() if key != "thread_count"
            },
            "execution_parameters": {"thread_count": compute.single_fit_threads},
            "training_seconds": training_seconds,
            "train_rows": len(train_matrix),
            "validation_rows": len(validation_frame),
            "reload_probability_max_abs_delta": float(np.max(np.abs(probabilities - reproduced))),
        }
        del model, reloaded
        gc.collect()
    predictions = (
        pd.concat(selected_predictions, ignore_index=True)[
            [KEY, "candidate_id", "high_cost_probability"]
        ]
        .sort_values(["candidate_id", KEY], kind="mergesort")
        .reset_index(drop=True)
    )
    if (
        len(predictions) != 2 * len(validation_targets)
        or predictions["candidate_id"].nunique() != 2
    ):
        raise FeatureSelectionError("Phase 11 validation prediction cardinality changed.")
    comparisons: dict[str, Any] = {}
    effective_candidates: dict[str, str] = {}
    for track in TRACKS:
        chosen_id = str(selected[track]["candidate_id"])
        parent_id = parents[track]["effective_parent_candidate_id"]
        parent_metrics = parents[track]["parent_validation_metrics"]
        selected_metric = validation_metrics[chosen_id]
        selected_row = {**selected_metric, "feature_count": selected[track]["feature_count"]}
        parent_row = {**parent_metrics, "feature_count": parents[track]["parent_feature_count"]}
        decision = replacement_decision(parent_row, selected_row, settings)
        comparisons[track] = {
            "track": track,
            "parent_candidate_id": parent_id,
            "selected_candidate_id": chosen_id,
            "parent_metrics": parent_row,
            "selected_metrics": selected_row,
            **decision,
        }
        effective_candidates[track] = chosen_id if decision["replace_parent"] else parent_id
    effective_outer = [
        {
            "track": track,
            "candidate_id": effective_candidates[track],
            "metrics": validation_metrics.get(
                effective_candidates[track], parents[track]["parent_validation_metrics"]
            ),
            "feature_count": selected[track]["feature_count"]
            if effective_candidates[track] == selected[track]["candidate_id"]
            else parents[track]["parent_feature_count"],
        }
        for track in TRACKS
    ]
    champion = sorted(
        effective_outer,
        key=lambda row: (
            -float(row["metrics"]["average_precision"]),
            -float(row["metrics"]["roc_auc"]),
            float(row["metrics"]["log_loss"]),
            int(row["feature_count"]),
            str(row["candidate_id"]),
        ),
    )[0]["candidate_id"]
    warnings = [
        "SYNTHETIC_POC",
        "BUSINESS_TARGET_UNCONFIRMED",
        "UNVERSIONED_FAILURE_DESCRIPTION_DIMENSION",
    ]
    if all(not bool(comparisons[track]["replace_parent"]) for track in TRACKS):
        warnings.append("NO_FEATURE_REDUCTION_GAIN")
    for track in TRACKS:
        if not comparisons[track]["replace_parent"]:
            warnings.append(f"{track}_FEATURE_SELECTION_REGRESSION")
    if any(float(row["std_average_precision"]) > 0.04 for row in candidate_results):
        warnings.append("FEATURE_IMPORTANCE_INSTABILITY")
    warnings = sorted(set(warnings))
    # Persist all final artifacts by writing into the work directory first.
    write_json(
        work_dir / "parent_model_manifest.json",
        {"phase": 11, "phase10_run_id": REQUIRED_PHASE10_RUN_ID, "tracks": parents},
    )
    write_json(work_dir / "parent_inner_cv_replay.json", parent_replay)
    write_parquet(
        pd.DataFrame(importance_by_fold).sort_values(
            ["track", "fold_id", "feature"], kind="mergesort"
        ),
        work_dir / "feature_importance_by_fold.parquet",
        compression="snappy",
    )
    write_parquet(
        pd.DataFrame(importance_stability).sort_values(["track", "feature"], kind="mergesort"),
        work_dir / "feature_importance_stability.parquet",
        compression="snappy",
    )
    write_parquet(
        pd.DataFrame(family_results).sort_values(["track", "family"], kind="mergesort"),
        work_dir / "family_ablation_results.parquet",
        compression="snappy",
    )
    write_json(work_dir / "candidate_feature_sets.json", candidate_payload)
    write_parquet(
        pd.DataFrame(candidate_results),
        work_dir / "candidate_inner_cv_results.parquet",
        compression="snappy",
    )
    write_parquet(
        pd.DataFrame(candidate_fold_metrics),
        work_dir / "candidate_fold_metrics.parquet",
        compression="snappy",
    )
    for track in TRACKS:
        write_json(
            work_dir / f"selected_features_{track.lower()}.json",
            {
                "track": track,
                "candidate_id": selected[track]["candidate_id"],
                "feature_set_sha256": selected[track]["feature_set_sha256"],
                "features": selected[track]["feature_list"],
            },
        )
    write_parquet(predictions, work_dir / "validation_predictions.parquet", compression="snappy")
    write_json(
        work_dir / "validation_metrics.json",
        {
            "primary_metric": "average_precision",
            "candidate_metrics": validation_metrics,
            "comparisons": comparisons,
            "effective_candidates": effective_candidates,
            "phase11_development_champion": champion,
            "selection_decision_trace": decision_trace,
        },
    )
    target_audit = {
        "train_target_rows_loaded": int(len(train_targets)),
        "validation_target_rows_loaded_before_selection_freeze": 0,
        "validation_target_rows_loaded_after_selection_freeze": int(len(validation_targets)),
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
        "selection_freeze_sha256": freeze["selection_freeze_sha256"],
        "outer_validation_accessed_before_selection_freeze": False,
        "outer_validation_accessed_after_selection_freeze": True,
    }
    write_json(work_dir / "target_access_audit.json", target_audit)
    write_json(
        work_dir / "compute_manifest.json",
        {
            "phase": 11,
            **settings_compute,
            "cli_overrides": {
                "max_workers": max_workers,
                "threads_per_fit": threads_per_fit,
                "single_fit_threads": single_fit_threads,
            },
        },
    )
    model_manifest = {
        "phase": 11,
        "models": selected_model_entries,
        "selected_candidates": selected,
        "effective_candidates": effective_candidates,
        "reload_validation": {
            track: selected_model_entries[track]["reload_probability_max_abs_delta"]
            for track in TRACKS
        },
    }
    write_json(work_dir / "model_manifest.json", model_manifest)
    validation = {
        "valid": True,
        "status": "PASS WITH WARNINGS" if warnings else "PASS",
        "hardening_status": "HARDENED_PASS",
        "errors": [],
        "warnings": warnings,
        "upstream": {
            "phase9_hardened_status": "HARDENED_PASS",
            "phase10_hardened_status": "HARDENED_PASS",
            "phase10_overlay_valid": True,
        },
        "selection_freeze": {
            "outer_validation_accessed": freeze["outer_validation_accessed"],
            "written_before_validation": True,
        },
        "candidate_count_by_track": {track: len(candidate_definitions[track]) for track in TRACKS},
        "test_seal": {
            "test_target_rows_loaded": 0,
            "test_predictions_created": 0,
            "test_metrics_computed": False,
            "test_target_access_allowed": False,
            "first_allowed_test_target_phase": 15,
        },
    }
    write_json(work_dir / "validation.json", validation)
    artifact_names = [
        path.name
        for path in work_dir.iterdir()
        if path.is_file() and path.name not in {"phase11_manifest.json"}
    ]
    phase11_manifest = {
        "phase": 11,
        "run_id": selected_run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit_sha": git_commit_sha(root),
        "contract_version": CONTRACT_VERSION,
        "contract_checksum": plan["contract"].get("contract_checksum"),
        "phase9_run_id": REQUIRED_PHASE9_RUN_ID,
        "phase10_run_id": REQUIRED_PHASE10_RUN_ID,
        "phase10_manifest_sha256": sha256_file(
            upstream["phase10_dir"] / "optimization_manifest.json"
        ),
        "phase10_acceptance_overlay_sha256": sha256_file(
            upstream["phase10_dir"] / "phase10_acceptance_overlay.json"
        ),
        "phase10_validation_sha256": sha256_file(upstream["phase10_dir"] / "validation.json"),
        "input_feature_hashes": REQUIRED_FEATURE_HASHES,
        "inner_fold_sha256": folds.content_sha256,
        "effective_parent_candidates": {
            track: parents[track]["effective_parent_candidate_id"] for track in TRACKS
        },
        "effective_parent_parameter_hashes": {
            track: parents[track]["parent_parameter_sha256"] for track in TRACKS
        },
        "compute_configuration": settings_compute,
        "candidate_inventory": {track: candidate_payload[track] for track in TRACKS},
        "selected_feature_set_hashes": {
            track: selected[track]["feature_set_sha256"] for track in TRACKS
        },
        "selection_freeze_sha256": freeze["selection_freeze_sha256"],
        "outer_validation_accessed": True,
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
        "phase11_development_champion": champion,
        "warnings": warnings,
        "artifact_file_sha256": {
            name: sha256_file(work_dir / name) for name in sorted(artifact_names)
        },
    }
    write_json(work_dir / "phase11_manifest.json", phase11_manifest)
    # Publish atomically only after every artifact has been written.
    if final_dir.exists():
        shutil.rmtree(final_dir)
    work_dir.replace(final_dir)
    report_directory = report_root / selected_run_id
    report_directory.mkdir(parents=True, exist_ok=True)
    _write_reports(
        report_directory,
        phase11_manifest,
        parents,
        group_manifest,
        candidate_results,
        comparisons,
        validation,
    )
    return {
        "status": validation["status"],
        "run_directory": final_dir,
        "report_directory": report_directory,
        "validation": validation,
        "phase11_development_champion": champion,
        "parents": parents,
        "selected": selected,
        "comparisons": comparisons,
        "compute_plan": compute.as_dict(),
    }


def _write_reports(  # pragma: no cover
    directory: Path,
    manifest: dict[str, Any],
    parents: dict[str, Any],
    groups: dict[str, Any],
    candidates: list[dict[str, Any]],
    comparisons: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    summary = {
        "phase": 11,
        "run_id": manifest["run_id"],
        "phase11_development_champion": manifest["phase11_development_champion"],
        "phase9_run_id": manifest["phase9_run_id"],
        "phase10_run_id": manifest["phase10_run_id"],
        "parent_feature_counts": {
            track: value["parent_feature_count"] for track, value in parents.items()
        },
        "number_of_feature_families": groups["family_count"],
        "number_of_family_ablation_experiments": groups["family_count"] * len(TRACKS),
        "number_of_subset_candidates": {
            track: sum(1 for row in candidates if row["track"] == track) for track in TRACKS
        },
        "replacement_decisions": comparisons,
        "warnings": manifest["warnings"],
        "test_seal": validation["test_seal"],
    }
    write_json(directory / "phase_11_summary.json", summary)
    write_json(
        directory / "parent_comparison.json", {track: comparisons[track] for track in TRACKS}
    )
    write_json(directory / "feature_family_summary.json", groups)
    write_json(
        directory / "feature_importance_summary.json",
        {"tracks": {track: "feature_importance_stability.parquet" for track in TRACKS}},
    )
    write_json(
        directory / "candidate_selection_summary.json",
        {"candidates": candidates, "comparisons": comparisons},
    )
    write_json(
        directory / "validation_metrics.json",
        {"comparisons": comparisons, "champion": manifest["phase11_development_champion"]},
    )
    write_json(directory / "validation.json", validation)
    lines = [
        f"# Phase 11 Feature Selection & Ablation — {manifest['run_id']}",
        "",
        f"Status: **{validation['status']}**",
        "",
        f"Development champion: `{manifest['phase11_development_champion']}`",
        "",
        "## Locked evidence",
        "",
        f"- Phase 9: `{manifest['phase9_run_id']}` (`HARDENED_PASS`)",
        f"- Phase 10: `{manifest['phase10_run_id']}` (`HARDENED_PASS` + acceptance overlay)",
        f"- Inner folds: `{manifest['inner_fold_sha256']}`",
        "- TEST rows: 0; TEST predictions: 0; TEST metrics: false",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in manifest["warnings"])
    lines.extend(
        [
            "",
            "This is a synthetic proof-of-concept; Phase 12 remains gated on this acceptance result and no TEST data was accessed.",
            "",
        ]
    )
    (directory / "phase_11_summary.md").write_text("\n".join(lines), encoding="utf-8")


def validate_existing_selection(  # pragma: no cover
    selection_dir: Path, *, project_root: Path | None = None
) -> dict[str, Any]:
    from .validation import validate_selection_directory

    return validate_selection_directory(selection_dir, project_root=project_root)

