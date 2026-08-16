"""Standalone fail-closed Phase 11 artifact validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from ..baseline_model.adapters import adapt_matrix
from ..baseline_model.catboost_baseline import build_pool, effective_parameters
from ..baseline_model.config import load_baseline_settings
from ..catboost_optimization.input import (
    load_locked_phase9_inputs,
    load_validation_targets_after_freeze,
)
from ..catboost_optimization.metrics import aggregate_fold_metrics, metrics_for_predictions
from ..catboost_optimization.provenance import (
    canonical_json_sha256,
    fold_content_sha256,
    sha256_file,
)
from ..paths import discover_repository_root
from .config import (
    TRACK_TO_EXPERIMENT,
    TRACKS,
    FeatureSelectionError,
    load_feature_selection_settings,
)
from .contract import REQUIRED_FEATURE_HASHES, validate_feature_selection_contract
from .grouping import validate_group_membership
from .runner import _read_json, _validate_upstream
from .selection import (
    feature_list_sha256,
    feature_set_sha256,
    replacement_decision,
    select_candidate,
    select_phase11_champion,
    subset_feature_set,
)

REQUIRED_FILES = (
    "phase11_manifest.json",
    "parent_model_manifest.json",
    "feature_group_manifest.json",
    "feature_group_membership.parquet",
    "parent_inner_cv_replay.json",
    "feature_importance_by_fold.parquet",
    "feature_importance_stability.parquet",
    "family_ablation_results.parquet",
    "candidate_feature_sets.json",
    "candidate_inner_cv_results.parquet",
    "candidate_fold_metrics.parquet",
    "selection_freeze.json",
    "selected_features_t1.json",
    "selected_features_t3.json",
    "validation_predictions.parquet",
    "validation_metrics.json",
    "model_manifest.json",
    "target_access_audit.json",
    "compute_manifest.json",
    "validation.json",
)

REQUIRED_VALIDATION_METRICS = (
    "average_precision",
    "roc_auc",
    "log_loss",
    "brier_score",
    "threshold",
)
LOCKED_MODEL_PARAMETERS = (
    "iterations",
    "learning_rate",
    "depth",
    "l2_leaf_reg",
    "random_strength",
    "bagging_temperature",
    "border_count",
    "rsm",
    "loss_function",
    "bootstrap_type",
    "random_seed",
    "task_type",
    "use_best_model",
)
FORBIDDEN_MODEL_PARAMETERS = {
    "class_weights",
    "auto_class_weights",
    "scale_pos_weight",
    "early_stopping_rounds",
    "eval_set",
    "od_type",
    "od_wait",
    "calibration",
    "threshold",
    "resampling",
}
DISABLED_MODEL_VALUES = (None, False, 0, 1, "", "none", "None", "false", "disabled")
MODEL_PARAMETER_TOLERANCE = 1.0e-6


def _error(errors: list[str], message: str) -> None:
    errors.append(message)


def _metric_values_match(
    expected: dict[str, Any], actual: dict[str, Any], label: str, errors: list[str]
) -> None:  # pragma: no cover
    """Compare persisted and independently recomputed metric payloads."""

    for key, value in expected.items():
        if key not in actual:
            _error(errors, f"{label} is missing metric {key}.")
            continue
        observed = actual[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not np.isclose(float(value), float(observed), rtol=0.0, atol=1e-10):
                _error(errors, f"{label}.{key} differs from recomputation.")
        elif value != observed:
            _error(errors, f"{label}.{key} differs from recomputation.")


def _require_metric_schema(
    expected: Any, label: str, errors: list[str]
) -> bool:  # pragma: no cover
    """Require the complete persisted outer metric schema before comparison."""

    if not isinstance(expected, dict):
        _error(errors, f"{label} metric payload is missing or not an object.")
        return False
    missing = [key for key in REQUIRED_VALIDATION_METRICS if key not in expected]
    if missing:
        _error(errors, f"{label} metric payload is missing: {', '.join(missing)}.")
        return False
    return True


def _artifact_path(directory: Path, value: Any) -> Path:  # pragma: no cover
    return directory / str(value).replace("\\", "/")


def _parameter_matches(expected: Any, actual: Any) -> bool:  # pragma: no cover
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return bool(
                np.isclose(float(actual), float(expected), rtol=0.0, atol=MODEL_PARAMETER_TOLERANCE)
            )
        except (TypeError, ValueError):
            return False
    return bool(actual == expected)


def _parameter_is_disabled(value: Any) -> bool:  # pragma: no cover
    if value in DISABLED_MODEL_VALUES:
        return True
    if isinstance(value, (list, tuple)):
        try:
            return all(float(item) == 1.0 for item in value)
        except (TypeError, ValueError):
            return False
    return False


def _validate_frozen_parameters(
    model_entry: dict[str, Any],
    parent_entry: dict[str, Any],
    expected_parent: dict[str, Any],
    label: str,
    errors: list[str],
) -> None:  # pragma: no cover
    """Validate manifest parameters against source truth, not another manifest."""

    statistical = model_entry.get("statistical_parameters")
    parent_statistical = parent_entry.get("statistical_parameters")
    if not isinstance(statistical, dict) or not isinstance(parent_statistical, dict):
        _error(errors, f"{label} statistical parameter manifest is incomplete.")
        return
    bad = sorted(str(key) for key in statistical if str(key).lower() in FORBIDDEN_MODEL_PARAMETERS)
    if bad:
        _error(errors, f"{label} contains forbidden statistical parameters: {', '.join(bad)}.")
    if "thread_count" in statistical:
        _error(errors, f"{label} thread_count leaked into statistical parameters.")
    if statistical != parent_statistical:
        _error(errors, f"{label} statistical CatBoost parameters drifted from its parent.")
    for key, value in expected_parent.items():
        if key in {"thread_count", "verbose"}:
            continue
        if key not in statistical or not _parameter_matches(value, statistical[key]):
            _error(errors, f"{label} frozen parameter {key} drifted from source truth.")


def _validate_actual_model_parameters(
    actual: Any, expected_parent: dict[str, Any], label: str, errors: list[str]
) -> None:  # pragma: no cover
    """Validate the effective parameters embedded in a serialized CatBoost model."""

    if not isinstance(actual, dict):
        _error(errors, f"{label} effective CatBoost parameters are missing.")
        return
    for key in LOCKED_MODEL_PARAMETERS:
        if key not in expected_parent:
            _error(errors, f"{label} source truth is missing locked parameter {key}.")
            continue
        if key not in actual or not _parameter_matches(expected_parent[key], actual[key]):
            _error(errors, f"{label} actual CatBoost parameter differs: {key}.")
    for key in FORBIDDEN_MODEL_PARAMETERS:
        if key in actual and not _parameter_is_disabled(actual[key]):
            _error(errors, f"{label} actual model enables prohibited parameter: {key}.")


def _resolve_parent_source(
    track: str,
    parent_id: str,
    upstream: dict[str, Any],
    phase10_model_manifest: dict[str, Any],
    root: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], str]:  # pragma: no cover
    """Resolve the permitted parent and reconstruct its parameters from source truth."""

    experiment = TRACK_TO_EXPERIMENT[track]
    if parent_id == f"P10_{track}_{experiment}_OPTIMIZED":
        source_dir = upstream["phase10_dir"]
        source_entry = phase10_model_manifest.get("models", {}).get(parent_id, {})
        if not isinstance(source_entry, dict) or not source_entry:
            raise FeatureSelectionError(f"{track} locked Phase 10 parent model is missing.")
        best_payload = _read_json(source_dir / "best_params.json").get(track, {})
        fixed = (
            _read_json(source_dir / "optimization_manifest.json")
            .get("settings", {})
            .get("fixed_parameters", {})
        )
        best = best_payload.get("best_params")
        parameter_hash = best_payload.get("best_param_sha256")
        if not isinstance(fixed, dict) or not isinstance(best, dict) or not parameter_hash:
            raise FeatureSelectionError(f"{track} Phase 10 frozen parameter source is incomplete.")
        expected = {str(key): value for key, value in {**fixed, **best}.items()}
        if source_entry.get("parameter_sha256") != parameter_hash:
            raise FeatureSelectionError(
                f"{track} Phase 10 parameter hash differs from source truth."
            )
        return source_dir, source_entry, expected, dict(expected), str(parameter_hash)
    if parent_id == f"P9_{experiment}_BASELINE":
        source_dir = upstream["phase9_dir"]
        source_manifest = _read_json(source_dir / "model_manifest.json")
        # Phase 9 manifests are keyed by E1/E3, not by Phase 11 candidate IDs.
        source_entry = source_manifest.get("models", {}).get(experiment, {})
        if not isinstance(source_entry, dict) or not source_entry:
            raise FeatureSelectionError(f"{track} locked Phase 9 parent model is missing.")
        baseline = load_baseline_settings(root)
        manifest_expected = {str(key): value for key, value in baseline.catboost_parameters.items()}
        # These are CatBoost CPU defaults when they are intentionally absent from
        # the Phase 9 baseline configuration, but they are present in get_all_params().
        actual_model_expected = dict(manifest_expected)
        actual_model_expected.setdefault("border_count", 254)
        actual_model_expected.setdefault("rsm", 1.0)
        return (
            source_dir,
            source_entry,
            manifest_expected,
            actual_model_expected,
            canonical_json_sha256(baseline.catboost_parameters),
        )
    raise FeatureSelectionError(f"{track} effective parent is not permitted: {parent_id}.")


def _load_validation_model_metrics(
    model_file: Path,
    feature_set: Any,
    validation_frame: pd.DataFrame,
    validation_targets: pd.DataFrame,
    baseline: Any,
) -> tuple[np.ndarray, dict[str, Any]]:  # pragma: no cover
    model = CatBoostClassifier()
    model.load_model(str(model_file), format="cbm")
    matrix = adapt_matrix(
        validation_frame.drop(columns=["warranty_claim_key"]), feature_set, baseline
    )
    probabilities = np.asarray(
        model.predict_proba(build_pool(matrix, feature_set))[:, 1], dtype="float64"
    )
    y_true = (
        validation_targets.set_index("warranty_claim_key")["target__high_cost_claim_flag"]
        .loc[validation_frame["warranty_claim_key"].tolist()]
        .to_numpy(dtype="int8")
    )
    return probabilities, metrics_for_predictions(y_true, probabilities, 0.5)


def validate_selection_directory(  # pragma: no cover
    selection_dir: Path, *, project_root: Path | None = None
) -> dict[str, Any]:
    """Validate a published run from persisted evidence, without rerunning search."""

    root = discover_repository_root(project_root)
    directory = selection_dir.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    recomputation: dict[str, Any] = {}
    try:
        missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
        if missing:
            raise FeatureSelectionError("Phase 11 artifacts missing: " + ", ".join(missing))
        manifest = _read_json(directory / "phase11_manifest.json")
        if manifest.get("phase") != 11:
            raise FeatureSelectionError("Phase 11 manifest phase is not 11.")
        contract = validate_feature_selection_contract(root)
        if not contract.get("valid"):
            raise FeatureSelectionError(
                "Phase 11 contract invalid: " + "; ".join(contract.get("errors", []))
            )
        if manifest.get("contract_checksum") != contract.get("contract_checksum"):
            raise FeatureSelectionError(
                "Phase 11 contract checksum differs from current repository."
            )
        settings = load_feature_selection_settings(root)
        upstream = _validate_upstream(
            root / "artifacts" / "catboost_optimization" / "20260811T_PHASE10", root
        )
        if (
            manifest.get("phase10_run_id") != "20260811T_PHASE10"
            or manifest.get("phase9_run_id") != "20260811T_PHASE9_FINAL"
        ):
            raise FeatureSelectionError("Phase 11 upstream run ids drifted.")
        if manifest.get("phase10_manifest_sha256") != sha256_file(
            upstream["phase10_dir"] / "optimization_manifest.json"
        ):
            raise FeatureSelectionError("Phase 11 Phase 10 manifest hash drifted.")
        if manifest.get("phase10_acceptance_overlay_sha256") != sha256_file(
            upstream["phase10_dir"] / "phase10_acceptance_overlay.json"
        ):
            raise FeatureSelectionError("Phase 11 Phase 10 overlay hash drifted.")
        for name, digest in manifest.get("artifact_file_sha256", {}).items():
            if name == "phase11_manifest.json":
                continue
            path = directory / str(name)
            if not path.is_file() or sha256_file(path) != str(digest):
                raise FeatureSelectionError(f"Phase 11 artifact hash mismatch: {name}")

        inputs = load_locked_phase9_inputs(upstream["phase9_dir"], project_root=root)
        parent_features = {
            track: inputs.feature_sets[TRACK_TO_EXPERIMENT[track]].feature_names for track in TRACKS
        }
        if manifest.get("input_feature_hashes") != REQUIRED_FEATURE_HASHES:
            raise FeatureSelectionError("Phase 11 parent feature hashes drifted.")
        membership = pd.read_parquet(directory / "feature_group_membership.parquet")
        validate_group_membership(
            membership, set(name for values in parent_features.values() for name in values)
        )
        family_by_feature = {str(row.feature): str(row.family) for row in membership.itertuples()}

        candidates = json.loads(
            (directory / "candidate_feature_sets.json").read_text(encoding="utf-8")
        )
        candidate_rows: list[dict[str, Any]] = []
        candidate_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        canonical_feature_hashes: dict[tuple[str, str], str] = {}
        for track in TRACKS:
            inventory = candidates.get(track)
            if not isinstance(inventory, list) or not inventory or len(inventory) > 8:
                raise FeatureSelectionError(f"{track} candidate inventory is invalid.")
            parent = parent_features[track]
            for candidate in inventory:
                if not isinstance(candidate, dict):
                    raise FeatureSelectionError(f"{track} candidate definition is invalid.")
                candidate_id = str(candidate.get("candidate_id"))
                features = tuple(str(value) for value in candidate.get("feature_list", []))
                if not features or len(features) != len(set(features)):
                    raise FeatureSelectionError(f"{track} candidate feature list is not unique.")
                if not set(features).issubset(parent):
                    raise FeatureSelectionError(
                        f"{track} candidate adds a feature outside its parent."
                    )
                if len(features) < settings.minimum_feature_count or len(features) != int(
                    candidate.get("feature_count", -1)
                ):
                    raise FeatureSelectionError(f"{track} candidate feature count is invalid.")
                if feature_set_sha256(candidate_id, features) != candidate.get(
                    "feature_set_sha256"
                ):
                    raise FeatureSelectionError(f"{track} candidate feature hash is invalid.")
                canonical = feature_list_sha256(features)
                canonical_key = (track, canonical)
                if canonical_key in canonical_feature_hashes:
                    warnings.append(
                        f"{track}_LEGACY_DUPLICATE_CANDIDATE_FEATURE_LIST_"
                        f"{canonical_feature_hashes[canonical_key]}_{candidate_id}"
                    )
                else:
                    canonical_feature_hashes[canonical_key] = candidate_id
                row = {"track": track, **candidate, "feature_list": list(features)}
                candidate_rows.append(row)
                candidate_lookup[(track, candidate_id)] = row

        freeze = _read_json(directory / "selection_freeze.json")
        if (
            freeze.get("phase") != 11
            or freeze.get("outer_validation_accessed") is not False
            or freeze.get("test_target_accessed") is not False
        ):
            raise FeatureSelectionError("Selection freeze is missing the pre-validation gate.")
        if freeze.get("inner_fold_sha256") != manifest.get("inner_fold_sha256"):
            raise FeatureSelectionError("Selection freeze inner-fold hash drifted.")
        freeze_without_hash = dict(freeze)
        freeze_without_hash.pop("selection_freeze_sha256", None)
        if canonical_json_sha256(freeze_without_hash) != freeze.get("selection_freeze_sha256"):
            raise FeatureSelectionError("Selection freeze checksum does not reproduce.")

        fold_table = pd.read_parquet(upstream["phase10_dir"] / "inner_cv_folds.parquet")
        if fold_content_sha256(fold_table) != manifest.get("inner_fold_sha256"):
            raise FeatureSelectionError("Phase 11 did not reuse the exact Phase 10 inner folds.")
        replay = _read_json(directory / "parent_inner_cv_replay.json")
        if any(item.get("status") != "PASS" for item in replay.values()):
            raise FeatureSelectionError("Parent replay gate did not pass.")

        importance_by_fold = pd.read_parquet(directory / "feature_importance_by_fold.parquet")
        importance_stability = pd.read_parquet(directory / "feature_importance_stability.parquet")
        importance_schemas = (
            (
                importance_by_fold,
                {
                    "track",
                    "feature",
                    "family",
                    "fold_id",
                    "loss_function_change",
                    "loss_rank",
                    "loss_percentile_rank",
                    "mean_abs_shap",
                    "shap_rank",
                    "shap_percentile_rank",
                },
            ),
            (
                importance_stability,
                {
                    "track",
                    "feature",
                    "family",
                    "median_loss_percentile",
                    "median_shap_percentile",
                    "top_25_percent_fold_count",
                    "top_50_percent_fold_count",
                    "loss_rank_std",
                    "shap_rank_std",
                    "stable_score",
                },
            ),
        )
        for frame, columns in importance_schemas:
            if not columns.issubset(frame.columns):
                raise FeatureSelectionError("Phase 11 feature importance schema is incomplete.")
        if importance_stability.duplicated(["track", "feature"]).any():
            raise FeatureSelectionError(
                "Phase 11 feature importance stability contains duplicates."
            )

        family = pd.read_parquet(directory / "family_ablation_results.parquet")
        if not {
            "track",
            "family",
            "removed_feature_count",
            "remaining_feature_count",
            "delta_ap_vs_parent",
        }.issubset(family.columns):
            raise FeatureSelectionError("Phase 11 family ablation schema is incomplete.")
        for track in TRACKS:
            expected_families = {family_by_feature[name] for name in parent_features[track]}
            actual_rows = family.loc[family["track"].astype(str) == track]
            actual_families = set(actual_rows["family"].astype(str))
            missing_families = expected_families - actual_families
            if missing_families:
                raise FeatureSelectionError(
                    f"{track} family ablation is missing: {', '.join(sorted(missing_families))}."
                )
            extra_families = actual_families - expected_families
            if extra_families:
                legacy = actual_rows.loc[actual_rows["family"].isin(extra_families)]
                if (pd.to_numeric(legacy["removed_feature_count"], errors="coerce") != 0).any():
                    raise FeatureSelectionError(
                        f"{track} family ablation removes a family absent from the track."
                    )
                warnings.append(f"{track}_LEGACY_NOOP_FAMILY_ABLATION_ROWS_IGNORED")

        candidate_fold_metrics = pd.read_parquet(directory / "candidate_fold_metrics.parquet")
        candidate_inner = pd.read_parquet(directory / "candidate_inner_cv_results.parquet")
        required_fold_columns = {
            "track",
            "candidate_id",
            "fold_id",
            "average_precision",
            "roc_auc",
            "log_loss",
            "brier_score",
            "training_seconds",
        }
        required_inner_columns = {
            "track",
            "candidate_id",
            "feature_count",
            "feature_set_sha256",
            "feature_list",
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
        }
        if not required_fold_columns.issubset(candidate_fold_metrics.columns):
            raise FeatureSelectionError("Phase 11 candidate fold metric schema is incomplete.")
        if not required_inner_columns.issubset(candidate_inner.columns):
            raise FeatureSelectionError("Phase 11 candidate aggregate schema is incomplete.")
        if len(candidate_inner) != len(candidate_rows):
            raise FeatureSelectionError("Phase 11 candidate aggregate cardinality changed.")
        aggregate_rows: list[dict[str, Any]] = []
        aggregate_fields = (
            "mean_average_precision",
            "min_average_precision",
            "max_average_precision",
            "std_average_precision",
            "mean_roc_auc",
            "min_roc_auc",
            "mean_log_loss",
            "mean_brier_score",
            "fold_count",
        )
        for candidate in candidate_rows:
            track = str(candidate["track"])
            candidate_id = str(candidate["candidate_id"])
            fold_rows = candidate_fold_metrics.loc[
                (candidate_fold_metrics["track"].astype(str) == track)
                & (candidate_fold_metrics["candidate_id"].astype(str) == candidate_id)
            ].sort_values("fold_id", kind="mergesort")
            if list(fold_rows["fold_id"].astype(int)) != [1, 2, 3]:
                raise FeatureSelectionError(f"{track}/{candidate_id} must contain folds 1, 2, 3.")
            if fold_rows.duplicated("fold_id").any():
                raise FeatureSelectionError(f"{track}/{candidate_id} has duplicate fold metrics.")
            aggregate = aggregate_fold_metrics(fold_rows.to_dict("records"))
            aggregate["training_seconds"] = float(fold_rows["training_seconds"].sum())
            inner_match = candidate_inner.loc[
                (candidate_inner["track"].astype(str) == track)
                & (candidate_inner["candidate_id"].astype(str) == candidate_id)
            ]
            if len(inner_match) != 1:
                raise FeatureSelectionError(f"{track}/{candidate_id} aggregate row is missing.")
            persisted = inner_match.iloc[0].to_dict()
            for field in aggregate_fields + ("training_seconds",):
                if field == "fold_count":
                    if int(persisted[field]) != int(aggregate[field]):
                        raise FeatureSelectionError(f"{track}/{candidate_id} fold count differs.")
                elif not np.isclose(
                    float(persisted[field]), float(aggregate[field]), rtol=0.0, atol=1e-10
                ):
                    raise FeatureSelectionError(
                        f"{track}/{candidate_id} aggregate {field} differs from fold evidence."
                    )
            if list(persisted["feature_list"]) != list(candidate["feature_list"]):
                raise FeatureSelectionError(f"{track}/{candidate_id} feature list drifted.")
            aggregate_row = {**candidate, **aggregate}
            aggregate_rows.append(aggregate_row)
        aggregate_rows.sort(key=lambda row: (str(row["track"]), str(row["candidate_id"])))
        if canonical_json_sha256(aggregate_rows) != freeze.get("candidate_results_sha256"):
            raise FeatureSelectionError("Candidate aggregate evidence checksum does not reproduce.")
        recomputation["candidate_aggregates"] = {
            track: len([row for row in aggregate_rows if row["track"] == track]) for track in TRACKS
        }

        selected: dict[str, dict[str, Any]] = {}
        decision_trace: dict[str, dict[str, Any]] = {}
        for track in TRACKS:
            unique_rows: list[dict[str, Any]] = []
            seen_feature_lists: set[str] = set()
            for row in aggregate_rows:
                if row["track"] != track:
                    continue
                digest = feature_list_sha256(row["feature_list"])
                if digest in seen_feature_lists:
                    continue
                seen_feature_lists.add(digest)
                unique_rows.append(row)
            selected[track], decision_trace[track] = select_candidate(unique_rows, settings)
            frozen = freeze.get("tracks", {}).get(track, {})
            if selected[track]["candidate_id"] != frozen.get("selected_candidate_id"):
                raise FeatureSelectionError(f"{track} selection freeze does not reproduce.")
            if tuple(selected[track]["feature_list"]) != tuple(
                str(value) for value in frozen.get("selected_features", [])
            ):
                raise FeatureSelectionError(f"{track} frozen selected feature list drifted.")
            if feature_set_sha256(
                str(frozen.get("selected_candidate_id")), frozen.get("selected_features", [])
            ) != frozen.get("selected_feature_sha256"):
                raise FeatureSelectionError(f"{track} selected feature freeze hash is invalid.")
            if decision_trace[track] != frozen.get("selection_decision_trace"):
                raise FeatureSelectionError(f"{track} selection decision trace does not reproduce.")
        recomputation["selection"] = {track: selected[track]["candidate_id"] for track in TRACKS}

        audit = _read_json(directory / "target_access_audit.json")
        expected_audit = {
            "test_target_rows_loaded": 0,
            "test_predictions_created": 0,
            "test_metrics_computed": False,
            "test_target_access_allowed": False,
            "first_allowed_test_target_phase": 15,
        }
        if (
            any(audit.get(key) != value for key, value in expected_audit.items())
            or audit.get("train_target_rows_loaded") != 5952
            or audit.get("validation_target_rows_loaded_before_selection_freeze") != 0
            or audit.get("validation_target_rows_loaded_after_selection_freeze") != 1275
        ):
            raise FeatureSelectionError("Phase 11 target access audit is not sealed.")

        predictions = pd.read_parquet(directory / "validation_predictions.parquet")
        if (
            list(predictions.columns)
            != ["warranty_claim_key", "candidate_id", "high_cost_probability"]
            or len(predictions) != 2550
            or predictions["candidate_id"].nunique() != 2
        ):
            raise FeatureSelectionError(
                "Phase 11 outer validation prediction schema/cardinality changed."
            )
        if predictions.duplicated(["warranty_claim_key", "candidate_id"]).any():
            raise FeatureSelectionError("Phase 11 outer validation predictions contain duplicates.")

        model_manifest = _read_json(directory / "model_manifest.json")
        parent_manifest = _read_json(directory / "parent_model_manifest.json")
        phase10_model_manifest = _read_json(upstream["phase10_dir"] / "model_manifest.json")
        phase10_metrics = _read_json(upstream["phase10_dir"] / "validation_metrics.json")
        validation_metrics = _read_json(directory / "validation_metrics.json")
        baseline = load_baseline_settings(root)
        validation_targets, _ = load_validation_targets_after_freeze(inputs, study_frozen=True)
        validation_frame = inputs.development.loc[
            inputs.development["split"] == "VALIDATION"
        ].sort_values("warranty_claim_key", kind="mergesort")
        parent_metrics_by_track: dict[str, dict[str, Any]] = {}
        selected_metrics_by_track: dict[str, dict[str, Any]] = {}
        effective_candidates: dict[str, str] = {}
        for track in TRACKS:
            parent_entry = parent_manifest.get("tracks", {}).get(track, {})
            selected_entry = model_manifest.get("models", {}).get(track, {})
            if not parent_entry or not selected_entry:
                raise FeatureSelectionError(f"{track} model manifest entry is missing.")
            parent_id = str(parent_entry.get("effective_parent_candidate_id"))
            (
                source_dir,
                source_entry,
                manifest_expected_parent,
                actual_model_expected_parent,
                expected_parameter_hash,
            ) = _resolve_parent_source(track, parent_id, upstream, phase10_model_manifest, root)
            if parent_entry.get("parent_parameter_sha256") != expected_parameter_hash:
                raise FeatureSelectionError(
                    f"{track} parent parameter hash drifted from source truth."
                )
            _validate_frozen_parameters(
                selected_entry, parent_entry, manifest_expected_parent, track, errors
            )
            selected_model_file = _artifact_path(directory, selected_entry.get("model_file"))
            if not selected_model_file.is_file() or sha256_file(
                selected_model_file
            ) != selected_entry.get("model_sha256"):
                raise FeatureSelectionError(f"{track} selected model hash mismatch.")
            candidate = selected[track]
            if selected_entry.get("candidate_id") != candidate["candidate_id"]:
                raise FeatureSelectionError(
                    f"{track} selected model candidate differs from freeze."
                )
            if selected_entry.get("feature_set_sha256") != candidate["feature_set_sha256"]:
                raise FeatureSelectionError(f"{track} model feature hash differs from freeze.")
            if selected_entry.get("feature_list_sha256") != feature_list_sha256(
                candidate["feature_list"]
            ):
                raise FeatureSelectionError(f"{track} model canonical feature hash differs.")
            if selected_entry.get("feature_count") != candidate["feature_count"]:
                raise FeatureSelectionError(f"{track} model feature count differs from freeze.")

            parent_model_file = source_dir / str(source_entry.get("model_file")).replace("\\", "/")
            if not parent_model_file.is_file() or sha256_file(
                parent_model_file
            ) != source_entry.get("model_sha256"):
                raise FeatureSelectionError(f"{track} locked parent model hash mismatch.")

            parent_model = CatBoostClassifier()
            parent_model.load_model(str(parent_model_file), format="cbm")
            _validate_actual_model_parameters(
                effective_parameters(parent_model),
                actual_model_expected_parent,
                f"{track} parent model",
                errors,
            )
            del parent_model

            selected_model = CatBoostClassifier()
            selected_model.load_model(str(selected_model_file), format="cbm")
            _validate_actual_model_parameters(
                effective_parameters(selected_model),
                actual_model_expected_parent,
                f"{track} selected model",
                errors,
            )
            del selected_model

            parent_spec = inputs.feature_sets[TRACK_TO_EXPERIMENT[track]]
            parent_probabilities, parent_metrics = _load_validation_model_metrics(
                parent_model_file,
                parent_spec,
                validation_frame,
                validation_targets,
                baseline,
            )
            del parent_probabilities
            selected_spec = subset_feature_set(
                parent_spec, candidate["feature_list"], str(candidate["candidate_id"])
            )
            selected_probabilities, selected_metrics = _load_validation_model_metrics(
                selected_model_file,
                selected_spec,
                validation_frame,
                validation_targets,
                baseline,
            )
            expected_predictions = predictions.loc[
                predictions["candidate_id"].astype(str) == str(candidate["candidate_id"])
            ].sort_values("warranty_claim_key", kind="mergesort")
            if list(expected_predictions["warranty_claim_key"].astype(int)) != list(
                validation_frame["warranty_claim_key"].astype(int)
            ):
                raise FeatureSelectionError(f"{track} validation prediction keys drifted.")
            observed = expected_predictions["high_cost_probability"].to_numpy(dtype="float64")
            if not np.allclose(observed, selected_probabilities, rtol=0.0, atol=1e-10):
                raise FeatureSelectionError(
                    f"{track} selected model probabilities do not reproduce."
                )
            selected_persisted_metrics = validation_metrics.get("candidate_metrics", {}).get(
                candidate["candidate_id"], {}
            )
            parent_persisted_metrics = parent_entry.get("parent_validation_metrics", {})
            phase10_parent_metrics = phase10_metrics.get("candidate_metrics", {}).get(parent_id, {})
            if _require_metric_schema(
                selected_persisted_metrics, f"{track} selected validation", errors
            ):
                _metric_values_match(
                    selected_persisted_metrics,
                    selected_metrics,
                    f"{track} selected validation metrics",
                    errors,
                )
            if _require_metric_schema(
                parent_persisted_metrics, f"{track} parent validation", errors
            ):
                _metric_values_match(
                    parent_persisted_metrics,
                    parent_metrics,
                    f"{track} parent validation metrics",
                    errors,
                )
            if _require_metric_schema(
                phase10_parent_metrics, f"{track} Phase 10 parent validation", errors
            ):
                _metric_values_match(
                    phase10_parent_metrics,
                    parent_metrics,
                    f"{track} Phase 10 parent validation metrics",
                    errors,
                )
            parent_metrics_by_track[track] = {
                **parent_metrics,
                "feature_count": parent_entry.get("parent_feature_count"),
            }
            selected_metrics_by_track[track] = {
                **selected_metrics,
                "feature_count": candidate["feature_count"],
            }
            selected_model_delta = float(
                selected_entry.get("reload_probability_max_abs_delta", 1.0)
            )
            if selected_model_delta > 1e-10:
                raise FeatureSelectionError(f"{track} selected model reload delta exceeds 1e-10.")
            decision = replacement_decision(
                parent_metrics_by_track[track], selected_metrics_by_track[track], settings
            )
            persisted_comparison = validation_metrics.get("comparisons", {}).get(track, {})
            for key in (
                "replace_parent",
                "reason",
                "average_precision_delta",
                "feature_reduction_fraction",
                "complexity_tradeoff_eligible",
                "ap_improvement_eligible",
            ):
                if key in decision and key in persisted_comparison:
                    if isinstance(decision[key], float):
                        if not np.isclose(
                            float(decision[key]),
                            float(persisted_comparison[key]),
                            rtol=0.0,
                            atol=1e-10,
                        ):
                            _error(errors, f"{track} replacement decision {key} differs.")
                    elif decision[key] != persisted_comparison[key]:
                        _error(errors, f"{track} replacement decision {key} differs.")
                else:
                    _error(errors, f"{track} replacement decision is incomplete.")
            effective_candidates[track] = (
                candidate["candidate_id"] if decision["replace_parent"] else parent_id
            )
        if validation_metrics.get("effective_candidates") != effective_candidates:
            raise FeatureSelectionError("Effective candidate mapping does not reproduce.")
        if model_manifest.get("effective_candidates") != effective_candidates:
            raise FeatureSelectionError("Model manifest effective candidates do not reproduce.")
        effective_outer = []
        for track in TRACKS:
            candidate_id = effective_candidates[track]
            metrics = (
                selected_metrics_by_track[track]
                if candidate_id == selected[track]["candidate_id"]
                else parent_metrics_by_track[track]
            )
            effective_outer.append(
                {
                    "track": track,
                    "candidate_id": candidate_id,
                    "metrics": metrics,
                    "feature_count": metrics["feature_count"],
                    "parent_candidate_id": parent_manifest["tracks"][track][
                        "effective_parent_candidate_id"
                    ],
                    "is_parent_candidate": candidate_id
                    == parent_manifest["tracks"][track]["effective_parent_candidate_id"],
                }
            )
        champion = select_phase11_champion(effective_outer)["candidate_id"]
        if champion != validation_metrics.get("phase11_development_champion"):
            raise FeatureSelectionError("Phase 11 development champion does not reproduce.")
        if champion != manifest.get("phase11_development_champion"):
            raise FeatureSelectionError("Phase 11 manifest champion does not reproduce.")
        recomputation["outer_validation"] = {
            "validation_rows": len(validation_frame),
            "selected_metrics": selected_metrics_by_track,
            "effective_candidates": effective_candidates,
            "champion": champion,
        }

        validation = _read_json(directory / "validation.json")
        if (
            validation.get("hardening_status") != "HARDENED_PASS"
            or validation.get("valid") is not True
        ):
            raise FeatureSelectionError("Phase 11 validation status is not HARDENED_PASS.")
        if validation.get("candidate_count_by_track") != {
            track: len([row for row in candidate_rows if row["track"] == track]) for track in TRACKS
        }:
            raise FeatureSelectionError("Phase 11 validation candidate counts do not reproduce.")
    except Exception as exc:
        _error(errors, str(exc))
    status = "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS")
    return {
        "valid": not errors,
        "status": status,
        "hardening_status": "HARDENED_PASS" if not errors else "BLOCKED",
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "independent_recomputation": recomputation,
    }


__all__ = ["validate_selection_directory"]
