"""Standalone fail-closed Phase 11 artifact validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..catboost_optimization.provenance import fold_content_sha256, sha256_file
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
from .selection import feature_set_sha256

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


def _error(errors: list[str], message: str) -> None:
    errors.append(message)


def validate_selection_directory(  # pragma: no cover
    selection_dir: Path, *, project_root: Path | None = None
) -> dict[str, Any]:
    root = discover_repository_root(project_root)
    directory = selection_dir.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
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
        settings = load_feature_selection_settings(root)
        inputs = __import__(
            "warranty_analytics_model.catboost_optimization.input",
            fromlist=["load_locked_phase9_inputs"],
        ).load_locked_phase9_inputs(upstream["phase9_dir"], project_root=root)
        parent_features = {
            track: inputs.feature_sets[TRACK_TO_EXPERIMENT[track]].feature_names for track in TRACKS
        }
        if manifest.get("input_feature_hashes") != REQUIRED_FEATURE_HASHES:
            raise FeatureSelectionError("Phase 11 parent feature hashes drifted.")
        membership = pd.read_parquet(directory / "feature_group_membership.parquet")
        validate_group_membership(
            membership, set(name for values in parent_features.values() for name in values)
        )
        candidates = json.loads(
            (directory / "candidate_feature_sets.json").read_text(encoding="utf-8")
        )
        candidate_rows: list[dict[str, Any]] = []
        for track in TRACKS:
            if (
                not isinstance(candidates.get(track), list)
                or not candidates[track]
                or len(candidates[track]) > 8
            ):
                raise FeatureSelectionError(f"{track} candidate inventory is invalid.")
            parent = parent_features[track]
            for candidate in candidates[track]:
                features = tuple(str(value) for value in candidate.get("feature_list", []))
                if not set(features).issubset(parent):
                    raise FeatureSelectionError(
                        f"{track} candidate adds a feature outside its parent."
                    )
                if len(features) < settings.minimum_feature_count or len(features) != candidate.get(
                    "feature_count"
                ):
                    raise FeatureSelectionError(f"{track} candidate feature count is invalid.")
                if feature_set_sha256(str(candidate["candidate_id"]), features) != candidate.get(
                    "feature_set_sha256"
                ):
                    raise FeatureSelectionError(f"{track} candidate feature hash is invalid.")
                candidate_rows.append({"track": track, **candidate})
        if len({str(row["feature_set_sha256"]) for row in candidate_rows}) != len(candidate_rows):
            raise FeatureSelectionError("Phase 11 candidate feature sets are not unique.")
        freeze = _read_json(directory / "selection_freeze.json")
        if (
            freeze.get("phase") != 11
            or freeze.get("outer_validation_accessed") is not False
            or freeze.get("test_target_accessed") is not False
        ):
            raise FeatureSelectionError("Selection freeze is missing the pre-validation gate.")
        if freeze.get("inner_fold_sha256") != manifest.get("inner_fold_sha256"):
            raise FeatureSelectionError("Selection freeze inner-fold hash drifted.")
        for track in TRACKS:
            entry = freeze.get("tracks", {}).get(track, {})
            selected = tuple(str(value) for value in entry.get("selected_features", []))
            if not set(selected).issubset(parent_features[track]) or feature_set_sha256(
                str(entry.get("selected_candidate_id")), selected
            ) != entry.get("selected_feature_sha256"):
                raise FeatureSelectionError(f"{track} selected feature freeze is invalid.")
        fold_table = pd.read_parquet(upstream["phase10_dir"] / "inner_cv_folds.parquet")
        if fold_content_sha256(fold_table) != manifest.get("inner_fold_sha256"):
            raise FeatureSelectionError("Phase 11 did not reuse the exact Phase 10 inner folds.")
        replay = _read_json(directory / "parent_inner_cv_replay.json")
        if any(item.get("status") != "PASS" for item in replay.values()):
            raise FeatureSelectionError("Parent replay gate did not pass.")
        importance_by_fold = pd.read_parquet(directory / "feature_importance_by_fold.parquet")
        importance_stability = pd.read_parquet(directory / "feature_importance_stability.parquet")
        for frame, columns in (
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
        ):
            if not columns.issubset(frame.columns):
                raise FeatureSelectionError("Phase 11 feature importance schema is incomplete.")
            if "fold_id" not in frame.columns and frame.duplicated(["track", "feature"]).any():
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
        for track in TRACKS:
            entry = model_manifest.get("models", {}).get(track, {})
            model_file = directory / str(entry.get("model_file"))
            if not model_file.is_file() or sha256_file(model_file) != entry.get("model_sha256"):
                raise FeatureSelectionError(f"{track} selected model hash mismatch.")
            if (
                entry.get("feature_set_sha256")
                != freeze["tracks"][track]["selected_feature_sha256"]
            ):
                raise FeatureSelectionError(f"{track} model feature hash differs from freeze.")
        validation = _read_json(directory / "validation.json")
        if (
            validation.get("hardening_status") != "HARDENED_PASS"
            or validation.get("valid") is not True
        ):
            raise FeatureSelectionError("Phase 11 validation status is not HARDENED_PASS.")
    except Exception as exc:
        _error(errors, str(exc))
    status = "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS")
    return {
        "valid": not errors,
        "status": status,
        "hardening_status": "HARDENED_PASS" if not errors else "BLOCKED",
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
    }


__all__ = ["validate_selection_directory"]

