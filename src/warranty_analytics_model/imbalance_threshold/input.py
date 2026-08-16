"""Locked Phase 11 parent resolution and exact Phase 10 fold reconstruction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..baseline_model.catboost_baseline import effective_parameters, load_model
from ..baseline_model.models import FeatureSetSpec, Phase9Inputs
from ..catboost_optimization.inner_folds import DATE, KEY, build_inner_fold_plan
from ..catboost_optimization.input import (
    CLAIM_DATE,
    load_locked_phase9_inputs,
    load_train_targets_for_optimization,
)
from ..catboost_optimization.models import InnerFold, InnerFoldPlan, Phase10Inputs
from ..catboost_optimization.provenance import (
    canonical_json_sha256,
    fold_content_sha256,
    sha256_file,
)
from ..feature_mart.manifest import write_json
from ..feature_selection.runner import validate_existing_selection
from ..paths import discover_repository_root
from .config import TRACKS, ImbalanceThresholdError

TARGET = "target__high_cost_claim_flag"
PHASE11_RUN_ID = "20260816T_PHASE11"
PHASE10_RUN_ID = "20260811T_PHASE10"
PHASE9_RUN_ID = "20260811T_PHASE9_FINAL"


@dataclass(frozen=True, slots=True)
class ParentResolution:
    track: str
    effective_candidate_id: str
    source_phase: int
    source_model: Path
    source_model_sha256: str
    feature_set: FeatureSetSpec
    feature_list_sha256: str
    parent_parameter_sha256: str
    statistical_parameters: dict[str, Any]
    phase11_parent_candidate_id: str
    selected_feature_candidate_id: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "track": self.track,
            "effective_candidate_id": self.effective_candidate_id,
            "source_phase": self.source_phase,
            "source_model": str(self.source_model),
            "source_model_sha256": self.source_model_sha256,
            "feature_count": self.feature_set.feature_count,
            "feature_set_sha256": self.feature_set.feature_set_sha256,
            "feature_list_sha256": self.feature_list_sha256,
            "features": list(self.feature_set.feature_names),
            "parent_parameter_sha256": self.parent_parameter_sha256,
            "statistical_parameters": self.statistical_parameters,
            "phase11_parent_candidate_id": self.phase11_parent_candidate_id,
            "selected_feature_candidate_id": self.selected_feature_candidate_id,
        }


@dataclass(frozen=True, slots=True)
class Phase12Inputs:
    root: Path
    phase11_dir: Path
    phase11_manifest: dict[str, Any]
    phase11_validation: dict[str, Any]
    phase11_model_manifest: dict[str, Any]
    phase11_parent_model_manifest: dict[str, Any]
    phase11_manifest_sha256: str
    phase11_validation_sha256: str
    phase11_model_manifest_sha256: str
    phase10_dir: Path
    phase9_dir: Path
    phase9_inputs: Phase9Inputs
    phase10_inputs: Phase10Inputs
    development: pd.DataFrame
    parents: dict[str, ParentResolution]
    fold_plan: InnerFoldPlan


def _read_json(path: Path) -> dict[str, Any]:  # pragma: no cover
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ImbalanceThresholdError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ImbalanceThresholdError(f"JSON artifact must be an object: {path}")
    return payload


def _resolve(root: Path, value: str | Path) -> Path:  # pragma: no cover
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_test_seal(payload: dict[str, Any], label: str) -> None:  # pragma: no cover
    expected = {
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ImbalanceThresholdError(f"{label} TEST seal changed: {key}.")


def _phase11_artifact_hashes(directory: Path, manifest: dict[str, Any]) -> None:  # pragma: no cover
    declared = manifest.get("artifact_file_sha256")
    if not isinstance(declared, dict):
        raise ImbalanceThresholdError("Phase 11 artifact_file_sha256 is missing.")
    for name, digest in declared.items():
        path = directory / str(name)
        if not path.is_file() or sha256_file(path) != str(digest):
            raise ImbalanceThresholdError(f"Phase 11 artifact hash changed: {name}.")


def _frozen_from_assignments(
    assignments: pd.DataFrame, manifest: dict[str, Any]
) -> InnerFoldPlan:  # pragma: no cover
    folds: list[InnerFold] = []
    for item in manifest.get("folds", []):
        fold_id = int(item["fold_id"])
        part = assignments.loc[assignments["fold_id"] == fold_id]
        train_keys = tuple(int(value) for value in part.loc[part["role"] == "TRAIN", KEY].tolist())
        validation_keys = tuple(
            int(value) for value in part.loc[part["role"] == "VALIDATION", KEY].tolist()
        )
        folds.append(
            InnerFold(
                fold_id,
                train_keys,
                validation_keys,
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
        raise ImbalanceThresholdError("Phase 12 requires exactly three frozen inner folds.")
    return InnerFoldPlan(
        assignments=assignments,
        folds=tuple(sorted(folds, key=lambda item: item.fold_id)),
        manifest=manifest,
        content_sha256=str(manifest["fold_content_sha256"]),
    )


def reconstruct_frozen_fold_plan(  # pragma: no cover
    phase10_dir: Path,
    phase10_inputs: Phase10Inputs,
    train_targets: pd.DataFrame,
) -> InnerFoldPlan:
    folds_path = phase10_dir / "inner_cv_folds.parquet"
    manifest_path = phase10_dir / "inner_cv_manifest.json"
    if not folds_path.is_file() or not manifest_path.is_file():
        raise ImbalanceThresholdError("Phase 10 inner-fold artifacts are missing.")
    assignments = pd.read_parquet(folds_path)
    if list(assignments.columns) != [KEY, DATE, "fold_id", "role"]:
        raise ImbalanceThresholdError("Phase 10 inner fold schema changed.")
    manifest = _read_json(manifest_path)
    digest = fold_content_sha256(assignments)
    if digest != manifest.get("fold_content_sha256"):
        raise ImbalanceThresholdError("Phase 10 inner fold content hash changed.")
    train = phase10_inputs.development.loc[phase10_inputs.development["split"] == "TRAIN"].copy()
    reconstructed = build_inner_fold_plan(
        train[[KEY, CLAIM_DATE]].rename(columns={CLAIM_DATE: DATE}),
        train_targets,
        fractions=(0.55, 0.70, 0.85, 1.0),
        minimum_train_positive=40,
        minimum_validation_positive=10,
    )
    if reconstructed.content_sha256 != digest:
        raise ImbalanceThresholdError("Phase 12 could not reproduce the exact Phase 10 folds.")
    return _frozen_from_assignments(assignments, manifest)


def _feature_list_hash(features: tuple[str, ...]) -> str:  # pragma: no cover
    return canonical_json_sha256(list(features))


def _resolve_parent(  # pragma: no cover
    root: Path,
    phase11_dir: Path,
    track: str,
    phase11_manifest: dict[str, Any],
    model_manifest: dict[str, Any],
    parent_model_manifest: dict[str, Any],
    phase10_inputs: Phase10Inputs,
) -> ParentResolution:
    effective = phase11_manifest.get("effective_parent_candidates", {}).get(track)
    parent_candidate = parent_model_manifest.get("tracks", {}).get(track, {})
    if not isinstance(effective, str) or not isinstance(parent_candidate, dict):
        raise ImbalanceThresholdError(
            f"Phase 11 effective parent resolution is missing for {track}."
        )
    parent_id = str(parent_candidate.get("effective_parent_candidate_id"))
    if effective != parent_id and not effective.startswith("P11_"):
        raise ImbalanceThresholdError(f"Phase 11 effective parent mismatch for {track}.")
    if effective.startswith("P11_"):
        entry = model_manifest.get("models", {}).get(track)
        selected_path = phase11_dir / f"selected_features_{track.lower()}.json"
        if not isinstance(entry, dict) or not selected_path.is_file():
            raise ImbalanceThresholdError(
                f"Phase 11 selected model evidence is missing for {track}."
            )
        selected = _read_json(selected_path)
        feature_names = tuple(str(item) for item in selected.get("features", []))
        parent_spec = phase10_inputs.feature_sets["E1" if track == "T1" else "E3"]
        from ..feature_selection.selection import subset_feature_set

        feature_set = subset_feature_set(
            parent_spec, feature_names, str(selected.get("candidate_id"))
        )
        model_path = _resolve(phase11_dir, str(entry.get("model_file")))
        source_phase = 11
        effective_id = effective
        selected_candidate = str(selected.get("candidate_id"))
        parameter_hash = str(parent_candidate.get("parent_parameter_sha256"))
        statistical_raw = entry.get(
            "statistical_parameters", parent_candidate.get("statistical_parameters", {})
        )
        statistical = dict(statistical_raw) if isinstance(statistical_raw, dict) else {}
        declared_feature_hash = str(selected.get("feature_set_sha256"))
        if feature_set.feature_set_sha256 != declared_feature_hash:
            raise ImbalanceThresholdError(f"Phase 11 selected feature hash changed for {track}.")
    else:
        experiment = "E1" if track == "T1" else "E3"
        feature_set = phase10_inputs.feature_sets[experiment]
        model_path = _resolve(root, str(parent_candidate.get("parent_model_source")))
        source_phase = (
            9
            if parent_id.startswith("P9_")
            or bool(parent_candidate.get("fallback_from_phase10_optimization"))
            else 10
        )
        effective_id = parent_id
        selected_candidate = None
        parameter_hash = str(parent_candidate.get("parent_parameter_sha256"))
        statistical_raw = parent_candidate.get("statistical_parameters", {})
        statistical = dict(statistical_raw) if isinstance(statistical_raw, dict) else {}
        declared_feature_hash = feature_set.feature_set_sha256
    if not model_path.is_file():
        raise ImbalanceThresholdError(f"Effective Phase 11 parent model is missing: {model_path}")
    model_digest = sha256_file(model_path)
    declared_model_digest = None
    if source_phase == 11:
        declared_model_digest = model_manifest.get("models", {}).get(track, {}).get("model_sha256")
    if declared_model_digest and model_digest != str(declared_model_digest):
        raise ImbalanceThresholdError(f"Phase 11 parent model hash changed for {track}.")
    model = load_model(model_path)
    actual = effective_parameters(model)
    # A Phase 11 model may persist execution-only values; keep the immutable statistical view.
    actual_statistical = {
        key: value
        for key, value in actual.items()
        if key not in {"thread_count", "allow_writing_files", "verbose", "use_best_model"}
    }
    if statistical:
        for key, value in statistical.items():
            if key in {"thread_count", "allow_writing_files", "verbose", "use_best_model"}:
                continue
            actual_value = actual_statistical.get(key)
            if isinstance(value, (int, float)) and isinstance(actual_value, (int, float)):
                if abs(float(actual_value) - float(value)) > 1.0e-6:
                    raise ImbalanceThresholdError(
                        f"Actual CatBoost parameters differ from Phase 11 parent for {track}: {key}."
                    )
            elif actual_value != value:
                raise ImbalanceThresholdError(
                    f"Actual CatBoost parameters differ from Phase 11 parent for {track}: {key}."
                )
    return ParentResolution(
        track=track,
        effective_candidate_id=effective_id,
        source_phase=source_phase,
        source_model=model_path,
        source_model_sha256=model_digest,
        feature_set=feature_set,
        feature_list_sha256=_feature_list_hash(feature_set.feature_names),
        parent_parameter_sha256=parameter_hash,
        statistical_parameters=statistical or actual_statistical,
        phase11_parent_candidate_id=parent_id,
        selected_feature_candidate_id=selected_candidate,
    )


def load_locked_phase11_inputs(  # pragma: no cover
    phase11_dir: Path,
    *,
    project_root: Path | None = None,
    validate_phase11: bool = False,
) -> Phase12Inputs:
    root = discover_repository_root(project_root)
    directory = phase11_dir.expanduser().resolve()
    required = (
        "phase11_manifest.json",
        "validation.json",
        "model_manifest.json",
        "parent_model_manifest.json",
        "selection_freeze.json",
        "target_access_audit.json",
        "selected_features_t1.json",
        "selected_features_t3.json",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise ImbalanceThresholdError("Phase 11 artifacts missing: " + ", ".join(missing))
    phase11_manifest = _read_json(directory / "phase11_manifest.json")
    validation = _read_json(directory / "validation.json")
    model_manifest = _read_json(directory / "model_manifest.json")
    parent_manifest = _read_json(directory / "parent_model_manifest.json")
    selection_freeze = _read_json(directory / "selection_freeze.json")
    audit = _read_json(directory / "target_access_audit.json")
    if phase11_manifest.get("phase") != 11 or phase11_manifest.get("run_id") != PHASE11_RUN_ID:
        raise ImbalanceThresholdError("Phase 12 requires the accepted Phase 11 run id.")
    if validation.get("valid") is not True or validation.get("hardening_status") != "HARDENED_PASS":
        raise ImbalanceThresholdError("Phase 11 validation is not HARDENED_PASS.")
    if (
        selection_freeze.get("outer_validation_accessed") is not False
        or selection_freeze.get("test_target_accessed") is not False
    ):
        raise ImbalanceThresholdError("Phase 11 selection freeze is not sealed.")
    _validate_test_seal(phase11_manifest, "Phase 11 manifest")
    _validate_test_seal(audit, "Phase 11 target audit")
    _phase11_artifact_hashes(directory, phase11_manifest)
    if validate_phase11:
        result = validate_existing_selection(directory)
        if not result.get("valid"):
            raise ImbalanceThresholdError(
                "Standalone Phase 11 validation failed: " + "; ".join(result.get("errors", []))
            )
    phase10_dir = root / "artifacts" / "catboost_optimization" / PHASE10_RUN_ID
    phase9_dir = root / "artifacts" / "baseline_models" / PHASE9_RUN_ID
    phase10_inputs = load_locked_phase9_inputs(phase9_dir, project_root=root)
    train_targets, _ = load_train_targets_for_optimization(phase10_inputs)
    fold_plan = reconstruct_frozen_fold_plan(phase10_dir, phase10_inputs, train_targets)
    parents = {
        track: _resolve_parent(
            root,
            directory,
            track,
            phase11_manifest,
            model_manifest,
            parent_manifest,
            phase10_inputs,
        )
        for track in TRACKS
    }
    return Phase12Inputs(
        root=root,
        phase11_dir=directory,
        phase11_manifest=phase11_manifest,
        phase11_validation=validation,
        phase11_model_manifest=model_manifest,
        phase11_parent_model_manifest=parent_manifest,
        phase11_manifest_sha256=sha256_file(directory / "phase11_manifest.json"),
        phase11_validation_sha256=sha256_file(directory / "validation.json"),
        phase11_model_manifest_sha256=sha256_file(directory / "model_manifest.json"),
        phase10_dir=phase10_dir,
        phase9_dir=phase9_dir,
        phase9_inputs=phase10_inputs.phase9_inputs,
        phase10_inputs=phase10_inputs,
        development=phase10_inputs.development.copy(),
        parents=parents,
        fold_plan=fold_plan,
    )


def write_parent_resolution(
    path: Path, inputs: Phase12Inputs
) -> dict[str, Any]:  # pragma: no cover
    payload = {
        "phase": 12,
        "phase11_run_id": inputs.phase11_manifest.get("run_id"),
        "phase11_manifest_sha256": inputs.phase11_manifest_sha256,
        "phase11_validation_sha256": inputs.phase11_validation_sha256,
        "phase11_model_manifest_sha256": inputs.phase11_model_manifest_sha256,
        "phase10_inner_fold_sha256": inputs.fold_plan.content_sha256,
        "tracks": {track: inputs.parents[track].as_dict() for track in TRACKS},
    }
    write_json(path, payload)
    return payload


__all__ = [
    "PHASE11_RUN_ID",
    "Phase12Inputs",
    "ParentResolution",
    "load_locked_phase11_inputs",
    "reconstruct_frozen_fold_plan",
    "write_parent_resolution",
]
