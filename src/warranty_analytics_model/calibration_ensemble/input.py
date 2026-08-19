"""Dynamic Phase 12 lock resolution and TRAIN OOF input guards."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..catboost_optimization.input import load_train_targets_for_optimization
from ..catboost_optimization.provenance import sha256_file
from ..imbalance_threshold.input import Phase12Inputs, load_locked_phase11_inputs
from ..imbalance_threshold.validation import validate_existing_phase12
from ..paths import discover_repository_root
from .config import TRACKS, CalibrationEnsembleError

KEY = "warranty_claim_key"
TARGET = "target__high_cost_claim_flag"
OOF_COLUMNS = [KEY, "track", "strategy_id", "fold_id", "high_cost_probability"]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationEnsembleError(f"Invalid Phase 13 upstream JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise CalibrationEnsembleError(f"Phase 13 upstream JSON must be an object: {path}")
    return payload


def current_git_commit(root: Path) -> str:
    for ref in ("refs/remotes/origin/main", "refs/heads/main", "HEAD"):
        result = subprocess.run(
            ["git", "rev-parse", ref],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return "unknown"


def _phase12_test_seal(payload: dict[str, Any], label: str) -> None:
    expected = {
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
    }
    for key, value in expected.items():
        # Older accepted Phase 12 manifests carried this control only in the
        # target-access audit.  Absence is therefore compatible; a present
        # value must still be the sealed value.
        if key == "test_target_access_allowed" and key not in payload:
            continue
        if payload.get(key) != value:
            raise CalibrationEnsembleError(f"{label} TEST seal changed: {key}.")


@dataclass(frozen=True, slots=True)
class Phase12Lock:
    root: Path
    phase12_dir: Path
    phase12_inputs: Phase12Inputs
    phase12_manifest: dict[str, Any]
    phase12_validation: dict[str, Any]
    phase12_freeze: dict[str, Any]
    phase12_audit: dict[str, Any]
    phase12_effective_model_manifest: dict[str, Any]
    phase12_parent_resolution: dict[str, Any]
    phase12_manifest_sha256: str
    phase12_validation_sha256: str
    phase12_effective_model_manifest_sha256: str
    phase12_freeze_sha256: str
    phase12_lock_commit: str
    train_targets: pd.DataFrame
    source_oof: pd.DataFrame
    effective_models: dict[str, dict[str, Any]]

    @property
    def run_id(self) -> str:
        return str(self.phase12_manifest["run_id"])


def _effective_models(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = payload.get("models")
    if not isinstance(entries, list):
        raise CalibrationEnsembleError("Phase 12 effective model manifest schema changed.")
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or str(entry.get("track")) not in TRACKS:
            raise CalibrationEnsembleError("Phase 12 effective model manifest has invalid tracks.")
        track = str(entry["track"])
        if track in result:
            raise CalibrationEnsembleError(f"Phase 12 has duplicate effective model for {track}.")
        result[track] = dict(entry)
    if set(result) != set(TRACKS):
        raise CalibrationEnsembleError("Phase 12 effective model manifest must contain T1 and T3.")
    return result


def _source_oof(
    lock_dir: Path,
    phase12_inputs: Phase12Inputs,
    effective: dict[str, dict[str, Any]],
    train_targets: pd.DataFrame,
) -> pd.DataFrame:
    path = lock_dir / "strategy_oof_predictions.parquet"
    if not path.is_file():
        raise CalibrationEnsembleError("Phase 12 strategy OOF predictions are missing.")
    frame = pd.read_parquet(path)
    if list(frame.columns) != OOF_COLUMNS:
        raise CalibrationEnsembleError("Phase 12 OOF prediction schema changed.")
    assignment = phase12_inputs.fold_plan.assignments
    expected = assignment.loc[
        assignment["role"] == "VALIDATION", [KEY, "fold_id", "claim_date"]
    ].copy()
    expected[KEY] = expected[KEY].astype(int)
    if expected[KEY].duplicated().any():
        raise CalibrationEnsembleError("Phase 12 validation fold membership is duplicated.")
    expected_keys = set(expected[KEY])
    targets = train_targets.set_index(KEY)[TARGET]
    result: list[pd.DataFrame] = []
    track_keys: dict[str, set[int]] = {}
    for track in TRACKS:
        strategy = str(effective[track].get("selected_imbalance_strategy", ""))
        if not strategy:
            raise CalibrationEnsembleError(f"Phase 12 effective strategy is missing for {track}.")
        selected = frame.loc[(frame["track"] == track) & (frame["strategy_id"] == strategy)].copy()
        if selected.empty:
            raise CalibrationEnsembleError(
                f"Phase 12 effective OOF strategy is missing for {track}."
            )
        if selected[KEY].duplicated().any():
            raise CalibrationEnsembleError(
                f"Phase 12 effective OOF claims are duplicated for {track}."
            )
        selected[KEY] = selected[KEY].astype(int)
        if set(selected[KEY]) != expected_keys:
            raise CalibrationEnsembleError(
                f"Phase 12 OOF membership differs from frozen folds for {track}."
            )
        if set(selected["fold_id"].astype(int)) != {1, 2, 3}:
            raise CalibrationEnsembleError(f"Phase 12 OOF source folds are incomplete for {track}.")
        if (
            not np.isfinite(selected["high_cost_probability"]).all()
            or (
                (selected["high_cost_probability"] < 0) | (selected["high_cost_probability"] > 1)
            ).any()
        ):
            raise CalibrationEnsembleError(f"Phase 12 OOF probabilities are invalid for {track}.")
        merged = selected.merge(expected, on=[KEY, "fold_id"], how="left", validate="one_to_one")
        if merged["claim_date"].isna().any():
            raise CalibrationEnsembleError(
                f"Phase 12 OOF fold membership cannot be reproduced for {track}."
            )
        if not set(merged[KEY]).issubset(set(targets.index)):
            raise CalibrationEnsembleError(
                f"Phase 12 effective OOF includes non-TRAIN claims for {track}."
            )
        merged["target"] = merged[KEY].map(targets).astype("int8")
        merged["track"] = track
        track_keys[track] = set(int(value) for value in merged[KEY])
        result.append(merged)
    if track_keys["T1"] != track_keys["T3"]:
        raise CalibrationEnsembleError("Phase 12 T1/T3 OOF populations do not match exactly.")
    return (
        pd.concat(result, ignore_index=True)
        .sort_values(["track", "fold_id", KEY], kind="mergesort")
        .reset_index(drop=True)
    )


def load_phase12_lock(
    phase12_dir: Path,
    *,
    project_root: Path | None = None,
    validate_phase12: bool = True,
) -> Phase12Lock:
    directory = phase12_dir.expanduser().resolve()
    # A caller may provide only the immutable Phase 12 artifact directory.  In
    # that case resolve the repository from the artifact path instead of
    # treating the artifact itself as the project root (which would make the
    # Phase 11 parent lookup point at the wrong tree).
    root = discover_repository_root(project_root or directory)
    required = (
        "phase12_manifest.json",
        "phase11_parent_resolution.json",
        "effective_model_manifest.json",
        "model_manifest.json",
        "threshold_policy.json",
        "phase12_freeze.json",
        "validation.json",
        "target_access_audit.json",
        "strategy_oof_predictions.parquet",
        "strategy_summary.parquet",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise CalibrationEnsembleError("Phase 12 artifacts missing: " + ", ".join(missing))
    phase12_manifest = _read_json(directory / "phase12_manifest.json")
    phase12_validation = _read_json(directory / "validation.json")
    phase12_freeze = _read_json(directory / "phase12_freeze.json")
    phase12_audit = _read_json(directory / "target_access_audit.json")
    effective_manifest = _read_json(directory / "effective_model_manifest.json")
    parent_resolution = _read_json(directory / "phase11_parent_resolution.json")
    if phase12_manifest.get("phase") != 12 or not phase12_manifest.get("run_id"):
        raise CalibrationEnsembleError("Phase 12 manifest is not a valid accepted run.")
    if (
        phase12_validation.get("valid") is not True
        or phase12_validation.get("hardening_status") != "HARDENED_PASS"
    ):
        raise CalibrationEnsembleError("Phase 12 is not independently HARDENED_PASS.")
    _phase12_test_seal(phase12_manifest, "Phase 12 manifest")
    _phase12_test_seal(phase12_audit, "Phase 12 target audit")
    if "outer_validation_accessed" not in phase12_freeze:
        raise CalibrationEnsembleError(
            "Accepted Phase 12 freeze is missing its validation-access seal."
        )
    if validate_phase12:
        upstream_result = validate_existing_phase12(directory, project_root=root)
        if (
            upstream_result.get("valid") is not True
            or upstream_result.get("hardening_status") != "HARDENED_PASS"
        ):
            raise CalibrationEnsembleError("Phase 12 standalone validator did not pass.")
    phase11_run_id = str(
        phase12_manifest.get("phase11_run_id") or parent_resolution.get("phase11_run_id", "")
    )
    if not phase11_run_id:
        raise CalibrationEnsembleError(
            "Phase 12 parent resolution does not identify its Phase 11 run."
        )
    phase11_dir = root / "artifacts" / "feature_selection" / phase11_run_id
    phase12_inputs = load_locked_phase11_inputs(phase11_dir, project_root=root)
    train_targets, _ = load_train_targets_for_optimization(phase12_inputs.phase10_inputs)
    effective_models = _effective_models(effective_manifest)
    for track, entry in effective_models.items():
        model_path = directory / str(entry.get("model_file", ""))
        if not model_path.is_file():
            raise CalibrationEnsembleError(f"Phase 12 effective model is missing for {track}.")
        declared = str(entry.get("model_sha256", ""))
        if not declared or sha256_file(model_path) != declared:
            raise CalibrationEnsembleError(f"Phase 12 effective model hash changed for {track}.")
        if not entry.get("feature_set_sha256") or not entry.get("feature_list_sha256"):
            raise CalibrationEnsembleError(
                f"Phase 12 effective feature provenance is incomplete for {track}."
            )
    source = _source_oof(directory, phase12_inputs, effective_models, train_targets)
    return Phase12Lock(
        root=root,
        phase12_dir=directory,
        phase12_inputs=phase12_inputs,
        phase12_manifest=phase12_manifest,
        phase12_validation=phase12_validation,
        phase12_freeze=phase12_freeze,
        phase12_audit=phase12_audit,
        phase12_effective_model_manifest=effective_manifest,
        phase12_parent_resolution=parent_resolution,
        phase12_manifest_sha256=sha256_file(directory / "phase12_manifest.json"),
        phase12_validation_sha256=sha256_file(directory / "validation.json"),
        phase12_effective_model_manifest_sha256=sha256_file(
            directory / "effective_model_manifest.json"
        ),
        phase12_freeze_sha256=sha256_file(directory / "phase12_freeze.json"),
        phase12_lock_commit=current_git_commit(root),
        train_targets=train_targets,
        source_oof=source,
        effective_models=effective_models,
    )


def write_phase12_parent_resolution(path: Path, lock: Phase12Lock) -> dict[str, Any]:
    tracks: dict[str, Any] = {}
    for track in TRACKS:
        entry = lock.effective_models[track]
        parent = lock.phase12_parent_resolution.get("tracks", {}).get(track, {})
        tracks[track] = {
            "effective_candidate_id": entry.get("candidate_id"),
            "source_phase": entry.get("source_phase", parent.get("source_phase")),
            "model_file": entry.get("model_file"),
            "model_sha256": entry.get("model_sha256"),
            "feature_count": entry.get("feature_count", parent.get("feature_count")),
            "feature_set_sha256": entry.get("feature_set_sha256", parent.get("feature_set_sha256")),
            "feature_list_sha256": entry.get(
                "feature_list_sha256", parent.get("feature_list_sha256")
            ),
            "parameter_sha256": entry.get(
                "parameter_sha256", parent.get("parent_parameter_sha256")
            ),
            "imbalance_strategy": entry.get("selected_imbalance_strategy"),
            "technical_threshold": entry.get("technical_threshold"),
            "score_space": "RAW_UNCALIBRATED_PROBABILITY",
        }
    payload = {
        "phase": 13,
        "phase12_run_id": lock.run_id,
        "phase12_lock_commit": lock.phase12_lock_commit,
        "phase12_manifest_sha256": lock.phase12_manifest_sha256,
        "phase12_validation_sha256": lock.phase12_validation_sha256,
        "phase12_effective_model_manifest_sha256": lock.phase12_effective_model_manifest_sha256,
        "phase12_freeze_sha256": lock.phase12_freeze_sha256,
        "tracks": tracks,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "KEY",
    "OOF_COLUMNS",
    "Phase12Lock",
    "current_git_commit",
    "load_phase12_lock",
    "write_phase12_parent_resolution",
]
