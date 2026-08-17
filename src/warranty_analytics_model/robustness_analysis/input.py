"""Dynamic Phase 13 parent resolution and safe Phase 14 population loading."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..baseline_model.adapters import adapt_matrix
from ..baseline_model.catboost_baseline import load_model, predict_probabilities
from ..baseline_model.config import load_baseline_settings
from ..baseline_model.models import FeatureSetSpec
from ..calibration_ensemble.calibrators import apply_calibrator
from ..calibration_ensemble.input import KEY, TARGET, load_phase12_lock
from ..calibration_ensemble.validation import validate_existing_phase13
from ..catboost_optimization.input import load_validation_targets_after_freeze
from ..catboost_optimization.provenance import canonical_json_sha256, sha256_file
from ..paths import discover_repository_root


class Phase14InputError(ValueError):
    """Raised when Phase 13 provenance or population safety is not reproducible."""


_UPSTREAM_VALIDATION_CACHE: dict[tuple[str, str, str, str], dict[str, Any]] = {}


@dataclass(frozen=True, slots=True)
class Phase14Component:
    track: str
    candidate_id: str
    model_path: Path
    model_sha256: str
    feature_set: FeatureSetSpec
    feature_list_sha256: str
    calibration_method: str
    calibrator_path: Path
    calibrator_sha256: str
    score_space: str
    threshold: float


@dataclass(frozen=True, slots=True)
class Phase14Resolved:
    root: Path
    phase13_dir: Path
    phase13_manifest: dict[str, Any]
    phase13_freeze: dict[str, Any]
    phase13_validation: dict[str, Any]
    phase13_metrics: dict[str, Any]
    effective_manifest: dict[str, Any]
    threshold_policy: dict[str, Any]
    phase13_audit: dict[str, Any]
    parent_resolution: dict[str, Any]
    phase13_manifest_sha256: str
    phase13_validation_sha256: str
    phase13_freeze_sha256: str
    effective_manifest_sha256: str
    champion_id: str
    champion_type: str
    score_space: str
    threshold: float
    components: tuple[Phase14Component, ...]
    ensemble_t1_weight: float | None
    development: pd.DataFrame
    train_targets: pd.DataFrame
    phase12_lock: Any

    @property
    def feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for component in self.components:
            for name in component.feature_set.feature_names:
                if name not in names:
                    names.append(name)
        return tuple(names)

    @property
    def validation_features(self) -> pd.DataFrame:
        return self.development.loc[self.development["split"] == "VALIDATION"].copy()

    @property
    def train_features(self) -> pd.DataFrame:
        return self.development.loc[self.development["split"] == "TRAIN"].copy()

    def load_validation_targets(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Stage B only: outer labels are loaded after the analysis freeze."""

        return load_validation_targets_after_freeze(
            self.phase12_lock.phase12_inputs.phase10_inputs,
            study_frozen=True,
        )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase14InputError(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise Phase14InputError(f"{label} must be a JSON object: {path}")
    return payload


def _test_seal(payload: dict[str, Any], label: str) -> list[str]:
    expected = {
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
    }
    return [
        f"{label} TEST seal changed: {key}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]


def current_git_commit(root: Path, ref: str = "HEAD") -> str:
    result = subprocess.run(
        ["git", "rev-parse", ref], cwd=root, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def phase13_merged_to_main(root: Path, phase13_commit: str) -> bool:
    """Return whether the accepted Phase 13 commit is reachable from local main."""

    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(phase13_commit), "refs/heads/main"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _component_from_entry(
    phase13_dir: Path,
    lock: Any,
    entry: dict[str, Any],
    track: str,
) -> Phase14Component:
    parent = lock.effective_models.get(track, {})
    parent_spec = lock.phase12_inputs.parents[track].feature_set
    model_path = phase13_dir / str(entry.get("model_file", ""))
    if not model_path.is_file():
        # A future accepted Phase 13 may reference the Phase 12 parent directly.
        model_path = lock.phase12_dir / str(parent.get("model_file", ""))
    if not model_path.is_file():
        raise Phase14InputError(f"Phase 13 component model is missing for {track}.")
    declared_model = str(entry.get("source_model_sha256") or entry.get("model_sha256") or "")
    actual_model = sha256_file(model_path)
    if declared_model and actual_model != declared_model:
        raise Phase14InputError(f"Phase 13 component model SHA mismatch for {track}.")
    feature_hash = str(entry.get("feature_list_sha256", ""))
    actual_feature_hash = canonical_json_sha256(list(parent_spec.feature_names))
    if feature_hash and feature_hash != actual_feature_hash:
        raise Phase14InputError(f"Phase 13 feature-list SHA mismatch for {track}.")
    calibrator_path = phase13_dir / "calibrators" / f"{track.lower()}.json"
    if not calibrator_path.is_file():
        raise Phase14InputError(f"Phase 13 calibrator is missing for {track}.")
    calibrator = _read_json(calibrator_path, f"Phase 13 {track} calibrator")
    declared_calibrator = str(entry.get("calibrator_sha", ""))
    actual_calibrator = sha256_file(calibrator_path)
    if declared_calibrator and actual_calibrator != declared_calibrator:
        # The serialized calibrator SHA is also checked by the Phase 13
        # validator; accepting either the manifest or payload digest here keeps
        # this loader compatible with both accepted artifact generations.
        payload_sha = str(calibrator.get("calibrator_sha", ""))
        if payload_sha != declared_calibrator:
            raise Phase14InputError(f"Phase 13 calibrator SHA mismatch for {track}.")
    method = str(entry.get("calibration_method", calibrator.get("method", "NONE")))
    threshold = float(entry.get("technical_threshold", 0.5))
    if not 0.0 < threshold < 1.0:
        raise Phase14InputError(f"Phase 13 technical threshold is invalid for {track}.")
    return Phase14Component(
        track=track,
        candidate_id=str(entry.get("candidate_id", parent.get("candidate_id", ""))),
        model_path=model_path,
        model_sha256=actual_model,
        feature_set=parent_spec,
        feature_list_sha256=actual_feature_hash,
        calibration_method=method,
        calibrator_path=calibrator_path,
        calibrator_sha256=actual_calibrator,
        score_space=str(entry.get("score_space", "RAW_UNCALIBRATED_PROBABILITY")),
        threshold=threshold,
    )


def _validate_phase13_once(directory: Path, root: Path) -> dict[str, Any]:
    """Validate one immutable Phase 13 directory once per process.

    The Phase 13 validator reconstructs calibration and threshold decisions and
    can take several minutes.  Phase 14 calls it while resolving the parent
    and again from its independent artifact validator; the hash-keyed cache
    avoids redundant work while still invalidating if any parent contract file
    changes.
    """

    key = (
        str(directory),
        sha256_file(directory / "phase13_manifest.json"),
        sha256_file(directory / "validation.json"),
        sha256_file(directory / "effective_model_manifest.json"),
    )
    if key not in _UPSTREAM_VALIDATION_CACHE:
        _UPSTREAM_VALIDATION_CACHE[key] = validate_existing_phase13(directory, project_root=root)
    return _UPSTREAM_VALIDATION_CACHE[key]


def resolve_phase13_parent(
    phase13_dir: Path,
    *,
    project_root: Path | None = None,
    require_main_merge: bool = False,
) -> Phase14Resolved:
    """Resolve an explicit accepted Phase 13 run without guessing a run ID."""

    root = discover_repository_root(project_root or phase13_dir)
    directory = phase13_dir.expanduser().resolve()
    required = (
        "phase13_manifest.json",
        "phase13_freeze.json",
        "validation.json",
        "validation_metrics.json",
        "effective_model_manifest.json",
        "threshold_policy.json",
        "target_access_audit.json",
        "phase12_parent_resolution.json",
        "validation_predictions.parquet",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise Phase14InputError("Phase 13 artifacts missing: " + ", ".join(missing))
    manifest = _read_json(directory / "phase13_manifest.json", "Phase 13 manifest")
    freeze = _read_json(directory / "phase13_freeze.json", "Phase 13 freeze")
    phase13_validation = _read_json(directory / "validation.json", "Phase 13 validation")
    metrics = _read_json(directory / "validation_metrics.json", "Phase 13 validation metrics")
    effective = _read_json(
        directory / "effective_model_manifest.json", "Phase 13 effective model manifest"
    )
    threshold_policy = _read_json(directory / "threshold_policy.json", "Phase 13 threshold policy")
    audit = _read_json(directory / "target_access_audit.json", "Phase 13 target audit")
    parent = _read_json(directory / "phase12_parent_resolution.json", "Phase 13 parent resolution")
    if manifest.get("phase") != 13 or not manifest.get("run_id"):
        raise Phase14InputError("Explicit Phase 13 directory is not a Phase 13 run.")
    phase13_commit = str(manifest.get("git_commit_sha", ""))
    if require_main_merge and not phase13_merged_to_main(root, phase13_commit):
        raise Phase14InputError("Phase 13 implementation is not merged to local main.")
    upstream = _validate_phase13_once(directory, root)
    if upstream.get("valid") is not True or upstream.get("hardening_status") != "HARDENED_PASS":
        raise Phase14InputError("Phase 13 standalone validation is not HARDENED_PASS.")
    errors = _test_seal(manifest, "Phase 13 manifest") + _test_seal(audit, "Phase 13 target audit")
    if errors:
        raise Phase14InputError("; ".join(errors))
    if (
        freeze.get("outer_validation_accessed") is not False
        or freeze.get("test_target_accessed") is not False
    ):
        raise Phase14InputError("Phase 13 freeze is not sealed for Phase 14.")
    phase12_dir = Path(str(manifest.get("phase12_dir", "")))
    if not phase12_dir.is_absolute():
        phase12_dir = (root / phase12_dir).resolve()
    lock = load_phase12_lock(phase12_dir, project_root=root)
    entries = effective.get("models")
    if not isinstance(entries, list):
        raise Phase14InputError("Phase 13 effective model manifest has no model entries.")
    champion_id = str(
        manifest.get("phase13_development_champion")
        or metrics.get("phase13_development_champion", "")
    )
    champion_entry = next(
        (
            item
            for item in entries
            if isinstance(item, dict) and item.get("candidate_id") == champion_id
        ),
        None,
    )
    if not isinstance(champion_entry, dict):
        # BEST_SINGLE artifacts persist the component candidates but can omit
        # the selected champion from the effective list in older versions.
        candidate_entries = manifest.get("effective_candidates", [])
        champion_entry = next(
            (
                item
                for item in candidate_entries
                if isinstance(item, dict) and item.get("candidate_id") == champion_id
            ),
            None,
        )
    if not isinstance(champion_entry, dict):
        raise Phase14InputError(f"Phase 13 champion cannot be resolved: {champion_id}.")
    champion_type = str(champion_entry.get("candidate_type", "SINGLE_TRACK"))
    if champion_type == "ENSEMBLE":
        tracks = ("T1", "T3")
        weight = float(
            champion_entry.get("t1_weight", effective.get("selected_ensemble_weight", 0.5))
        )
        component_entries = {
            track: next(
                (item for item in entries if isinstance(item, dict) and item.get("track") == track),
                None,
            )
            for track in tracks
        }
        if any(not isinstance(item, dict) for item in component_entries.values()):
            raise Phase14InputError("Phase 13 ensemble component manifest is incomplete.")
        complete_entries = {
            track: item for track, item in component_entries.items() if isinstance(item, dict)
        }
        components = tuple(
            _component_from_entry(directory, lock, complete_entries[track], track)
            for track in tracks
        )
        score_space = str(champion_entry.get("score_space", "CALIBRATED_ENSEMBLE_PROBABILITY"))
        threshold = float(
            champion_entry.get(
                "technical_threshold",
                threshold_policy.get("candidates", {}).get("ENSEMBLE", {}).get("threshold", 0.5),
            )
        )
    else:
        track = str(champion_entry.get("track") or ("T1" if "T1" in champion_id else "T3"))
        if track not in {"T1", "T3"}:
            raise Phase14InputError("Phase 13 single-track champion has no valid track.")
        components = (_component_from_entry(directory, lock, champion_entry, track),)
        weight = None
        score_space = str(champion_entry.get("score_space", components[0].score_space))
        threshold = float(champion_entry.get("technical_threshold", components[0].threshold))
    development = lock.phase12_inputs.phase10_inputs.development.copy()
    if KEY not in development or "split" not in development:
        raise Phase14InputError("Phase 13 development feature frame is missing controls.")
    if "claim__claim_date" not in development:
        raise Phase14InputError("Phase 14 requires the prediction-time claim date.")
    train_targets = lock.train_targets[[KEY, TARGET]].copy()
    if train_targets[KEY].duplicated().any():
        raise Phase14InputError("Phase 13 TRAIN targets are duplicated.")
    return Phase14Resolved(
        root=root,
        phase13_dir=directory,
        phase13_manifest=manifest,
        phase13_freeze=freeze,
        phase13_validation=phase13_validation,
        phase13_metrics=metrics,
        effective_manifest=effective,
        threshold_policy=threshold_policy,
        phase13_audit=audit,
        parent_resolution=parent,
        phase13_manifest_sha256=sha256_file(directory / "phase13_manifest.json"),
        phase13_validation_sha256=sha256_file(directory / "validation.json"),
        phase13_freeze_sha256=sha256_file(directory / "phase13_freeze.json"),
        effective_manifest_sha256=sha256_file(directory / "effective_model_manifest.json"),
        champion_id=champion_id,
        champion_type=champion_type,
        score_space=score_space,
        threshold=threshold,
        components=components,
        ensemble_t1_weight=weight,
        development=development,
        train_targets=train_targets,
        phase12_lock=lock,
    )


def prepare_scorer(
    resolved: Phase14Resolved,
    *,
    threads: int | None = None,
) -> Any:
    """Load frozen components once and return a reusable prediction callable.

    Phase 14 repeatedly scores the same validation population for row-order and
    batch invariance checks.  Loading CatBoost models and calibrator JSON for
    every partition is needlessly expensive and can exhaust memory.  The
    returned scorer keeps only immutable, serialized Phase 13 state in memory;
    it does not fit or mutate a model.
    """

    baseline_settings = load_baseline_settings(resolved.root)
    loaded = tuple(
        (
            component,
            load_model(component.model_path),
            _read_json(component.calibrator_path, f"{component.track} calibrator"),
        )
        for component in resolved.components
    )

    def scorer(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[[KEY]].copy().reset_index(drop=True)
        component_scores: dict[str, np.ndarray] = {}
        for component, model, calibrator in loaded:
            matrix = adapt_matrix(frame, component.feature_set, baseline_settings)
            raw = predict_probabilities(
                model,
                matrix,
                component.feature_set,
                thread_count=threads,
            )
            calibrated = apply_calibrator(calibrator, raw)
            effective = (
                calibrated if component.calibration_method in {"SIGMOID", "ISOTONIC"} else raw
            )
            component_scores[component.track] = effective
            result[f"{component.track}_raw_probability"] = raw
            result[f"{component.track}_effective_probability"] = effective
        if resolved.champion_type == "ENSEMBLE":
            weight = float(resolved.ensemble_t1_weight or 0.5)
            result["probability"] = (
                weight * component_scores["T1"] + (1.0 - weight) * component_scores["T3"]
            )
        else:
            track = resolved.components[0].track
            result["probability"] = component_scores[track]
        if (
            not np.isfinite(result["probability"]).all()
            or ((result["probability"] < 0) | (result["probability"] > 1)).any()
        ):
            raise Phase14InputError("Phase 13 produced a non-finite or unbounded probability.")
        return result

    return scorer


def score_frame(
    resolved: Phase14Resolved, frame: pd.DataFrame, *, threads: int | None = None
) -> pd.DataFrame:
    """Score a TRAIN/VALIDATION frame from serialized Phase 13 components only."""

    return prepare_scorer(resolved, threads=threads)(frame)


def train_oof_scores(resolved: Phase14Resolved) -> pd.DataFrame:
    """Load frozen TRAIN OOF scores; no VALIDATION labels are used."""

    path = resolved.phase13_dir / "selected_calibrated_oof_predictions.parquet"
    if not path.is_file():
        path = (
            resolved.phase13_dir.parent.parent
            / "imbalance_threshold"
            / resolved.phase13_manifest["phase12_run_id"]
            / "strategy_oof_predictions.parquet"
        )
    if not path.is_file():
        raise Phase14InputError("Frozen TRAIN OOF score evidence is missing.")
    frame = pd.read_parquet(path)
    if "effective_probability" in frame.columns:
        result = frame[[KEY, "effective_probability"]].rename(
            columns={"effective_probability": "probability"}
        )
    else:
        if resolved.champion_type == "ENSEMBLE":
            parts = []
            for track in ("T1", "T3"):
                selected = frame.loc[
                    frame["track"] == track, [KEY, "raw_probability", "calibrated_probability"]
                ].copy()
                selected["probability"] = selected["calibrated_probability"]
                parts.append(selected[[KEY, "probability"]].rename(columns={"probability": track}))
            merged = parts[0].merge(parts[1], on=KEY, validate="one_to_one")
            weight = float(resolved.ensemble_t1_weight or 0.5)
            result = merged[[KEY]].copy()
            result["probability"] = weight * merged["T1"] + (1.0 - weight) * merged["T3"]
        else:
            track = resolved.components[0].track
            selected = frame.loc[frame["track"] == track].copy()
            column = (
                "calibrated_probability"
                if "calibrated_probability" in selected
                else "raw_probability"
            )
            result = selected[[KEY, column]].rename(columns={column: "probability"})
    if result[KEY].duplicated().any():
        raise Phase14InputError("TRAIN OOF scores contain duplicate claim keys.")
    return result.sort_values(KEY, kind="mergesort").reset_index(drop=True)


__all__ = [
    "KEY",
    "TARGET",
    "Phase14Component",
    "Phase14InputError",
    "Phase14Resolved",
    "current_git_commit",
    "phase13_merged_to_main",
    "prepare_scorer",
    "resolve_phase13_parent",
    "score_frame",
    "train_oof_scores",
]
