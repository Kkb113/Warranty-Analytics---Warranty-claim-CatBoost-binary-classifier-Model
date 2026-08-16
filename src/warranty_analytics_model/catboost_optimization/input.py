"""Load the immutable Phase 9 chain without opening outer validation during search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds

from ..baseline_model.adapters import build_development_feature_frame
from ..baseline_model.feature_sets import resolve_feature_sets
from ..baseline_model.input import load_phase9_inputs
from ..baseline_model.models import BaselineModelError
from ..feature_mart.manifest import sha256_file
from ..paths import discover_repository_root
from .config import TRACK_TO_EXPERIMENT
from .models import OptimizationError, Phase10Inputs

KEY = "warranty_claim_key"
TARGET = "target__high_cost_claim_flag"
CLAIM_DATE = "claim__claim_date"

EXPECTED_PHASE9_TARGET_HASHES = {
    "train": "9d6fde99c726825c1d683cb4fe9394a93d58de1f69a1c5e0561eda698cdf7744",
    "validation": "fded123cbfe67899d4cba0f29e150a93d2bffd79cda96c97ab459e4c2bef49d2",
}
EXPECTED_FEATURE_SETS = {
    "E1": (301, "4a8de5a69ce72bf6059f9856252d68465d464fecfb56242f5fa55646edae7b89"),
    "E3": (536, "13859692eec0494879712b6ac66a3ce06f64cd75ff93de5c80e2c0a67b701738"),
}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OptimizationError(f"Required Phase 10 {label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise OptimizationError(f"Required Phase 10 {label} must be a JSON object: {path}")
    return payload


def _validate_locked_phase9_directory(directory: Path) -> dict[str, Any]:
    required = (
        "experiment_manifest.json",
        "feature_sets.json",
        "model_input_schema.json",
        "model_manifest.json",
        "target_access_audit.json",
        "validation_metrics.json",
        "validation.json",
        "validation_predictions.parquet",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise OptimizationError("Locked Phase 9 artifacts are missing: " + ", ".join(missing))
    manifest = _read_json(directory / "experiment_manifest.json", "Phase 9 manifest")
    if manifest.get("phase") != 9 or manifest.get("hardened_status") != "HARDENED_PASS":
        raise OptimizationError("Phase 10 requires a Phase 9 HARDENED_PASS run.")
    if manifest.get("hardening_version") != "phase9_corrective_hardening_v1":
        raise OptimizationError("Phase 9 hardening version is not the approved immutable version.")
    validation = _read_json(directory / "validation.json", "Phase 9 validation")
    if validation.get("valid") is not True or validation.get("hardening_status") != "HARDENED_PASS":
        raise OptimizationError("Locked Phase 9 validation.json is not HARDENED_PASS.")
    audit = _read_json(directory / "target_access_audit.json", "Phase 9 target audit")
    for key, expected in {
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
    }.items():
        if audit.get(key) != expected:
            raise OptimizationError(f"Phase 9 TEST seal changed: {key}.")
    if manifest.get("target_hashes") != EXPECTED_PHASE9_TARGET_HASHES:
        raise OptimizationError("Phase 9 TRAIN/VALIDATION target hashes are not the locked values.")
    declared_hashes = manifest.get("artifact_file_sha256", {})
    if not isinstance(declared_hashes, dict):
        raise OptimizationError("Phase 9 artifact_file_sha256 is missing.")
    for name, declared in declared_hashes.items():
        path = directory / str(name)
        if not path.is_file() or sha256_file(path) != declared:
            raise OptimizationError(f"Locked Phase 9 artifact hash differs: {name}.")
    return manifest


def load_locked_phase9_inputs(
    phase9_dir: Path,
    *,
    project_root: Path | None = None,
) -> Phase10Inputs:
    """Load exact Phase 9 features and lineage without loading any target column."""

    root = discover_repository_root(project_root)
    directory = phase9_dir.expanduser().resolve()
    manifest = _validate_locked_phase9_directory(directory)
    paths = manifest.get("input_directories")
    if not isinstance(paths, dict) or not all(
        key in paths for key in ("phase5", "phase6", "phase7", "phase8")
    ):
        raise OptimizationError("Phase 9 input_directories are incomplete.")
    try:
        inputs = load_phase9_inputs(
            Path(str(paths["phase5"])),
            Path(str(paths["phase6"])),
            Path(str(paths["phase7"])),
            Path(str(paths["phase8"])),
            project_root=root,
        )
        feature_sets = resolve_feature_sets(inputs.phase7_lineage, inputs.phase8_lineage)
        development = build_development_feature_frame(inputs, feature_sets)
        if CLAIM_DATE not in development.columns:
            claim_dates = inputs.structured_features[[KEY, CLAIM_DATE]].copy()
            if claim_dates[KEY].duplicated().any():
                raise OptimizationError("Phase 7 claim dates are duplicated by warranty_claim_key.")
            development = development.merge(
                claim_dates,
                on=KEY,
                how="left",
                validate="one_to_one",
            )
    except (BaselineModelError, OSError, ValueError) as exc:
        raise OptimizationError(f"Phase 9 locked input chain blocks Phase 10: {exc}") from exc
    for experiment_id, (count, digest) in EXPECTED_FEATURE_SETS.items():
        spec = feature_sets.get(experiment_id)
        if spec is None or spec.feature_count != count or spec.feature_set_sha256 != digest:
            raise OptimizationError(f"Phase 9 {experiment_id} feature set is not immutable.")
        persisted = _read_json(directory / "feature_sets.json", "Phase 9 feature sets").get(
            experiment_id
        )
        if not isinstance(persisted, dict) or persisted.get("feature_set_sha256") != digest:
            raise OptimizationError(f"Persisted Phase 9 {experiment_id} feature metadata differs.")
    if set(development["split"].astype(str)) - {"TRAIN", "VALIDATION"}:
        raise OptimizationError("Phase 9 development matrix contains TEST rows.")
    if CLAIM_DATE not in development.columns:
        raise OptimizationError(
            "Phase 9 development matrix lacks claim__claim_date for inner folds."
        )
    input_hashes = manifest.get("input_hashes", {})
    expected_phase5 = inputs.phase5_manifest.get(
        "artifact_content_fingerprints", inputs.phase5_manifest.get("artifact_content_sha256", {})
    ).get("claim_snapshot")
    expected_inputs = {
        "phase5_claim_snapshot": expected_phase5,
        "phase6_split_assignment": inputs.frozen_membership.get("split_assignment_sha256"),
        "phase7_structured_features": inputs.phase7_manifest.get("artifact_content_sha256", {}).get(
            "structured_features"
        ),
        "phase8_text_features": inputs.phase8_manifest.get("artifact_content_sha256", {}).get(
            "text_features"
        ),
    }
    if input_hashes != expected_inputs or any(value is None for value in expected_inputs.values()):
        raise OptimizationError("Phase 9 locked input hashes do not match Phase 5–8 sources.")
    return Phase10Inputs(
        root=root,
        phase9_dir=directory,
        phase9_manifest=manifest,
        phase9_inputs=inputs,
        feature_sets={key: feature_sets[key] for key in TRACK_TO_EXPERIMENT.values()},
        development=development,
        claim_snapshot_path=inputs.mart_dir / "claim_snapshot.parquet",
    )


def _load_target_rows(
    phase10_inputs: Phase10Inputs,
    split: str,
    *,
    frozen: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if split == "VALIDATION" and not frozen:
        raise OptimizationError(
            "Outer VALIDATION target access is blocked until study_freeze.json is written."
        )
    if split == "TEST":
        raise OptimizationError("Phase 10 never loads TEST targets before Phase 15.")
    assignments = phase10_inputs.phase9_inputs.assignments
    keys = set(assignments.loc[assignments["split"] == split, KEY].astype(int))
    if not keys:
        raise OptimizationError(f"Phase 10 {split} assignment population is empty.")
    dataset = ds.dataset(str(phase10_inputs.claim_snapshot_path), format="parquet")
    if not {KEY, TARGET}.issubset(dataset.schema.names):
        raise OptimizationError("Phase 5 claim snapshot lacks the authoritative target columns.")
    table = dataset.to_table(columns=[KEY, TARGET], filter=ds.field(KEY).isin(sorted(keys)))
    frame = table.to_pandas()
    if set(frame[KEY].astype(int)) != keys or frame[KEY].duplicated().any():
        raise OptimizationError(f"Phase 10 {split} target membership differs from Phase 6.")
    values = pd.to_numeric(frame[TARGET], errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise OptimizationError(f"Phase 10 {split} target is not binary and non-null.")
    frame[TARGET] = values.astype("int8")
    frame = frame.sort_values(KEY, kind="mergesort").reset_index(drop=True)
    if split == "TRAIN":
        expected_hash = EXPECTED_PHASE9_TARGET_HASHES["train"]
    else:
        expected_hash = EXPECTED_PHASE9_TARGET_HASHES["validation"]
    from ..baseline_model.target import target_content_sha256

    actual_hash = target_content_sha256(frame)
    if actual_hash != expected_hash:
        raise OptimizationError(f"Phase 10 {split} target hash differs from locked Phase 9.")
    audit = {
        "phase": "TRAIN_ONLY_INNER_CV" if split == "TRAIN" else "POST_STUDY_FREEZE_VALIDATION",
        "split": split,
        "target_rows_loaded": int(len(frame)),
        "target_content_sha256": actual_hash,
        "validation_target_access_allowed": bool(frozen),
        "test_target_rows_loaded": 0,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
    }
    return frame[[KEY, TARGET]], audit


def load_train_targets_for_optimization(
    phase10_inputs: Phase10Inputs,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stage A loader: materialize TRAIN labels only."""

    return _load_target_rows(phase10_inputs, "TRAIN", frozen=False)


def load_validation_targets_after_freeze(
    phase10_inputs: Phase10Inputs,
    *,
    study_frozen: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stage B loader: outer VALIDATION labels require an explicit freeze gate."""

    return _load_target_rows(phase10_inputs, "VALIDATION", frozen=study_frozen)
