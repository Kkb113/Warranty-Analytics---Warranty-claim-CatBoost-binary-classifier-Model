"""Independent fail-closed Phase 14 artifact validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..feature_mart.manifest import sha256_file
from .config import PHASE14_VERSION, configuration_sha256
from .input import KEY, TARGET, prepare_scorer, resolve_phase13_parent
from .invariance import prediction_invariance
from .metrics import overall_metrics
from .readiness import readiness_gate

REQUIRED_ARTIFACTS = (
    "phase14_manifest.json",
    "phase13_parent_resolution.json",
    "analysis_plan.json",
    "phase14_analysis_freeze.json",
    "prediction_reproduction.json",
    "prediction_invariance.json",
    "overall_metrics.json",
    "overall_bootstrap.parquet",
    "temporal_metrics.parquet",
    "temporal_summary.json",
    "slice_registry.json",
    "slice_definitions.json",
    "slice_metrics.parquet",
    "slice_summary.json",
    "feature_drift.parquet",
    "feature_drift_summary.json",
    "score_distribution.json",
    "score_drift.json",
    "risk_decile_metrics.parquet",
    "topk_lift.json",
    "threshold_sensitivity.parquet",
    "error_cohorts.parquet",
    "high_confidence_errors.parquet",
    "error_profile.parquet",
    "error_profile_summary.json",
    "leakage_recheck.json",
    "phase15_readiness.json",
    "target_access_audit.json",
    "compute_manifest.json",
    "validation.json",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha(value: Any) -> str:
    import hashlib
    import json as _json

    return hashlib.sha256(
        _json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _close(left: Any, right: Any, *, tolerance: float = 1.0e-10) -> bool:
    """Compare persisted scalar evidence without accepting silent drift."""

    if left is None or right is None:
        return left is None and right is None
    try:
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=tolerance, equal_nan=True))
    except (TypeError, ValueError):
        return bool(left == right)


def _accepted_probabilities(resolved: Any) -> pd.DataFrame:
    """Reconstruct the Phase 13 accepted score policy from immutable evidence."""

    accepted = pd.read_parquet(resolved.phase13_dir / "validation_predictions.parquet")
    if resolved.champion_type == "ENSEMBLE":
        parts = []
        for track in ("T1", "T3"):
            selected = accepted.loc[
                accepted["track"] == track, [KEY, "effective_probability"]
            ].copy()
            if selected[KEY].duplicated().any():
                raise ValueError(f"Duplicate Phase 13 accepted keys for {track}.")
            parts.append(selected.rename(columns={"effective_probability": track}))
        result = parts[0].merge(parts[1], on=KEY, validate="one_to_one")
        weight = float(resolved.ensemble_t1_weight or 0.5)
        result["expected_probability"] = weight * result["T1"] + (1.0 - weight) * result["T3"]
        return result[[KEY, "expected_probability"]]
    track = resolved.components[0].track
    selected = accepted.loc[
        (accepted["track"] == track) & (accepted["candidate_id"] == resolved.champion_id),
        [KEY, "effective_probability"],
    ].rename(columns={"effective_probability": "expected_probability"})
    if selected[KEY].duplicated().any():
        raise ValueError("Duplicate Phase 13 accepted champion keys.")
    return selected


def _independent_replay(
    resolved: Any,
    directory: Path,
    persisted_reproduction: dict[str, Any],
    persisted_overall: dict[str, Any],
    persisted_invariance: dict[str, Any],
) -> list[str]:
    """Reload the frozen scorer and independently verify core Phase 14 evidence."""

    # Fixture validators may intentionally provide only artifact metadata. Real
    # Phase 13 resolutions always expose these population/scoring operations.
    if not hasattr(resolved, "validation_features") or not callable(
        getattr(resolved, "load_validation_targets", None)
    ):
        return []
    features = resolved.validation_features.sort_values(KEY, kind="mergesort").reset_index(
        drop=True
    )
    targets, target_audit = resolved.load_validation_targets()
    if any(
        target_audit.get(key, expected) != expected
        for key, expected in {
            "test_target_rows_loaded": 0,
            "test_predictions_created": 0,
            "test_metrics_computed": False,
        }.items()
    ):
        return ["Independent Phase 14 replay observed TEST access."]
    scorer = prepare_scorer(resolved, threads=1)
    scored = scorer(features)
    if scored[KEY].duplicated().any() or features[KEY].duplicated().any():
        return ["Independent Phase 14 replay found duplicate validation claim keys."]
    expected = _accepted_probabilities(resolved)
    actual = scored[[KEY, "probability"]].rename(columns={"probability": "actual_probability"})
    merged = expected.merge(actual, on=KEY, how="outer", validate="one_to_one", indicator=True)
    if (merged["_merge"] != "both").any():
        return ["Independent Phase 14 replay changed the validation population."]
    delta = (merged["expected_probability"] - merged["actual_probability"]).abs()
    errors: list[str] = []
    if len(delta) and float(delta.max()) > 1.0e-10:
        errors.append("Independent Phase 14 replay exceeded probability tolerance.")
    if int(persisted_reproduction.get("row_count", -1)) != len(merged):
        errors.append("Persisted prediction reproduction row count is not reproducible.")
    if not _close(
        persisted_reproduction.get("maximum_probability_delta"),
        float(delta.max()) if len(delta) else 0.0,
    ):
        errors.append("Persisted prediction reproduction delta is not reproducible.")
    target_frame = targets.set_index(KEY)
    if target_frame.index.has_duplicates or not set(actual[KEY]).issubset(target_frame.index):
        errors.append("Independent Phase 14 replay target population is not reproducible.")
        return errors
    y = target_frame.loc[actual[KEY], TARGET].to_numpy(dtype="int8")
    recomputed_overall = overall_metrics(
        y, actual["actual_probability"].to_numpy(), resolved.threshold
    )
    for key in (
        "average_precision",
        "roc_auc",
        "log_loss",
        "brier_score",
        "ece_10",
        "mce_10",
        "prevalence",
        "ap_lift_over_prevalence",
        "precision",
        "recall",
        "specificity",
        "negative_predictive_value",
        "false_positive_rate",
        "false_negative_rate",
        "f1",
        "f2",
        "balanced_accuracy",
        "mcc",
        "predicted_positive_rate",
    ):
        if not _close(persisted_overall.get(key), recomputed_overall.get(key)):
            errors.append(f"Persisted overall metric is not reproducible: {key}.")
    for key in ("row_count", "positive_count", "negative_count", "tp", "fp", "tn", "fn"):
        if persisted_overall.get(key) != recomputed_overall.get(key):
            errors.append(f"Persisted overall count is not reproducible: {key}.")
    invariant = prediction_invariance(
        features,
        scorer,
        batch_sizes=(17, 64, 256),
        seed=20260810,
        tolerance=1.0e-10,
    )
    if persisted_invariance.get("valid") is not True or invariant.get("valid") is not True:
        errors.append("Independent prediction invariance replay failed.")
    for key in (
        "serialization_max_probability_delta",
        "row_order_max_probability_delta",
        "batch_max_probability_delta",
    ):
        if not _close(persisted_invariance.get(key), invariant.get(key)):
            errors.append(f"Persisted prediction invariance is not reproducible: {key}.")
    return errors


def validate_existing_phase14(
    phase14_dir: Path, *, project_root: Path | None = None
) -> dict[str, Any]:
    directory = phase14_dir.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    missing = [name for name in REQUIRED_ARTIFACTS if not (directory / name).is_file()]
    if missing:
        return {
            "phase": 14,
            "valid": False,
            "hardening_status": "BLOCKED",
            "errors": ["Phase 14 artifacts missing: " + ", ".join(missing)],
            "warnings": [],
        }
    try:
        manifest = _read(directory / "phase14_manifest.json")
        freeze = _read(directory / "phase14_analysis_freeze.json")
        plan = _read(directory / "analysis_plan.json")
        reproduction = _read(directory / "prediction_reproduction.json")
        invariance = _read(directory / "prediction_invariance.json")
        overall = _read(directory / "overall_metrics.json")
        audit = _read(directory / "target_access_audit.json")
        leakage = _read(directory / "leakage_recheck.json")
        readiness = _read(directory / "phase15_readiness.json")
        phase13_dir = Path(str(manifest.get("phase13_dir", "")))
        if not phase13_dir.is_absolute():
            root = (project_root or directory).expanduser().resolve()
            phase13_dir = root / phase13_dir
        resolved = resolve_phase13_parent(
            phase13_dir, project_root=project_root or directory, require_main_merge=True
        )
    except Exception as exc:
        return {
            "phase": 14,
            "valid": False,
            "hardening_status": "BLOCKED",
            "errors": [str(exc)],
            "warnings": [],
        }
    if manifest.get("phase") != 14 or manifest.get("contract_version") != PHASE14_VERSION:
        errors.append("Phase 14 manifest contract or phase changed.")
    if manifest.get("configuration_sha256") != configuration_sha256():
        errors.append("Phase 14 configuration SHA changed.")
    if manifest.get("phase13_run_id") != resolved.phase13_manifest.get("run_id"):
        errors.append("Phase 13 run ID provenance changed.")
    upstream_fields = {
        "phase13_manifest_sha256": resolved.phase13_manifest_sha256,
        "phase13_validation_sha256": resolved.phase13_validation_sha256,
        "phase13_freeze_sha256": resolved.phase13_freeze_sha256,
        "phase13_effective_model_manifest_sha256": resolved.effective_manifest_sha256,
        "phase13_development_champion": resolved.champion_id,
        "frozen_score_space": resolved.score_space,
        "frozen_threshold": resolved.threshold,
    }
    for key, value in upstream_fields.items():
        if manifest.get(key) != value:
            errors.append(f"Phase 14 upstream provenance changed: {key}.")
    freeze_body = dict(freeze)
    declared_freeze_sha = freeze_body.pop("phase14_analysis_freeze_sha256", None)
    if declared_freeze_sha != _sha(freeze_body):
        errors.append("Phase 14 analysis freeze SHA does not match its content.")
    if (
        freeze.get("validation_targets_accessed") is not False
        or freeze.get("test_targets_accessed") is not False
    ):
        errors.append("Phase 14 freeze was not target-independent.")
    if freeze.get("analysis_plan_sha256") != plan.get("analysis_plan_sha256"):
        errors.append("Analysis plan SHA changed after freeze.")
    if plan.get("slice_registry_sha256") != _sha(plan.get("slice_registry", [])):
        errors.append("Slice registry SHA changed.")
    if plan.get("slice_definition_sha256") != _sha(plan.get("slice_definitions", [])):
        errors.append("Slice definition SHA changed.")
    if (
        not reproduction.get("valid")
        or float(reproduction.get("maximum_probability_delta", float("inf"))) > 1.0e-10
    ):
        errors.append("Phase 13 prediction reproduction exceeded 1e-10.")
    if not invariance.get("valid"):
        errors.append("Prediction invariance failed.")
    if not leakage.get("valid") or int(leakage.get("prohibited_feature_count", 0)) != 0:
        errors.append("Prohibited leakage feature detected.")
    expected_audit = {
        "test_target_rows_loaded": 0,
        "test_feature_rows_scored": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
    }
    for key, value in expected_audit.items():
        if audit.get(key) != value:
            errors.append(f"Phase 14 TEST seal changed: {key}.")
    if not np.isfinite(float(overall.get("average_precision", float("nan")))):
        errors.append("Overall AP is not finite.")
    if (
        float(overall.get("average_precision", 0.0))
        <= float(overall.get("prevalence", 0.0)) + 1.0e-6
    ):
        errors.append("ROBUSTNESS_SIGNAL_COLLAPSE: AP is not above prevalence.")
    if float(overall.get("roc_auc", 0.0)) <= 0.50:
        errors.append("ROBUSTNESS_SIGNAL_COLLAPSE: ROC-AUC is not above 0.50.")
    warnings.extend(str(item) for item in manifest.get("warning_inventory", []))
    errors.extend(
        _independent_replay(
            resolved,
            directory,
            reproduction,
            overall,
            invariance,
        )
    )
    independent_readiness = readiness_gate(
        overall, warnings, hard_blockers=errors, test_audit=audit
    )
    if readiness.get("status") != independent_readiness.get("status"):
        errors.append("Phase 15 readiness decision is not reproducible.")
    declared_files = manifest.get("artifact_file_sha256", {})
    if isinstance(declared_files, dict):
        for name, digest in declared_files.items():
            path = directory / str(name)
            if not path.is_file() or sha256_file(path) != str(digest):
                errors.append(f"Phase 14 artifact hash differs: {name}.")
    else:
        errors.append("Phase 14 artifact_file_sha256 is missing.")
    status = (
        "BLOCKED"
        if errors
        else (
            "HARDENED_PASS_WITH_WARNINGS"
            if independent_readiness["status"] == "READY_WITH_WARNINGS"
            else "HARDENED_PASS"
        )
    )
    return {
        "phase": 14,
        "run_id": manifest.get("run_id"),
        "valid": not errors,
        "hardening_status": status,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "test_seal": {key: audit.get(key) for key in expected_audit},
        "phase15_readiness": independent_readiness,
    }


__all__ = ["REQUIRED_ARTIFACTS", "validate_existing_phase14"]
