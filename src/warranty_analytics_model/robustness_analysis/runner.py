"""Phase 14 diagnostic runner and target-independent plan command."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..feature_mart.manifest import sha256_file, write_json
from .bootstrap import stratified_bootstrap
from .checkpoint import load_checkpoint, write_checkpoint
from .config import (
    PHASE14_VERSION,
    Phase14Settings,
    compute_plan,
    configuration_sha256,
    load_robustness_settings,
)
from .contract import phase14_contract_check
from .drift import feature_drift, score_drift
from .errors import error_cohorts, error_profile, high_confidence_errors
from .input import (
    KEY,
    TARGET,
    Phase14InputError,
    Phase14Resolved,
    current_git_commit,
    prepare_scorer,
    resolve_phase13_parent,
    train_oof_scores,
)
from .invariance import prediction_invariance
from .leakage import leakage_recheck
from .metrics import overall_metrics
from .planning import build_analysis_plan
from .ranking import risk_decile_metrics, topk_lift
from .readiness import readiness_gate
from .slices import evaluate_slices
from .temporal import temporal_metrics
from .threshold_diagnostics import threshold_sensitivity


def phase14_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_PHASE14")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "phase14_manifest.json"
    }


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False)


def _checkpoint_bindings(
    resolved: Phase14Resolved,
    plan: dict[str, Any],
    task: str,
    input_claim_sha256: str,
) -> dict[str, Any]:
    """Return the immutable provenance binding shared by every checkpoint."""

    return {
        "phase": 14,
        "task": task,
        "phase13_manifest_sha256": resolved.phase13_manifest_sha256,
        "phase13_champion": resolved.champion_id,
        "model_sha256": {
            component.track: component.model_sha256 for component in resolved.components
        },
        "calibrator_sha256": {
            component.track: component.calibrator_sha256 for component in resolved.components
        },
        "analysis_plan_sha256": plan["analysis_plan_sha256"],
        "input_claim_sha256": input_claim_sha256,
        "configuration_sha256": configuration_sha256(),
    }


def _checkpoint_result_sha(outputs: list[Path]) -> str:
    return _canonical_sha(
        {path.name: sha256_file(path) for path in sorted(outputs) if path.is_file()}
    )


def _checkpoint_reusable(
    checkpoint_path: Path,
    outputs: list[Path],
    bindings: dict[str, Any],
) -> bool:
    checkpoint = load_checkpoint(checkpoint_path, bindings)
    if checkpoint is None or any(not path.is_file() for path in outputs):
        return False
    return checkpoint.get("result_sha256") == _checkpoint_result_sha(outputs)


def _write_stage_checkpoint(
    checkpoint_path: Path,
    outputs: list[Path],
    bindings: dict[str, Any],
) -> None:
    write_checkpoint(
        checkpoint_path,
        {
            **bindings,
            "output_files": [path.name for path in sorted(outputs)],
            "result_sha256": _checkpoint_result_sha(outputs),
        },
    )


def _reproduction(resolved: Phase14Resolved, validation_scores: pd.DataFrame) -> dict[str, Any]:
    accepted = pd.read_parquet(resolved.phase13_dir / "validation_predictions.parquet")
    if resolved.champion_type == "ENSEMBLE":
        parts = []
        for track in ("T1", "T3"):
            item = accepted.loc[accepted["track"] == track, [KEY, "effective_probability"]].copy()
            if item[KEY].duplicated().any():
                raise Phase14InputError(
                    f"Phase 13 accepted validation predictions contain duplicate keys for {track}."
                )
            item = item.rename(columns={"effective_probability": track})
            parts.append(item)
        expected = parts[0].merge(parts[1], on=KEY, validate="one_to_one")
        weight = float(resolved.ensemble_t1_weight or 0.5)
        expected["expected_probability"] = weight * expected["T1"] + (1.0 - weight) * expected["T3"]
    else:
        track = resolved.components[0].track
        expected = accepted.loc[
            accepted["track"] == track, [KEY, "candidate_id", "effective_probability"]
        ].rename(columns={"effective_probability": "expected_probability"})
        expected = expected.loc[
            expected["candidate_id"] == resolved.champion_id, [KEY, "expected_probability"]
        ]
        if expected[KEY].duplicated().any():
            raise Phase14InputError(
                "Phase 13 accepted champion predictions contain duplicate keys."
            )
    actual = validation_scores[[KEY, "probability"]].rename(
        columns={"probability": "actual_probability"}
    )
    merged = expected.merge(actual, on=KEY, how="outer", validate="one_to_one", indicator=True)
    if (merged["_merge"] != "both").any():
        raise Phase14InputError("Phase 13 validation prediction membership changed.")
    delta = (merged["expected_probability"] - merged["actual_probability"]).abs()
    mismatch = int((delta > 1.0e-10).sum())
    return {
        "phase13_run_id": resolved.phase13_manifest["run_id"],
        "row_count": int(len(merged)),
        "expected_row_count": int(len(expected)),
        "actual_row_count": int(len(actual)),
        "duplicate_claim_ids": int(actual[KEY].duplicated().sum()),
        "maximum_probability_delta": float(delta.max()) if len(delta) else 0.0,
        "mean_probability_delta": float(delta.mean()) if len(delta) else 0.0,
        "mismatching_rows": mismatch,
        "probability_tolerance": 1.0e-10,
        "valid": bool(mismatch == 0),
    }


def _population_audit(
    resolved: Phase14Resolved, validation_audit: dict[str, Any], validation_rows: int
) -> dict[str, Any]:
    return {
        "phase": 14,
        "train_target_rows_loaded": int(len(resolved.train_targets)),
        "validation_target_rows_loaded_before_phase14_freeze": 0,
        "validation_target_rows_loaded_after_phase14_freeze": int(validation_rows),
        "validation_targets_accessed": True,
        "test_target_rows_loaded": 0,
        "test_feature_rows_scored": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
        "upstream_validation_audit": validation_audit,
    }


def _phase14_freeze(
    resolved: Phase14Resolved, plan: dict[str, Any], settings: Phase14Settings
) -> dict[str, Any]:
    body = {
        "phase": 14,
        "phase13_run_id": resolved.phase13_manifest["run_id"],
        "phase13_manifest_sha256": resolved.phase13_manifest_sha256,
        "phase13_validation_sha256": resolved.phase13_validation_sha256,
        "phase13_freeze_sha256": resolved.phase13_freeze_sha256,
        "phase13_effective_model_manifest_sha256": resolved.effective_manifest_sha256,
        "development_champion": resolved.champion_id,
        "champion_type": resolved.champion_type,
        "model_sha256": {
            component.track: component.model_sha256 for component in resolved.components
        },
        "calibrator_sha256": {
            component.track: component.calibrator_sha256 for component in resolved.components
        },
        "feature_list_sha256": {
            component.track: component.feature_list_sha256 for component in resolved.components
        },
        "score_space": resolved.score_space,
        "frozen_threshold": resolved.threshold,
        "slice_registry_sha256": plan["slice_registry_sha256"],
        "slice_definition_sha256": plan["slice_definition_sha256"],
        "temporal_definition_sha256": plan["temporal_definition_sha256"],
        "analysis_plan_sha256": plan["analysis_plan_sha256"],
        "bootstrap_policy": plan["bootstrap_policy"],
        "drift_policy": plan["drift_policy"],
        "threshold_diagnostic_policy": plan["threshold_diagnostic_policy"],
        "readiness_policy": plan["readiness_policy"],
        "development_decisions_frozen": True,
        "model_changes_after_phase14_analysis": "prohibited",
        "validation_targets_accessed": False,
        "test_targets_accessed": False,
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
    }
    body["phase14_analysis_freeze_sha256"] = _canonical_sha(body)
    return body


def _warning_inventory(
    overall: dict[str, Any],
    temporal: pd.DataFrame,
    slices: pd.DataFrame,
    drift: pd.DataFrame,
    score_shift: dict[str, Any],
    errors: pd.DataFrame,
) -> list[str]:
    warnings = ["SYNTHETIC_POC", "BUSINESS_TARGET_UNCONFIRMED"]
    if int(overall.get("positive_count", 0)) < 100:
        warnings.append("SMALL_VALIDATION_POSITIVE_COUNT")
    if (
        not temporal.empty
        and (
            temporal.get("stability_classification", pd.Series(dtype=str)) == "MODERATE_DEGRADATION"
        ).any()
    ):
        warnings.append("TEMPORAL_DEGRADATION")
    if (
        not temporal.empty
        and (
            temporal.get("stability_classification", pd.Series(dtype=str)) == "SEVERE_DEGRADATION"
        ).any()
    ):
        warnings.append("SEVERE_TEMPORAL_DEGRADATION")
    if not slices.empty:
        if (slices.get("status", pd.Series(dtype=str)) == "LOW_SUPPORT").any():
            warnings.append("LOW_SUPPORT_SLICE")
        if (
            slices.get("stability_classification", pd.Series(dtype=str)) == "MODERATE_DEGRADATION"
        ).any():
            warnings.append("SLICE_PERFORMANCE_DEGRADATION")
        if (
            slices.get("stability_classification", pd.Series(dtype=str)) == "SEVERE_DEGRADATION"
        ).any():
            warnings.append("SEVERE_SLICE_PERFORMANCE_DEGRADATION")
    if not drift.empty and (drift.get("psi", pd.Series(dtype=float)).fillna(0.0) >= 0.25).any():
        warnings.append("HIGH_FEATURE_DRIFT")
    if (
        not drift.empty
        and (
            drift.get("missingness_classification", pd.Series(dtype=str)).isin(
                ["MATERIAL_MISSINGNESS_SHIFT", "HIGH_MISSINGNESS_SHIFT"]
            )
        ).any()
    ):
        warnings.append("MATERIAL_MISSINGNESS_SHIFT")
    if score_shift.get("classification") == "SCORE_DISTRIBUTION_SHIFT":
        warnings.append("SCORE_DISTRIBUTION_SHIFT")
    if len(errors) and int((errors["error_type"] == "FALSE_NEGATIVE").sum()) > int(
        overall.get("positive_count", 0) * 0.75
    ):
        warnings.append("HIGH_FALSE_NEGATIVE_CONCENTRATION")
    return sorted(set(warnings))


def _reports(report_root: Path, run_id: str, summary: dict[str, Any]) -> Path:
    directory = report_root / run_id
    directory.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "phase": 14,
        "run_id": run_id,
        "status": summary.get("hardening_status"),
        "phase15_readiness": summary.get("phase15_readiness"),
        "overall_metrics": summary.get("overall_metrics"),
        "warnings": summary.get("warnings", []),
    }
    write_json(directory / "phase_14_summary.json", _json_safe(aggregate))
    write_json(
        directory / "robustness_summary.json",
        _json_safe(
            {
                "overall_metrics": summary.get("overall_metrics"),
                "bootstrap": summary.get("bootstrap"),
            }
        ),
    )
    write_json(directory / "temporal_summary.json", _json_safe(summary.get("temporal_summary", {})))
    write_json(directory / "slice_summary.json", _json_safe(summary.get("slice_summary", {})))
    write_json(directory / "drift_summary.json", _json_safe(summary.get("drift_summary", {})))
    write_json(
        directory / "error_analysis_summary.json",
        _json_safe(summary.get("error_analysis_summary", {})),
    )
    write_json(
        directory / "phase15_readiness.json", _json_safe(summary.get("phase15_readiness", {}))
    )
    write_json(directory / "validation.json", _json_safe(summary.get("validation", {})))
    markdown = [
        "# Phase 14 Robustness, Stability & Error Analysis",
        "",
        f"Run: `{run_id}`",
        "",
        f"Status: **{summary.get('hardening_status', 'UNKNOWN')}**",
        "",
        "This aggregate report contains no claim-level identifiers. Phase 14 is diagnostic only; no model-development decision was reopened.",
    ]
    (directory / "phase_14_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return directory


def build_phase14(
    phase13_dir: Path,
    *,
    project_root: Path | None = None,
    output_dir: Path | None = None,
    report_dir: Path | None = None,
    run_id: str | None = None,
    resume: bool = False,
    max_workers: int | None = None,
    bootstrap_replicates: int | None = None,
    catboost_inference_threads: int | None = None,
) -> dict[str, Any]:
    root = (project_root or Path.cwd()).expanduser().resolve()
    settings = load_robustness_settings(root)
    contract_result = phase14_contract_check(root)
    if not contract_result.get("valid"):
        raise Phase14InputError(
            "Phase 14 contract is blocked: " + "; ".join(contract_result.get("errors", []))
        )
    # Phase 14 is only admissible after the frozen Phase 13 implementation is
    # reachable from local ``main``.  This prevents a diagnostic artifact from
    # becoming an implicit acceptance of an unmerged development branch.
    resolved = resolve_phase13_parent(phase13_dir, project_root=root, require_main_merge=True)
    plan = build_analysis_plan(resolved, settings)
    execution = compute_plan(
        settings,
        max_workers=max_workers,
        bootstrap_replicates=bootstrap_replicates,
        catboost_inference_threads=catboost_inference_threads,
    )
    output_root = (output_dir or root / "artifacts" / "robustness_analysis").expanduser().resolve()
    report_root = (
        (report_dir or root / "reports" / "phase14_robustness_error_analysis")
        .expanduser()
        .resolve()
    )
    selected_run_id = str(run_id or phase14_run_id())
    final_dir = output_root / selected_run_id
    if final_dir.exists():
        raise Phase14InputError(f"Published Phase 14 run is immutable: {final_dir}")
    work_dir = output_root / f".phase14_{selected_run_id}.work"
    if work_dir.exists() and not resume:
        raise Phase14InputError(
            f"Phase 14 work directory exists; use --resume or another run ID: {work_dir}"
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = work_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plan_path = work_dir / "analysis_plan.json"
    freeze_path = work_dir / "phase14_analysis_freeze.json"
    if resume and plan_path.is_file():
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if _canonical_sha(existing_plan) != _canonical_sha(plan):
            raise Phase14InputError("Phase 14 analysis plan changed; stale resume is blocked.")
    write_json(plan_path, _json_safe(plan))
    write_json(
        work_dir / "slice_registry.json",
        _json_safe({"slices": plan["slice_registry"], "sha256": plan["slice_registry_sha256"]}),
    )
    write_json(
        work_dir / "slice_definitions.json",
        _json_safe(
            {"definitions": plan["slice_definitions"], "sha256": plan["slice_definition_sha256"]}
        ),
    )
    parent_resolution = dict(resolved.parent_resolution)
    parent_resolution.update(
        {
            "phase13_commit": resolved.phase13_manifest.get("git_commit_sha"),
            "phase13_merged_to_main": True,
            "phase13_post_merge_ci_green": "required_external_evidence",
        }
    )
    write_json(work_dir / "phase13_parent_resolution.json", _json_safe(parent_resolution))
    freeze = _phase14_freeze(resolved, plan, settings)
    if resume and freeze_path.is_file():
        existing_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if _canonical_sha(existing_freeze) != _canonical_sha(freeze):
            raise Phase14InputError("Phase 14 analysis freeze changed; stale resume is blocked.")
    write_json(freeze_path, _json_safe(freeze))

    # Stage B starts here. The first target access is after the immutable freeze.
    validation_targets, validation_audit = resolved.load_validation_targets()
    validation_features = resolved.validation_features.reset_index(drop=True)
    train_features = resolved.train_features.reset_index(drop=True)
    validation_targets = validation_targets.sort_values(KEY, kind="mergesort").reset_index(
        drop=True
    )
    validation_features = validation_features.sort_values(KEY, kind="mergesort").reset_index(
        drop=True
    )
    train_features = train_features.sort_values(KEY, kind="mergesort").reset_index(drop=True)
    if (
        set(validation_targets[KEY]) != set(validation_features[KEY])
        or validation_features[KEY].duplicated().any()
    ):
        raise Phase14InputError("Phase 14 validation population differs from Phase 13 features.")
    input_claim_sha256 = _canonical_sha(
        validation_targets[[KEY, TARGET]].sort_values(KEY, kind="mergesort").to_dict("records")
    )
    scorer = prepare_scorer(resolved, threads=execution["catboost_inference_threads"])
    validation_scored = scorer(validation_features)
    validation_scores = validation_scored[[KEY, "probability"]]
    reproduction = _reproduction(resolved, validation_scores)
    write_json(work_dir / "prediction_reproduction.json", _json_safe(reproduction))
    y_validation = (
        validation_targets.set_index(KEY).loc[validation_scores[KEY], TARGET].to_numpy(dtype="int8")
    )
    overall = overall_metrics(
        y_validation, validation_scores["probability"].to_numpy(), resolved.threshold
    )
    write_json(work_dir / "overall_metrics.json", _json_safe(overall))
    bootstrap_path = work_dir / "overall_bootstrap.parquet"
    bootstrap_summary_path = work_dir / "overall_bootstrap_summary.json"
    bootstrap_checkpoint = checkpoint_dir / "overall_bootstrap.json"
    bootstrap_bindings = _checkpoint_bindings(
        resolved, plan, "overall_bootstrap", input_claim_sha256
    )
    if resume and _checkpoint_reusable(
        bootstrap_checkpoint, [bootstrap_path, bootstrap_summary_path], bootstrap_bindings
    ):
        bootstrap_summary = json.loads(bootstrap_summary_path.read_text(encoding="utf-8"))
        bootstrap_rows = pd.read_parquet(bootstrap_path).to_dict("records")
    else:
        bootstrap_summary, bootstrap_rows = stratified_bootstrap(
            y_validation,
            validation_scores["probability"].to_numpy(),
            resolved.threshold,
            replicates=execution["overall_bootstrap_replicates"],
            seed=settings.seed,
            workers=execution["bootstrap_workers"],
            confidence_level=settings.confidence_level,
        )
        _write_parquet(pd.DataFrame(bootstrap_rows), bootstrap_path)
        write_json(bootstrap_summary_path, _json_safe(bootstrap_summary))
        _write_stage_checkpoint(
            bootstrap_checkpoint,
            [bootstrap_path, bootstrap_summary_path],
            bootstrap_bindings,
        )
    temporal_path = work_dir / "temporal_metrics.parquet"
    temporal_summary_path = work_dir / "temporal_summary.json"
    temporal_checkpoint = checkpoint_dir / "temporal.json"
    temporal_bindings = _checkpoint_bindings(resolved, plan, "temporal", input_claim_sha256)
    if resume and _checkpoint_reusable(
        temporal_checkpoint, [temporal_path, temporal_summary_path], temporal_bindings
    ):
        temporal = pd.read_parquet(temporal_path)
        temporal_summary = json.loads(temporal_summary_path.read_text(encoding="utf-8"))
    else:
        temporal = temporal_metrics(
            validation_features,
            pd.Series(y_validation),
            validation_scores["probability"],
            resolved.threshold,
            settings,
            overall=overall,
        )
        _write_parquet(temporal, temporal_path)
        temporal_summary = {
            "rows": int(len(temporal)),
            "supported_periods": int(
                (temporal.get("status", pd.Series(dtype=str)) == "SUPPORTED").sum()
            )
            if not temporal.empty
            else 0,
            "strongest_period": str(
                temporal.sort_values("average_precision", ascending=False).iloc[0]["period"]
            )
            if not temporal.empty and "average_precision" in temporal
            else None,
            "weakest_supported_period": str(
                temporal.loc[temporal["status"] == "SUPPORTED"]
                .sort_values("average_precision")
                .iloc[0]["period"]
            )
            if not temporal.empty
            and (temporal.get("status", pd.Series(dtype=str)) == "SUPPORTED").any()
            else None,
        }
        write_json(temporal_summary_path, _json_safe(temporal_summary))
        _write_stage_checkpoint(
            temporal_checkpoint, [temporal_path, temporal_summary_path], temporal_bindings
        )
    slice_path = work_dir / "slice_metrics.parquet"
    slice_summary_path = work_dir / "slice_summary.json"
    slice_checkpoint = checkpoint_dir / "slice_family.json"
    slice_bindings = _checkpoint_bindings(resolved, plan, "slice_family", input_claim_sha256)
    if resume and _checkpoint_reusable(
        slice_checkpoint, [slice_path, slice_summary_path], slice_bindings
    ):
        slices = pd.read_parquet(slice_path)
        slice_summary = json.loads(slice_summary_path.read_text(encoding="utf-8"))
    else:
        slices, slice_summary = evaluate_slices(
            plan["slice_definitions"],
            validation_features,
            pd.Series(y_validation),
            validation_scores["probability"],
            resolved.threshold,
            overall,
            settings,
        )
        _write_parquet(slices, slice_path)
        write_json(slice_summary_path, _json_safe(slice_summary))
        _write_stage_checkpoint(slice_checkpoint, [slice_path, slice_summary_path], slice_bindings)
    categorical = set(
        name
        for component in resolved.components
        for name in component.feature_set.categorical_features + component.feature_set.text_features
    )
    drift_path = work_dir / "feature_drift.parquet"
    drift_summary_path = work_dir / "feature_drift_summary.json"
    drift_checkpoint = checkpoint_dir / "drift.json"
    drift_bindings = _checkpoint_bindings(resolved, plan, "drift", input_claim_sha256)
    if resume and _checkpoint_reusable(
        drift_checkpoint, [drift_path, drift_summary_path], drift_bindings
    ):
        drift = pd.read_parquet(drift_path)
        drift_summary = json.loads(drift_summary_path.read_text(encoding="utf-8"))
    else:
        drift = feature_drift(
            train_features, validation_features, list(resolved.feature_names), categorical
        )
        _write_parquet(drift, drift_path)
        drift_summary = {
            "model_features_evaluated": int(len(drift)),
            "high_psi_feature_count": int(
                (drift.get("psi", pd.Series(dtype=float)).fillna(0.0) >= 0.25).sum()
            )
            if not drift.empty
            else 0,
            "max_psi": float(drift["psi"].max()) if not drift.empty and "psi" in drift else 0.0,
            "max_missingness_shift": float(drift["missing_rate_delta"].abs().max())
            if not drift.empty and "missing_rate_delta" in drift
            else 0.0,
            "psi_thresholds_are_heuristic_monitoring_conventions": True,
        }
        write_json(drift_summary_path, _json_safe(drift_summary))
        _write_stage_checkpoint(drift_checkpoint, [drift_path, drift_summary_path], drift_bindings)
    oof = train_oof_scores(resolved)
    score_distribution_path = work_dir / "score_distribution.json"
    score_drift_path = work_dir / "score_drift.json"
    decile_path = work_dir / "risk_decile_metrics.parquet"
    topk_path = work_dir / "topk_lift.json"
    ranking_checkpoint = checkpoint_dir / "risk_deciles.json"
    ranking_outputs = [score_distribution_path, score_drift_path, decile_path, topk_path]
    ranking_bindings = _checkpoint_bindings(resolved, plan, "risk_deciles", input_claim_sha256)
    if resume and _checkpoint_reusable(ranking_checkpoint, ranking_outputs, ranking_bindings):
        score_distribution = json.loads(score_distribution_path.read_text(encoding="utf-8"))
        score_shift = json.loads(score_drift_path.read_text(encoding="utf-8"))
        deciles = pd.read_parquet(decile_path)
        topk = json.loads(topk_path.read_text(encoding="utf-8"))
    else:
        score_distribution, score_shift = score_drift(
            oof["probability"].to_numpy(), validation_scores["probability"].to_numpy()
        )
        write_json(score_distribution_path, _json_safe(score_distribution))
        write_json(score_drift_path, _json_safe(score_shift))
        deciles = risk_decile_metrics(
            oof, validation_features, pd.Series(y_validation), validation_scores["probability"]
        )
        _write_parquet(deciles, decile_path)
        topk = topk_lift(
            validation_scores[KEY],
            y_validation,
            validation_scores["probability"],
            tuple(settings.top_k),
        )
        write_json(topk_path, _json_safe(topk))
        _write_stage_checkpoint(ranking_checkpoint, ranking_outputs, ranking_bindings)
    sensitivity_path = work_dir / "threshold_sensitivity.parquet"
    cohorts_path = work_dir / "error_cohorts.parquet"
    confidence_path = work_dir / "high_confidence_errors.parquet"
    profile_path = work_dir / "error_profile.parquet"
    error_summary_path = work_dir / "error_profile_summary.json"
    error_checkpoint = checkpoint_dir / "error_analysis.json"
    error_outputs = [
        sensitivity_path,
        cohorts_path,
        confidence_path,
        profile_path,
        error_summary_path,
    ]
    error_bindings = _checkpoint_bindings(resolved, plan, "error_analysis", input_claim_sha256)
    if resume and _checkpoint_reusable(error_checkpoint, error_outputs, error_bindings):
        sensitivity = pd.read_parquet(sensitivity_path)
        cohorts = pd.read_parquet(cohorts_path)
        confidence = pd.read_parquet(confidence_path)
        profile = pd.read_parquet(profile_path)
        error_summary = json.loads(error_summary_path.read_text(encoding="utf-8"))
    else:
        sensitivity = threshold_sensitivity(
            y_validation,
            validation_scores["probability"],
            resolved.threshold,
            settings.threshold_multipliers,
        )
        _write_parquet(sensitivity, sensitivity_path)
        cohorts = error_cohorts(
            validation_scores[KEY],
            y_validation,
            validation_scores["probability"],
            resolved.threshold,
        )
        _write_parquet(cohorts, cohorts_path)
        confidence = high_confidence_errors(cohorts)
        _write_parquet(confidence, confidence_path)
        context = validation_features[[KEY, "claim__claim_date"]].copy()
        context["feature_missingness_count"] = (
            validation_features[list(resolved.feature_names)].isna().sum(axis=1)
            if resolved.feature_names
            else 0
        )
        profile = error_profile(cohorts, context)
        _write_parquet(profile, profile_path)
        error_summary = {
            "false_positives": int((cohorts["error_type"] == "FALSE_POSITIVE").sum()),
            "false_negatives": int((cohorts["error_type"] == "FALSE_NEGATIVE").sum()),
            "highest_confidence_fp": int((confidence["error_type"] == "FALSE_POSITIVE").sum()),
            "highest_confidence_fn": int((confidence["error_type"] == "FALSE_NEGATIVE").sum()),
        }
        write_json(error_summary_path, _json_safe(error_summary))
        _write_stage_checkpoint(error_checkpoint, error_outputs, error_bindings)
    invariance_path = work_dir / "prediction_invariance.json"
    invariance_checkpoint = checkpoint_dir / "invariance.json"
    invariance_bindings = _checkpoint_bindings(resolved, plan, "invariance", input_claim_sha256)
    if resume and _checkpoint_reusable(
        invariance_checkpoint, [invariance_path], invariance_bindings
    ):
        invariance = json.loads(invariance_path.read_text(encoding="utf-8"))
    else:
        invariance = prediction_invariance(
            validation_features,
            scorer,
            batch_sizes=settings.batch_sizes,
            seed=settings.seed,
            tolerance=settings.probability_tolerance,
        )
        write_json(invariance_path, _json_safe(invariance))
        _write_stage_checkpoint(invariance_checkpoint, [invariance_path], invariance_bindings)
    leakage = leakage_recheck(list(resolved.feature_names))
    write_json(work_dir / "leakage_recheck.json", _json_safe(leakage))
    warnings = _warning_inventory(overall, temporal, slices, drift, score_shift, cohorts)
    hard_blockers: list[str] = []
    if not reproduction["valid"]:
        hard_blockers.append("PHASE13_PREDICTION_REPRODUCTION_DELTA")
    if not invariance["valid"]:
        hard_blockers.append("PREDICTION_INVARIANCE_FAILURE")
    if not leakage["valid"]:
        hard_blockers.append("PROHIBITED_LEAKAGE_FEATURE")
    if validation_audit.get("test_target_rows_loaded", 0) != 0:
        hard_blockers.append("TEST_ACCESS_VIOLATION")
    readiness = readiness_gate(
        overall,
        warnings,
        hard_blockers=hard_blockers,
        test_audit={
            "test_target_rows_loaded": 0,
            "test_predictions_created": 0,
            "test_metrics_computed": False,
        },
    )
    write_json(work_dir / "phase15_readiness.json", _json_safe(readiness))
    population_audit = _population_audit(resolved, validation_audit, len(validation_targets))
    write_json(work_dir / "target_access_audit.json", _json_safe(population_audit))
    write_json(
        work_dir / "compute_manifest.json",
        _json_safe(
            {
                **execution,
                "phase13_inference_threads": execution["catboost_inference_threads"],
                "bootstrap_replicates": execution["overall_bootstrap_replicates"],
            }
        ),
    )
    hardening = (
        "HARDENED_PASS"
        if readiness["status"] == "READY"
        else (
            "HARDENED_PASS_WITH_WARNINGS"
            if readiness["status"] == "READY_WITH_WARNINGS"
            else "BLOCKED"
        )
    )
    manifest = {
        "phase": 14,
        "run_id": selected_run_id,
        "git_commit_sha": current_git_commit(root),
        "contract_version": PHASE14_VERSION,
        "contract_sha256": contract_result.get("contract_sha256"),
        "configuration_sha256": configuration_sha256(),
        "phase13_run_id": resolved.phase13_manifest["run_id"],
        "phase13_dir": str(resolved.phase13_dir),
        "phase13_manifest_sha256": resolved.phase13_manifest_sha256,
        "phase13_validation_sha256": resolved.phase13_validation_sha256,
        "phase13_freeze_sha256": resolved.phase13_freeze_sha256,
        "phase13_effective_model_manifest_sha256": resolved.effective_manifest_sha256,
        "phase13_development_champion": resolved.champion_id,
        "candidate_type": resolved.champion_type,
        "model_sha256": {
            component.track: component.model_sha256 for component in resolved.components
        },
        "calibrator_sha256": {
            component.track: component.calibrator_sha256 for component in resolved.components
        },
        "feature_list_sha256": {
            component.track: component.feature_list_sha256 for component in resolved.components
        },
        "frozen_score_space": resolved.score_space,
        "frozen_threshold": resolved.threshold,
        "analysis_freeze_sha256": freeze["phase14_analysis_freeze_sha256"],
        "overall_validation_metrics": overall,
        "bootstrap_configuration": bootstrap_summary,
        "warning_inventory": warnings,
        "phase15_readiness_status": readiness["status"],
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "artifact_file_sha256": {},
    }
    write_json(work_dir / "phase14_manifest.json", _json_safe(manifest))
    write_json(
        work_dir / "validation.json", {"phase": 14, "run_id": selected_run_id, "valid": False}
    )
    # The independent validator runs before publication and does not trust the
    # runner's readiness decision.
    from .validation import validate_existing_phase14

    preliminary = validate_existing_phase14(work_dir, project_root=root)
    validation_payload = {
        **preliminary,
        "phase": 14,
        "run_id": selected_run_id,
        "hardening_status": hardening,
    }
    write_json(work_dir / "validation.json", _json_safe(validation_payload))
    manifest["artifact_file_sha256"] = _artifact_hashes(work_dir)
    write_json(work_dir / "phase14_manifest.json", _json_safe(manifest))
    final_validation = validate_existing_phase14(work_dir, project_root=root)
    if not final_validation.get("valid"):
        raise Phase14InputError(
            "Independent Phase 14 validation failed: "
            + "; ".join(final_validation.get("errors", []))
        )
    os.replace(work_dir, final_dir)
    summary = {
        "hardening_status": hardening,
        "overall_metrics": overall,
        "bootstrap": bootstrap_summary,
        "temporal_summary": temporal_summary,
        "slice_summary": slice_summary,
        "drift_summary": drift_summary,
        "error_analysis_summary": error_summary,
        "phase15_readiness": readiness,
        "warnings": warnings,
        "validation": final_validation,
    }
    report_directory = _reports(report_root, selected_run_id, summary)
    return {
        "phase": 14,
        "run_id": selected_run_id,
        "run_directory": str(final_dir),
        "report_directory": str(report_directory),
        "hardening_status": hardening,
        "phase15_readiness": readiness,
        "phase13_development_champion": resolved.champion_id,
        "overall_metrics": overall,
        "warnings": warnings,
        "validation": final_validation,
        "compute_manifest": execution,
    }


def phase14_plan_check(phase13_dir: Path, *, project_root: Path | None = None) -> dict[str, Any]:
    """Stage A command: inspect features and freeze a plan without loading labels."""

    root = (project_root or Path.cwd()).expanduser().resolve()
    try:
        settings = load_robustness_settings(root)
        contract = phase14_contract_check(root)
        if not contract.get("valid"):
            return {
                "phase": 14,
                "valid": False,
                "status": "BLOCKED",
                "errors": contract.get("errors", []),
                "warnings": [],
            }
        resolved = resolve_phase13_parent(phase13_dir, project_root=root, require_main_merge=True)
        plan = build_analysis_plan(resolved, settings)
        return {
            "phase": 14,
            "valid": True,
            "status": "PASS",
            "errors": [],
            "warnings": [],
            "phase13_run_id": resolved.phase13_manifest["run_id"],
            "phase13_development_champion": resolved.champion_id,
            "analysis_plan": plan,
            "validation_targets_accessed": False,
            "test_targets_accessed": False,
        }
    except Exception as exc:
        return {
            "phase": 14,
            "valid": False,
            "status": "BLOCKED",
            "errors": [str(exc)],
            "warnings": [],
        }


__all__ = ["build_phase14", "phase14_plan_check", "phase14_run_id"]
