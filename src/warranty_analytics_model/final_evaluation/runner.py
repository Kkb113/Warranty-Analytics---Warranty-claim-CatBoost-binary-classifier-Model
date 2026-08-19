"""Phase 15 pre-TEST freeze and final untouched TEST evaluation runner."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ..feature_mart.manifest import sha256_file, write_json
from ..paths import discover_repository_root
from ..robustness_analysis.config import load_robustness_settings
from ..robustness_analysis.errors import build_error_context, error_cohorts, error_profile
from ..robustness_analysis.input import KEY, TARGET, prepare_scorer, train_oof_scores
from ..robustness_analysis.invariance import prediction_invariance
from ..robustness_analysis.ranking import risk_decile_metrics, topk_lift
from ..robustness_analysis.slices import evaluate_slices
from ..robustness_analysis.temporal import temporal_metrics
from .bootstrap import stratified_bootstrap
from .checkpoint import write_checkpoint
from .config import compute_plan, load_final_test_settings
from .contract import phase15_contract_check
from .errors import high_confidence_errors
from .input import (
    Phase15InputError,
    build_test_membership_audit,
    leakage_audit,
    load_test_targets_after_freeze,
    resolve_phase14_parent,
)
from .metrics import reliability_table, signal_status, test_metrics, validation_test_comparison
from .planning import (
    build_evaluation_plan,
    frozen_test_slice_definitions,
    test_slice_memberships,
)
from .provenance import artifact_hashes, canonical_sha, current_scientific_commit, json_safe
from .reporting import write_phase15_reports
from .scoring import build_final_model_policy, score_test_in_batches
from .status import final_model_status
from .validation import validate_existing_phase15


def phase15_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_PHASE15")


def _write_json(directory: Path, name: str, payload: Any) -> None:
    write_json(directory / name, json_safe(payload))


def _required_phase14_gate_error(exc: Exception) -> dict[str, Any]:
    return {
        "phase": 15,
        "valid": False,
        "status": "BLOCKED",
        "errors": [str(exc)],
        "warnings": [],
        "test_targets_accessed": False,
        "test_predictions_created": False,
        "test_metrics_computed": False,
    }


def phase15_plan_check(
    phase14_dir: Path,
    *,
    project_root: Path | None = None,
    max_workers: int | None = None,
    catboost_inference_threads: int | None = None,
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    """Run every target-independent Phase 15 gate without loading TEST labels."""

    root = discover_repository_root(project_root or phase14_dir)
    contract = phase15_contract_check(root)
    if not contract.get("valid"):
        return {**contract, "test_targets_accessed": False}
    try:
        settings = load_final_test_settings(root)
        execution = compute_plan(
            settings,
            max_workers=max_workers,
            bootstrap_replicates=bootstrap_replicates,
            catboost_inference_threads=catboost_inference_threads,
        )
        resolved = resolve_phase14_parent(
            phase14_dir, project_root=root, require_main_merge=True, validate_upstream=True
        )
        membership = build_test_membership_audit(
            resolved.phase13.phase12_lock.phase12_inputs.phase10_inputs.phase9_inputs.assignments,
            resolved.phase6_manifest,
            resolved.test_lock,
        )
        definitions = frozen_test_slice_definitions(resolved)
        leakage = leakage_audit(resolved)
        if not leakage.get("valid"):
            raise Phase15InputError("Phase 15 leakage audit failed.")
        plan = build_evaluation_plan(resolved, settings, membership, definitions, execution)
        return {
            "phase": 15,
            "valid": True,
            "status": "PASS WITH WARNINGS"
            if resolved.phase14_readiness.get("warnings")
            else "PASS",
            "phase14_run_id": resolved.phase14_manifest.get("run_id"),
            "phase13_run_id": resolved.phase13.phase13_manifest.get("run_id"),
            "final_model_policy": "REUSE_FROZEN_PHASE14_CHAMPION",
            "test_expected_row_count": int(len(resolved.test_features)),
            "test_membership": membership,
            "feature_schema_sha256": resolved.feature_schema_sha256,
            "model_sha256": {item.track: item.model_sha256 for item in resolved.components},
            "calibrator_sha256": {
                item.track: item.calibrator_sha256 for item in resolved.components
            },
            "feature_list_sha256": {
                item.track: item.feature_list_sha256 for item in resolved.components
            },
            "effective_score_space": resolved.score_space,
            "frozen_threshold": resolved.threshold,
            "bootstrap_policy": plan["bootstrap_policy"],
            "compute_plan": execution,
            "slice_definition_sha256": plan["slice_definition_sha256"],
            "leakage_recheck": leakage,
            "test_targets_accessed": False,
            "test_predictions_created": False,
            "test_metrics_computed": False,
            "errors": [],
            "warnings": list(resolved.phase14_readiness.get("warnings", [])),
        }
    except (OSError, KeyError, TypeError, ValueError, Phase15InputError) as exc:
        return _required_phase14_gate_error(exc)


def _freeze_payload(
    resolved: Any,
    plan: dict[str, Any],
    definitions: list[dict[str, Any]],
    membership_audit: dict[str, Any],
    final_policy: dict[str, Any],
    leakage: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "phase": 15,
        "phase14_run_id": resolved.phase14_manifest.get("run_id"),
        "phase14_manifest_sha256": resolved.phase14_manifest_sha256,
        "phase14_validation_sha256": resolved.phase14_validation_sha256,
        "phase14_freeze_sha256": resolved.phase14_freeze_sha256,
        "phase14_contract_sha256": resolved.phase14_contract_sha256,
        "phase14_configuration_sha256": resolved.phase14_configuration_sha256,
        "phase13_run_id": resolved.phase13.phase13_manifest.get("run_id"),
        "phase13_manifest_sha256": resolved.phase13_manifest_sha256,
        "phase13_validation_sha256": resolved.phase13.phase13_validation_sha256,
        "phase13_freeze_sha256": resolved.phase13.phase13_freeze_sha256,
        "champion_id": resolved.champion_id,
        "candidate_type": resolved.champion_type,
        "model_sha256": final_policy["model_sha256"],
        "calibrator_sha256": final_policy["calibrator_sha256"],
        "feature_list_sha256": final_policy["feature_list_sha256"],
        "feature_schema_sha256": resolved.feature_schema_sha256,
        "effective_score_space": resolved.score_space,
        "frozen_threshold": float(resolved.threshold),
        "test_membership": membership_audit,
        "test_expected_row_count": membership_audit["expected_test_row_count"],
        "test_slice_definition_sha256": canonical_sha(definitions),
        "test_slice_membership_sha256": None,
        "leakage_recheck_sha256": canonical_sha(leakage),
        "bootstrap_policy": plan["bootstrap_policy"],
        "top_k": plan["ranking_policy"]["top_k"],
        "invariance_policy": plan["invariance_policy"],
        "compute_plan": execution,
        "final_model_policy": final_policy["policy"],
        "scientific_git_commit_sha": current_scientific_commit(resolved.root),
        "test_targets_accessed": False,
        "test_predictions_created": False,
        "test_metrics_computed": False,
        "test_target_rows_loaded": 0,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
        "development_decisions_frozen": True,
        "freeze_immutable": True,
    }
    payload["phase15_evaluation_freeze_sha256"] = canonical_sha(payload)
    return payload


def _validation_metrics(phase14_dir: Path) -> dict[str, Any]:
    path = phase14_dir / "overall_metrics.json"
    if path.is_file():
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    raise Phase15InputError("Accepted Phase 14 overall metrics are missing.")


def _false_negative_summary(cohorts: pd.DataFrame) -> dict[str, Any]:
    negatives = cohorts.loc[cohorts["error_type"] == "FALSE_NEGATIVE"]
    positives = cohorts.loc[cohorts["target"] == 1]
    return {
        "false_negative_count": int(len(negatives)),
        "actual_positive_count": int(len(positives)),
        "false_negative_rate_among_actual_positives": float(len(negatives) / len(positives))
        if len(positives)
        else 0.0,
        "mean_false_negative_probability": float(negatives["probability"].mean())
        if len(negatives)
        else 0.0,
        "max_false_negative_probability": float(negatives["probability"].max())
        if len(negatives)
        else 0.0,
    }


def build_phase15(
    phase14_dir: Path,
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
    """Build and atomically publish one complete Phase 15 TEST evaluation."""

    root = discover_repository_root(project_root or phase14_dir)
    contract = phase15_contract_check(root)
    if not contract.get("valid"):
        raise Phase15InputError("Phase 15 contract is blocked: " + "; ".join(contract["errors"]))
    settings = load_final_test_settings(root)
    execution = compute_plan(
        settings,
        max_workers=max_workers,
        bootstrap_replicates=bootstrap_replicates,
        catboost_inference_threads=catboost_inference_threads,
    )
    # This call is intentionally before any TEST target loader and enforces the
    # Phase 14 merge/green-CI gate.
    resolved = resolve_phase14_parent(
        phase14_dir, project_root=root, require_main_merge=True, validate_upstream=True
    )
    selected_run_id = str(run_id or phase15_run_id())
    output_root = (output_dir or root / "artifacts" / "final_evaluation").expanduser().resolve()
    report_root = (
        (report_dir or root / "reports" / "phase15_final_evaluation").expanduser().resolve()
    )
    final_dir = output_root / selected_run_id
    work_dir = output_root / f".phase15_{selected_run_id}.work"
    if final_dir.exists():
        raise Phase15InputError(f"Published Phase 15 run is immutable: {final_dir}")
    if work_dir.exists() and not resume:
        raise Phase15InputError(f"Phase 15 work directory exists; use --resume: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    assignments = (
        resolved.phase13.phase12_lock.phase12_inputs.phase10_inputs.phase9_inputs.assignments
    )
    membership_audit = build_test_membership_audit(
        assignments, resolved.phase6_manifest, resolved.test_lock
    )
    definitions = frozen_test_slice_definitions(resolved)
    leakage = leakage_audit(resolved)
    if not leakage.get("valid"):
        raise Phase15InputError("Phase 15 leakage audit failed before TEST access.")
    final_policy = build_final_model_policy(resolved)
    plan = build_evaluation_plan(resolved, settings, membership_audit, definitions, execution)
    freeze = _freeze_payload(
        resolved, plan, definitions, membership_audit, final_policy, leakage, execution
    )
    _write_json(
        work_dir,
        "phase14_parent_resolution.json",
        {
            **resolved.phase14_manifest,
            "phase14_dir": str(resolved.phase14_dir),
            "phase14_validation": resolved.phase14_validation,
            "phase14_readiness": resolved.phase14_readiness,
        },
    )
    _write_json(work_dir, "final_model_policy.json", final_policy)
    _write_json(work_dir, "phase15_test_slice_definitions.json", {"definitions": definitions})
    _write_json(work_dir, "phase15_evaluation_plan.json", plan)
    _write_json(work_dir, "test_membership_audit.json", membership_audit)
    _write_json(work_dir, "leakage_recheck.json", leakage)
    _write_json(work_dir, "phase15_evaluation_freeze.json", freeze)
    _write_json(work_dir, "compute_manifest.json", execution)
    write_checkpoint(
        work_dir / "checkpoints" / "pre_test.json",
        {
            "phase": 15,
            "run_id": selected_run_id,
            "phase15_evaluation_freeze_sha256": freeze["phase15_evaluation_freeze_sha256"],
            "test_targets_accessed": False,
            "result_sha256": canonical_sha(freeze),
        },
    )

    # Stage B starts only after the freeze has been persisted to disk.
    scored = score_test_in_batches(
        resolved,
        resolved.test_features,
        inference_threads=execution["catboost_inference_threads"],
    )
    scored_columns = [
        column
        for column in scored.columns
        if column in {KEY, "probability", "frozen_threshold", "predicted_class"}
        or str(column).endswith("_raw_probability")
        or str(column).endswith("_effective_probability")
    ]
    scored = scored[scored_columns].sort_values(KEY, kind="mergesort").reset_index(drop=True)
    scored.to_parquet(work_dir / "test_predictions.parquet", index=False)
    write_checkpoint(
        work_dir / "checkpoints" / "test_scoring.json",
        {
            "phase": 15,
            "run_id": selected_run_id,
            "freeze_sha256": freeze["phase15_evaluation_freeze_sha256"],
            "test_targets_accessed": False,
            "test_row_count": len(scored),
            "predictions_sha256": sha256_file(work_dir / "test_predictions.parquet"),
        },
    )
    probabilities = scored["probability"].astype(float)
    memberships = test_slice_memberships(definitions, resolved.test_features, probabilities)
    memberships.to_parquet(work_dir / "phase15_test_slice_memberships.parquet", index=False)
    freeze["test_slice_membership_sha256"] = sha256_file(
        work_dir / "phase15_test_slice_memberships.parquet"
    )
    # Risk-decile membership is computed from frozen probabilities, so its
    # digest can only be added after scoring.  Finalize and persist the freeze
    # now, still before the sole TEST-target loader is called; after this write
    # the freeze is never mutated again.
    freeze.pop("phase15_evaluation_freeze_sha256", None)
    freeze["phase15_evaluation_freeze_sha256"] = canonical_sha(freeze)
    _write_json(work_dir, "phase15_evaluation_freeze.json", freeze)
    write_checkpoint(
        work_dir / "checkpoints" / "pre_target_access.json",
        {
            "phase": 15,
            "run_id": selected_run_id,
            "phase15_evaluation_freeze_sha256": freeze["phase15_evaluation_freeze_sha256"],
            "test_targets_accessed": False,
            "slice_membership_sha256": freeze["test_slice_membership_sha256"],
        },
    )
    targets, target_audit = load_test_targets_after_freeze(resolved, freeze)
    target_audit["freeze_sha256"] = freeze["phase15_evaluation_freeze_sha256"]
    _write_json(work_dir, "target_access_audit.json", target_audit)
    keys = scored[KEY].astype(int)
    target_map = targets.set_index(KEY)[TARGET]
    aligned_targets = keys.map(target_map)
    if aligned_targets.isna().any():
        raise Phase15InputError("TEST target alignment is incomplete.")
    y = aligned_targets.to_numpy(dtype="int8")
    p = probabilities.to_numpy(dtype="float64")
    threshold = float(resolved.threshold)
    metrics = test_metrics(y, p, threshold)
    signal = signal_status(metrics)
    validation_metrics = _validation_metrics(resolved.phase14_dir)
    comparison = validation_test_comparison(
        validation_metrics,
        metrics,
        moderate_ap_ratio=settings.moderate_ap_ratio,
        moderate_roc_drop=settings.moderate_roc_drop,
    )
    threshold_metrics = {
        key: metrics[key]
        for key in (
            "threshold",
            "tp",
            "fp",
            "tn",
            "fn",
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
            "predicted_positive_count",
            "predicted_positive_rate",
        )
        if key in metrics
    }
    confusion = {
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "tn": int(metrics["tn"]),
        "fn": int(metrics["fn"]),
        "threshold": threshold,
    }
    topk = topk_lift(keys, y, p, tuple(settings.top_k))
    train_oof = train_oof_scores(resolved.phase13)
    test_frame = resolved.test_features.sort_values(KEY, kind="mergesort").reset_index(drop=True)
    deciles = risk_decile_metrics(train_oof, test_frame, pd.Series(y), pd.Series(p))
    concentration = {
        "positive_share_d10": float(
            deciles.loc[deciles["decile"] == "D10", "positive_count"].sum() / max(int(y.sum()), 1)
        ),
        "positive_share_d10_d9": float(
            deciles.loc[deciles["decile"].isin(["D10", "D9"]), "positive_count"].sum()
            / max(int(y.sum()), 1)
        ),
        "positive_share_d10_d8": float(
            deciles.loc[deciles["decile"].isin(["D10", "D9", "D8"]), "positive_count"].sum()
            / max(int(y.sum()), 1)
        ),
    }
    bootstrap_summary, bootstrap_rows = stratified_bootstrap(
        y,
        p,
        threshold,
        replicates=execution["test_bootstrap_replicates"],
        seed=settings.seed,
        workers=execution["bootstrap_workers"],
        confidence_level=settings.confidence_level,
    )
    reliability, reliability_summary = reliability_table(y, p)
    robust_settings = load_robustness_settings(root)
    temporal = temporal_metrics(
        test_frame, pd.Series(y), pd.Series(p), threshold, robust_settings, overall=metrics
    )
    slices, slice_summary = evaluate_slices(
        definitions, test_frame, pd.Series(y), pd.Series(p), threshold, metrics, robust_settings
    )
    cohorts = error_cohorts(keys, y, p, threshold)
    context = build_error_context(test_frame, definitions, p, resolved.feature_names)
    profile = error_profile(cohorts, context)
    high_confidence_errors(cohorts).to_parquet(
        work_dir / "high_confidence_errors.parquet", index=False
    )
    errors_summary = {
        "row_count": int(len(cohorts)),
        "false_positive_count": int((cohorts["error_type"] == "FALSE_POSITIVE").sum()),
        "false_negative_count": int((cohorts["error_type"] == "FALSE_NEGATIVE").sum()),
        "true_positive_count": int((cohorts["error_type"] == "TRUE_POSITIVE").sum()),
        "true_negative_count": int((cohorts["error_type"] == "TRUE_NEGATIVE").sum()),
        "threshold": threshold,
        "profile_row_count": int(len(profile)),
    }
    scorer = prepare_scorer(resolved.phase13, threads=execution["catboost_inference_threads"])
    fresh_scorer = prepare_scorer(resolved.phase13, threads=execution["catboost_inference_threads"])
    invariance = prediction_invariance(
        test_frame,
        scorer,
        fresh_scorer=fresh_scorer,
        batch_sizes=settings.batch_sizes,
        seed=settings.seed,
        tolerance=settings.probability_tolerance,
    )
    test_use_audit = {
        "phase": 15,
        "one_frozen_scoring_policy": True,
        "scoring_policy_count": 1,
        "model_selection_using_TEST": False,
        "threshold_tuning_using_TEST": False,
        "calibration_tuning_using_TEST": False,
        "ensemble_tuning_using_TEST": False,
        "feature_selection_using_TEST": False,
        "class_weight_tuning_using_TEST": False,
        "test_targets_accessed_after_freeze": True,
        "test_predictions_created": True,
        "test_metrics_computed": True,
    }
    warnings = list(resolved.phase14_readiness.get("warnings", []))
    if comparison.get("generalization_status") != "STABLE_GENERALIZATION":
        warnings.append(str(comparison["generalization_status"]))
    if not invariance.get("valid"):
        warnings.append("PREDICTION_INVARIANCE_FAILURE")
    status = final_model_status(
        signal,
        comparison,
        provenance_valid=True,
        leakage_valid=bool(leakage.get("valid")),
        scoring_valid=bool(invariance.get("valid")),
        test_use_valid=True,
        warnings=warnings,
    )
    _write_json(work_dir, "test_metrics.json", metrics)
    _write_json(work_dir, "test_signal_status.json", signal)
    _write_json(work_dir, "validation_test_comparison.json", comparison)
    _write_json(work_dir, "test_threshold_metrics.json", threshold_metrics)
    _write_json(work_dir, "test_confusion_matrix.json", confusion)
    _write_json(work_dir, "test_topk_lift.json", topk)
    deciles.to_parquet(work_dir / "test_risk_deciles.parquet", index=False)
    _write_json(work_dir, "ranking_concentration_summary.json", concentration)
    pd.DataFrame(bootstrap_rows).to_parquet(work_dir / "test_bootstrap.parquet", index=False)
    _write_json(work_dir, "test_bootstrap_summary.json", bootstrap_summary)
    reliability.to_parquet(work_dir / "test_reliability.parquet", index=False)
    _write_json(work_dir, "test_reliability_summary.json", reliability_summary)
    temporal.to_parquet(work_dir / "test_temporal_metrics.parquet", index=False)
    _write_json(work_dir, "test_temporal_summary.json", {"row_count": len(temporal)})
    slices.to_parquet(work_dir / "test_slice_metrics.parquet", index=False)
    _write_json(work_dir, "test_slice_summary.json", slice_summary)
    cohorts.to_parquet(work_dir / "test_error_cohorts.parquet", index=False)
    _write_json(work_dir, "test_error_summary.json", errors_summary)
    _write_json(work_dir, "false_negative_summary.json", _false_negative_summary(cohorts))
    profile.to_parquet(work_dir / "test_error_profile.parquet", index=False)
    _write_json(work_dir, "test_prediction_invariance.json", invariance)
    _write_json(work_dir, "test_use_audit.json", test_use_audit)
    _write_json(work_dir, "phase15_final_model_status.json", status)
    write_checkpoint(
        work_dir / "checkpoints" / "bootstrap.json",
        {
            "phase": 15,
            "run_id": selected_run_id,
            "freeze_sha256": freeze["phase15_evaluation_freeze_sha256"],
            "replicate_count": execution["test_bootstrap_replicates"],
            "bootstrap_sha256": sha256_file(work_dir / "test_bootstrap.parquet"),
        },
    )
    validation_payload = validate_existing_phase15(
        work_dir, project_root=root, allow_unpublished=True
    )
    _write_json(work_dir, "validation.json", validation_payload)
    if validation_payload.get("valid") is not True or validation_payload.get("errors"):
        raise Phase15InputError(
            "Independent Phase 15 validator failed; partial run is not published."
        )
    manifest_payload: dict[str, Any] = {
        "phase": 15,
        "run_id": selected_run_id,
        "scientific_git_commit_sha": current_scientific_commit(root),
        "contract_version": contract.get("contract_version"),
        "contract_sha256": contract.get("contract_sha256"),
        "configuration_sha256": contract.get("configuration_sha256"),
        "phase14_run_id": resolved.phase14_manifest.get("run_id"),
        "phase14_manifest_sha256": resolved.phase14_manifest_sha256,
        "phase14_validation_sha256": resolved.phase14_validation_sha256,
        "phase13_run_id": resolved.phase13.phase13_manifest.get("run_id"),
        "phase13_development_champion": resolved.champion_id,
        "final_model_policy": final_policy["policy"],
        "candidate_type": resolved.champion_type,
        "model_sha256": final_policy["model_sha256"],
        "calibrator_sha256": final_policy["calibrator_sha256"],
        "feature_list_sha256": final_policy["feature_list_sha256"],
        "effective_score_space": resolved.score_space,
        "frozen_threshold": threshold,
        "test_row_count": len(y),
        "test_positive_count": int(y.sum()),
        "test_negative_count": int((y == 0).sum()),
        "test_prevalence": metrics.get("prevalence"),
        "test_average_precision": metrics.get("average_precision"),
        "test_ap_lift": metrics.get("ap_lift_over_prevalence"),
        "test_roc_auc": metrics.get("roc_auc"),
        "test_log_loss": metrics.get("log_loss"),
        "test_brier": metrics.get("brier_score"),
        "threshold_precision": threshold_metrics.get("precision"),
        "threshold_recall": threshold_metrics.get("recall"),
        "threshold_mcc": threshold_metrics.get("mcc"),
        "threshold_f2": threshold_metrics.get("f2"),
        "topk_capture": topk.get("rows", []),
        "signal_status": signal.get("status"),
        "generalization_status": comparison.get("generalization_status"),
        "final_model_status": status.get("final_model_status"),
        "test_use_audit": test_use_audit,
        "phase15_evaluation_freeze_sha256": freeze["phase15_evaluation_freeze_sha256"],
        "artifact_file_sha256": {},
        "validation": validation_payload,
    }
    manifest_payload["artifact_file_sha256"] = artifact_hashes(
        work_dir, exclude={"phase15_manifest.json", "validation.json"}
    )
    _write_json(work_dir, "phase15_manifest.json", manifest_payload)
    # Re-validate including the manifest and publish only after every required
    # artifact and hash is present.
    final_validation = validate_existing_phase15(
        work_dir, project_root=root, allow_unpublished=True
    )
    _write_json(work_dir, "validation.json", final_validation)
    if final_validation.get("valid") is not True or final_validation.get("errors"):
        raise Phase15InputError("Final Phase 15 validation failed; partial run is not published.")
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir.replace(final_dir)
    report_payload = {
        "run_id": selected_run_id,
        "test_metrics": metrics,
        "validation_test_comparison": comparison,
        "test_topk_lift": topk,
        "ranking_concentration_summary": concentration,
        "test_threshold_metrics": threshold_metrics,
        "test_bootstrap_summary": bootstrap_summary,
        "temporal_summary": {"row_count": len(temporal)},
        "slice_summary": slice_summary,
        "test_error_summary": errors_summary,
        "phase15_final_model_status": status,
        "validation": final_validation,
    }
    report_directory = write_phase15_reports(report_root, selected_run_id, report_payload)
    return {
        "phase": 15,
        "status": status.get("final_model_status"),
        "run_directory": str(final_dir),
        "report_directory": str(report_directory),
        "validation": final_validation,
        "phase15_final_model_status": status,
        "test_metrics": metrics,
    }


__all__ = ["build_phase15", "phase15_plan_check", "phase15_run_id"]
