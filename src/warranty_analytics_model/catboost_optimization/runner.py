"""Phase 10 planning, sequential optimization, finalist publication, and reports."""

from __future__ import annotations

import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..baseline_model.provenance import validate_runtime_dependency_constraints
from ..feature_mart.manifest import git_commit_sha, write_json
from ..paths import discover_repository_root
from .config import TRACK_TO_EXPERIMENT, load_optimization_settings, settings_payload
from .contract import validate_optimization_contract
from .finalists import fit_phase10_finalists
from .inner_folds import DATE, KEY, build_inner_fold_plan
from .input import (
    CLAIM_DATE,
    load_locked_phase9_inputs,
    load_train_targets_for_optimization,
    load_validation_targets_after_freeze,
)
from .manifest import artifact_hashes, freeze_payload_sha256, write_table
from .models import OptimizationError, Phase10Inputs, StudyResult
from .objective import baseline_search_parameters, evaluate_parameters
from .provenance import runtime_provenance
from .reporting import write_phase10_reports
from .selection import select_development_champion
from .study import require_trial_history_schema, run_track_study, study_history_sha256
from .validation import validate_optimization_directory


def _resolve(root: Path, value: Path | None, default: str) -> Path:
    path = value if value is not None else Path(default)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def phase10_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def phase10_contract_check(project_root: Path | None = None) -> dict[str, Any]:
    return validate_optimization_contract(project_root)


def _preserve_failed_run(temporary: Path, run_id: str, exc: Exception) -> Path:
    """Retain expensive intermediate artifacts and attach failure metadata."""

    failure = {
        "status": "FAILED",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "failed_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
    }
    try:
        write_json(temporary / "failure.json", failure)
        failed_directory = temporary.with_suffix(".failed")
        temporary.replace(failed_directory)
        return failed_directory
    except OSError:
        # Preserve the original exception and leave any existing artifacts in place.
        return temporary


def _train_rows(phase10_inputs: Phase10Inputs) -> pd.DataFrame:
    frame = phase10_inputs.development.loc[phase10_inputs.development["split"] == "TRAIN"].copy()
    return frame.sort_values(KEY, kind="mergesort").reset_index(drop=True)


def _fold_plan(
    phase10_inputs: Phase10Inputs,
    train_targets: pd.DataFrame,
    settings: Any,
) -> Any:
    train = _train_rows(phase10_inputs)
    metadata = train[[KEY, CLAIM_DATE]].rename(columns={CLAIM_DATE: DATE})
    return build_inner_fold_plan(
        metadata,
        train_targets,
        fractions=settings.inner_fold_fractions,
        minimum_train_positive=settings.minimum_train_positive,
        minimum_validation_positive=settings.minimum_validation_positive,
    )


def phase10_plan_check(
    phase9_dir: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Validate Phase 9, exact T1/T3 inputs, TRAIN targets, and inner-fold plan only."""

    root = discover_repository_root(project_root)
    contract = validate_optimization_contract(root)
    errors = list(contract.get("errors", []))
    warnings = list(contract.get("warnings", []))
    inputs: Phase10Inputs | None = None
    fold_plan = None
    train_targets = None
    try:
        settings = load_optimization_settings(root)
        inputs = load_locked_phase9_inputs(phase9_dir, project_root=root)
        train_targets, train_audit = load_train_targets_for_optimization(inputs)
        fold_plan = _fold_plan(inputs, train_targets, settings)
        for key, value in train_audit.items():
            if key.startswith("test_") and value not in (0, False, 15):
                errors.append(f"Phase 10 TRAIN target audit changed: {key}")
    except Exception as exc:
        errors.append(str(exc))
    return {
        "status": "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS"),
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "contract": contract,
        "inputs": inputs,
        "train_targets": train_targets,
        "inner_fold_plan": fold_plan,
    }


def _baseline_inner_metrics(
    phase10_inputs: Phase10Inputs,
    train_targets: pd.DataFrame,
    fold_plan: Any,
    settings: Any,
) -> dict[str, dict[str, Any]]:
    train_matrix = _train_rows(phase10_inputs)
    fixed = settings.fixed_parameters
    baseline: dict[str, dict[str, Any]] = {}
    for track in settings.tracks:
        experiment_id = TRACK_TO_EXPERIMENT[track]
        evaluation = evaluate_parameters(
            train_matrix,
            train_targets,
            phase10_inputs.feature_sets[experiment_id],
            fold_plan,
            fixed,
            baseline_search_parameters(phase10_inputs.root),
            threshold=settings.threshold,
            project_root=phase10_inputs.root,
        )
        baseline[track] = {
            **evaluation.aggregate,
            "training_seconds": evaluation.training_seconds,
            "parameters": evaluation.params,
        }
    return baseline


def _optimization_warnings(
    studies: dict[str, StudyResult],
    comparisons: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    settings: Any,
) -> list[str]:
    warnings: list[str] = []
    if not any(bool(item.get("optimized_beats_baseline")) for item in comparisons.values()):
        warnings.append("NO_OPTIMIZATION_GAIN")
    for track in settings.tracks:
        comparison = comparisons[track]
        if float(comparison["optimized_average_precision"]) < float(
            comparison["baseline_average_precision"]
        ):
            warnings.append(f"{track}_OPTIMIZATION_REGRESSION")
        if (
            float(studies[track].best_inner_metrics["std_average_precision"])
            > settings.instability_std_ap_warning
        ):
            warnings.append("INNER_CV_INSTABILITY")
    for item in metrics.values():
        if float(item["roc_auc"]) >= 0.98 or float(item["average_precision"]) >= 0.90:
            warnings.append("SUSPICIOUSLY_HIGH_OPTIMIZATION_PERFORMANCE")
    for study in studies.values():
        warnings.extend(study.warnings)
    return list(dict.fromkeys(warnings))


def build_phase10(
    phase9_dir: Path,
    *,
    output_dir: Path | None = None,
    report_dir: Path | None = None,
    run_id: str | None = None,
    overwrite: bool = False,
    no_report: bool = False,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Run exactly two independent 50-trial studies and publish two finalists."""

    root = discover_repository_root(project_root)
    settings = load_optimization_settings(root)
    contract = validate_optimization_contract(root)
    if not contract["valid"]:
        raise OptimizationError(
            "Phase 10 contract blocks optimization: " + "; ".join(contract["errors"])
        )
    runtime = runtime_provenance()
    dependencies = validate_runtime_dependency_constraints(
        root,
        runtime,
        include_optimization=True,
    )
    if not dependencies["valid"]:
        raise OptimizationError(
            "Phase 10 runtime dependency validation failed: " + "; ".join(dependencies["errors"])
        )
    plan = phase10_plan_check(phase9_dir, project_root=root)
    if not plan["valid"] or plan["inputs"] is None or plan["train_targets"] is None:
        raise OptimizationError("Phase 10 plan blocks optimization: " + "; ".join(plan["errors"]))
    phase10_inputs: Phase10Inputs = plan["inputs"]
    train_targets: pd.DataFrame = plan["train_targets"]
    fold_plan = plan["inner_fold_plan"]
    if fold_plan is None:
        raise OptimizationError("Phase 10 inner fold plan is missing.")
    output_root = _resolve(root, output_dir, settings.output_directory)
    report_root = _resolve(root, report_dir, settings.report_directory)
    selected_run_id = run_id or phase10_run_id()
    final_dir = output_root / selected_run_id
    if final_dir.exists() and not overwrite:
        raise OptimizationError(f"Completed Phase 10 run is immutable: {final_dir}")
    if final_dir.exists():
        shutil.rmtree(final_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".phase10_{selected_run_id}_{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    try:
        write_table(
            fold_plan.assignments, temporary / "inner_cv_folds.parquet", settings.compression
        )
        write_json(temporary / "inner_cv_manifest.json", fold_plan.manifest)
        baseline_inner = _baseline_inner_metrics(phase10_inputs, train_targets, fold_plan, settings)
        train_matrix = _train_rows(phase10_inputs)
        studies: dict[str, StudyResult] = {}
        for track in settings.tracks:
            studies[track] = run_track_study(
                track,
                train_matrix,
                train_targets,
                phase10_inputs.feature_sets[TRACK_TO_EXPERIMENT[track]],
                fold_plan,
                settings.fixed_parameters,
                trials=settings.trials_per_track,
                seed=settings.random_seed,
                n_startup_trials=settings.n_startup_trials,
                threshold=settings.threshold,
                project_root=root,
                baseline_inner_cv_metrics=baseline_inner[track],
            )
        history = pd.concat([study.trial_history for study in studies.values()], ignore_index=True)
        require_trial_history_schema(history)
        fold_metrics = pd.concat(
            [study.fold_metrics for study in studies.values()], ignore_index=True
        )
        write_table(
            history.sort_values(["track", "trial_number"], kind="mergesort").reset_index(drop=True),
            temporary / "trial_history.parquet",
            settings.compression,
        )
        write_table(
            fold_metrics.sort_values(
                ["track", "trial_number", "fold_id"], kind="mergesort"
            ).reset_index(drop=True),
            temporary / "trial_fold_metrics.parquet",
            settings.compression,
        )
        best_params = {
            track: {
                "study_name": studies[track].study_name,
                "best_trial_number": studies[track].best_trial_number,
                "best_params": studies[track].best_params,
                "best_param_sha256": studies[track].best_param_sha256,
                "best_inner_metrics": studies[track].best_inner_metrics,
                "baseline_inner_cv_metrics": baseline_inner[track],
            }
            for track in settings.tracks
        }
        write_json(temporary / "best_params.json", best_params)
        freeze = {
            "phase": 10,
            "phase9_run_id": phase10_inputs.phase9_manifest.get("run_id"),
            "tracks": {
                track: {
                    "study_name": studies[track].study_name,
                    "best_trial_number": studies[track].best_trial_number,
                    "best_params": studies[track].best_params,
                    "best_param_sha256": studies[track].best_param_sha256,
                    "best_inner_metrics": studies[track].best_inner_metrics,
                    "baseline_inner_cv_metrics": baseline_inner[track],
                }
                for track in settings.tracks
            },
            "trial_history_content_sha256": study_history_sha256(history),
            "inner_fold_content_sha256": fold_plan.content_sha256,
            "frozen_at_utc": datetime.now(UTC).isoformat(),
            "outer_validation_accessed": False,
        }
        freeze["study_freeze_sha256"] = freeze_payload_sha256(freeze)
        write_json(temporary / "study_freeze.json", freeze)
        validation_targets, validation_audit = load_validation_targets_after_freeze(
            phase10_inputs, study_frozen=True
        )
        predictions, optimized_metrics, baseline_metrics, finalist_manifest = fit_phase10_finalists(
            phase10_inputs,
            train_targets,
            validation_targets,
            studies,
            settings.fixed_parameters,
            temporary / "models",
            threshold=settings.threshold,
        )
        write_table(predictions, temporary / "validation_predictions.parquet", settings.compression)
        all_metrics = {**baseline_metrics, **optimized_metrics}
        comparisons = finalist_manifest["comparisons"]
        candidates = [
            {
                "candidate_id": candidate_id,
                "metrics": all_metrics[candidate_id],
                "feature_count": phase10_inputs.feature_sets[experiment_id].feature_count,
            }
            for candidate_id, experiment_id in (
                ("P9_E1_BASELINE", "E1"),
                ("P10_T1_E1_OPTIMIZED", "E1"),
                ("P9_E3_BASELINE", "E3"),
                ("P10_T3_E3_OPTIMIZED", "E3"),
            )
        ]
        champion = select_development_champion(candidates)
        warnings = _optimization_warnings(studies, comparisons, optimized_metrics, settings)
        validation_metrics_payload = {
            "primary_metric": "average_precision",
            "threshold": settings.threshold,
            "candidate_metrics": all_metrics,
            "baseline_comparisons": comparisons,
            "replacement_decisions": {
                track: {
                    "fallback_to_baseline": comparisons[track]["fallback_to_baseline"],
                    "optimized_beats_baseline": comparisons[track]["optimized_beats_baseline"],
                }
                for track in settings.tracks
            },
            "phase10_development_champion": champion["candidate_id"],
            "champion_selection_scope": "PHASE10_DEVELOPMENT_ONLY",
        }
        write_json(temporary / "validation_metrics.json", validation_metrics_payload)
        write_json(temporary / "model_manifest.json", finalist_manifest)
        target_audit: dict[str, Any] = {
            "train_target_rows_loaded": int(len(train_targets)),
            "validation_target_rows_loaded": int(len(validation_targets)),
            "target_hashes": {
                "train": train_targets.attrs.get("target_content_sha256"),
                "validation": validation_audit.get("target_content_sha256"),
            },
            "outer_validation_accessed_before_study_freeze": False,
            "outer_validation_accessed_after_study_freeze": True,
            "test_target_rows_loaded": 0,
            "test_predictions_created": 0,
            "test_metrics_computed": False,
            "test_target_access_allowed": False,
            "first_allowed_test_target_phase": 15,
            "development_target_source": "Phase 5 claim snapshot",
        }
        # The TRAIN loader returns the authoritative digest in its audit; retain it without re-scanning.
        train_targets, train_audit = load_train_targets_for_optimization(phase10_inputs)
        target_audit["target_hashes"]["train"] = train_audit["target_content_sha256"]
        write_json(temporary / "target_access_audit.json", target_audit)
        manifest_artifacts = [
            "inner_cv_folds.parquet",
            "inner_cv_manifest.json",
            "trial_history.parquet",
            "trial_fold_metrics.parquet",
            "study_freeze.json",
            "best_params.json",
            "validation_predictions.parquet",
            "validation_metrics.json",
            "target_access_audit.json",
            "model_manifest.json",
        ]
        optimization_manifest = {
            "phase": 10,
            "run_id": selected_run_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_commit_sha": git_commit_sha(root),
            "phase9_dir": str(phase10_inputs.phase9_dir),
            "phase9_run_id": phase10_inputs.phase9_manifest.get("run_id"),
            "phase9_hardened_status": "HARDENED_PASS",
            "input_hashes": phase10_inputs.phase9_manifest.get("input_hashes"),
            "frozen_membership": phase10_inputs.phase9_manifest.get("frozen_membership"),
            "outer_population": phase10_inputs.phase9_manifest.get("frozen_membership", {}).get(
                "counts", {}
            ),
            "feature_set_inventory": {
                track: {
                    "phase9_experiment_id": TRACK_TO_EXPERIMENT[track],
                    "feature_count": phase10_inputs.feature_sets[
                        TRACK_TO_EXPERIMENT[track]
                    ].feature_count,
                    "feature_set_sha256": phase10_inputs.feature_sets[
                        TRACK_TO_EXPERIMENT[track]
                    ].feature_set_sha256,
                }
                for track in settings.tracks
            },
            "settings": settings_payload(settings),
            "runtime_versions": runtime,
            "dependency_compatibility": dependencies,
            "inner_fold_content_sha256": fold_plan.content_sha256,
            "study_freeze_sha256": freeze["study_freeze_sha256"],
            "outer_validation_accessed": True,
            "target_hashes": target_audit["target_hashes"],
            "test_target_rows_loaded": 0,
            "test_predictions_created": 0,
            "test_metrics_computed": False,
            "test_target_access_allowed": False,
            "first_allowed_test_target_phase": 15,
            "phase10_development_champion": champion["candidate_id"],
            "warnings": warnings,
            "artifact_file_sha256": artifact_hashes(temporary, manifest_artifacts),
        }
        write_json(temporary / "optimization_manifest.json", optimization_manifest)
        pending_validation = {
            "status": "PENDING",
            "valid": False,
            "errors": [],
            "warnings": [],
            "hardening_status": "PENDING",
        }
        write_json(temporary / "validation.json", pending_validation)
        validation = validate_optimization_directory(temporary, project_root=root)
        if validation["errors"]:
            raise OptimizationError(
                "Phase 10 artifact validation blocks publication: "
                + "; ".join(validation["errors"])
            )
        write_json(temporary / "validation.json", validation)
        temporary.replace(final_dir)
    except Exception as exc:
        if temporary.exists():
            _preserve_failed_run(temporary, selected_run_id, exc)
        raise
    summary = {
        "status": validation["status"],
        "run_id": selected_run_id,
        "phase10_development_champion": champion["candidate_id"],
        "optimization_comparison": comparisons,
        "inner_cv_summary": fold_plan.manifest,
        "best_parameters": best_params,
        "validation_metrics": validation_metrics_payload,
        "validation": validation,
        "warnings": warnings,
        "run_directory": str(final_dir),
        "report_directory": str(report_root / selected_run_id),
        "runtime_versions": runtime,
        "dependency_compatibility": dependencies,
    }
    if not no_report:
        report_directory = report_root / selected_run_id
        write_phase10_reports(report_directory, summary)
    else:
        summary["report_directory"] = None
    return summary


def validate_existing_optimization_run(
    optimization_dir: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    return validate_optimization_directory(optimization_dir, project_root=project_root)
