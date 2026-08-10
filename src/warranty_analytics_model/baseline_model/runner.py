"""Phase 9 contract, training, atomic publication, and validation orchestration."""

from __future__ import annotations

import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..feature_mart.manifest import git_commit_sha, sha256_file, write_json, write_parquet
from ..paths import discover_repository_root
from .adapters import build_development_feature_frame, split_development_frame
from .config import load_baseline_settings, settings_payload
from .contract import load_baseline_contract, validate_baseline_contract
from .experiments import run_experiments
from .feature_sets import feature_sets_payload
from .input import phase9_plan_check
from .metrics import apply_ap_lift, performance_warnings, select_champion
from .models import BaselineModelError
from .reporting import write_phase9_reports
from .target import KEY, load_development_targets, target_summary
from .validation import validate_model_directory


def _resolve(root: Path, value: Path | None, default: str) -> Path:
    path = value if value is not None else Path(default)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def phase9_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def phase9_contract_check(project_root: Path | None = None) -> dict[str, Any]:
    return validate_baseline_contract(project_root)


def _metrics_payload(results: list[Any], champion_id: str) -> dict[str, Any]:
    return {
        "primary_metric": "average_precision",
        "champion_experiment_id": champion_id,
        "champion_selection_scope": "VALIDATION_ONLY",
        "experiments": {
            item.experiment_id: {
                "status": item.status,
                "model_type": item.model_type,
                "champion_eligible": item.experiment_id != "E0" and item.status == "SUCCESS",
                "feature_count": item.feature_set.feature_count if item.feature_set else 0,
                "metrics": item.metrics,
                "validation_probability_sha256": item.validation_probability_sha256,
                "training_seconds": item.training_seconds,
                "prediction_seconds": item.prediction_seconds,
                "warning": item.warning,
            }
            for item in results
        },
    }


def build_phase9(
    mart_dir: Path,
    split_dir: Path,
    structured_dir: Path,
    text_dir: Path,
    *,
    output_dir: Path | None = None,
    report_dir: Path | None = None,
    run_id: str | None = None,
    overwrite: bool = False,
    no_report: bool = False,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Train all fixed baselines and atomically publish a validated development run."""

    root = discover_repository_root(project_root)
    settings = load_baseline_settings(root)
    plan = phase9_plan_check(mart_dir, split_dir, structured_dir, text_dir, project_root=root)
    if not plan["valid"] or plan["inputs"] is None:
        raise BaselineModelError("Phase 9 plan blocks training: " + "; ".join(plan["errors"]))
    inputs = plan["inputs"]
    feature_sets = plan["feature_sets"]
    targets = load_development_targets(
        inputs.mart_dir / "claim_snapshot.parquet", inputs.assignments
    )
    development = build_development_feature_frame(inputs, feature_sets)
    output_root = _resolve(root, output_dir, settings.output_directory)
    report_root = _resolve(root, report_dir, settings.report_directory)
    selected_run_id = run_id or phase9_run_id()
    final_dir = output_root / selected_run_id
    if final_dir.exists() and not overwrite:
        raise BaselineModelError(f"Completed Phase 9 run is immutable: {final_dir}")
    if final_dir.exists():
        shutil.rmtree(final_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".phase9_{selected_run_id}_{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    try:
        results = run_experiments(
            development, targets, feature_sets, settings, temporary / "models"
        )
        metric_map = {
            item.experiment_id: item.metrics for item in results if item.status == "SUCCESS"
        }
        apply_ap_lift(metric_map)
        champion = select_champion(results)
        warnings = list(
            dict.fromkeys(list(plan["warnings"]) + performance_warnings(champion, metric_map["E0"]))
        )
        _, validation_rows = split_development_frame(development)
        prediction_parts = []
        for item in results:
            if item.status == "SUCCESS" and item.probabilities is not None:
                prediction_parts.append(
                    pd.DataFrame(
                        {
                            KEY: validation_rows[KEY].to_numpy(),
                            "experiment_id": item.experiment_id,
                            "probability": item.probabilities.to_numpy(dtype="float64"),
                        }
                    )
                )
        predictions = (
            pd.concat(prediction_parts, ignore_index=True)
            .sort_values(["experiment_id", KEY])
            .reset_index(drop=True)
        )
        prediction_meta = write_parquet(
            predictions,
            temporary / "validation_predictions.parquet",
            compression=settings.compression,
        )
        metrics_payload = _metrics_payload(results, champion.experiment_id)
        write_json(temporary / "validation_metrics.json", metrics_payload)
        write_json(temporary / "feature_sets.json", feature_sets_payload(feature_sets))
        write_json(
            temporary / "model_input_schema.json",
            {
                experiment_id: {
                    "ordered_features": list(spec.feature_names),
                    "numeric": list(spec.numeric_features),
                    "categorical": list(spec.categorical_features),
                    "boolean": list(spec.boolean_features),
                    "text": list(spec.text_features),
                }
                for experiment_id, spec in feature_sets.items()
            },
        )
        write_json(
            temporary / "target_access_audit.json", {**targets.audit, **target_summary(targets)}
        )
        model_payload = {
            "models": {
                item.experiment_id: {
                    "status": item.status,
                    "model_type": item.model_type,
                    "model_file": item.model_file,
                    "model_sha256": item.model_sha256,
                    "feature_set_sha256": item.feature_set.feature_set_sha256
                    if item.feature_set
                    else None,
                    "effective_parameters": item.effective_parameters,
                    "warning": item.warning,
                }
                for item in results
                if item.experiment_id != "E0"
            }
        }
        write_json(temporary / "model_manifest.json", model_payload)
        contract, contract_sha = load_baseline_contract(root)
        del contract
        phase5_content = inputs.phase5_manifest.get(
            "artifact_content_fingerprints",
            inputs.phase5_manifest.get("artifact_content_sha256", {}),
        )
        artifact_files = (
            "validation_predictions.parquet",
            "validation_metrics.json",
            "feature_sets.json",
            "model_input_schema.json",
            "target_access_audit.json",
            "model_manifest.json",
        )
        experiment_manifest = {
            "phase": 9,
            "run_id": selected_run_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_commit_sha": git_commit_sha(root),
            "contract_sha256": contract_sha,
            "input_directories": {
                "phase5": str(inputs.mart_dir),
                "phase6": str(inputs.split_dir),
                "phase7": str(inputs.structured_dir),
                "phase8": str(inputs.text_dir),
            },
            "input_hashes": {
                "phase5_claim_snapshot": phase5_content.get("claim_snapshot"),
                "phase6_split_assignment": inputs.frozen_membership["split_assignment_sha256"],
                "phase7_structured_features": inputs.phase7_manifest.get(
                    "artifact_content_sha256", {}
                ).get("structured_features"),
                "phase8_text_features": inputs.phase8_manifest.get(
                    "artifact_content_sha256", {}
                ).get("text_features"),
            },
            "frozen_membership": inputs.frozen_membership,
            "source_audit": inputs.source_audit,
            "target_summary": target_summary(targets),
            "test_seal": targets.audit,
            "settings": settings_payload(settings),
            "champion_experiment_id": champion.experiment_id,
            "prediction_artifact": prediction_meta,
            "artifact_file_sha256": {
                name: sha256_file(temporary / name) for name in artifact_files
            },
            "warnings": warnings,
            "production_approved": False,
        }
        write_json(temporary / "experiment_manifest.json", experiment_manifest)
        validation = validate_model_directory(temporary, project_root=root, inputs=inputs)
        if validation["errors"]:
            raise BaselineModelError(
                "Phase 9 artifact validation blocks publication: " + "; ".join(validation["errors"])
            )
        write_json(temporary / "validation.json", validation)
        temporary.replace(final_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    summary = {
        "status": "PASS WITH WARNINGS" if warnings else "PASS",
        "run_id": selected_run_id,
        "champion_experiment_id": champion.experiment_id,
        "experiments": metrics_payload["experiments"],
        "target_summary": target_summary(targets),
        "test_seal": targets.audit,
        "feature_sets": feature_sets_payload(feature_sets),
        "warnings": warnings,
        "run_directory": str(final_dir),
        "validation": validation,
    }
    if not no_report:
        report_path = report_root / selected_run_id
        write_phase9_reports(report_path, summary)
    else:
        report_path = None
    return {**summary, "report_directory": str(report_path) if report_path else None}


def validate_existing_model_run(
    model_dir: Path, *, project_root: Path | None = None
) -> dict[str, Any]:
    return validate_model_directory(model_dir, project_root=project_root)
