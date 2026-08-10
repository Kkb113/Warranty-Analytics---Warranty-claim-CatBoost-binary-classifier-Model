"""Focused acceptance test for the complete hardened Phase 9 bundle."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from warranty_analytics_model.baseline_model.models import DevelopmentTargets, FeatureSetSpec
from warranty_analytics_model.baseline_model.provenance import (
    prediction_content_sha256,
    validate_runtime_dependency_constraints,
)
from warranty_analytics_model.baseline_model.validation import validate_model_directory

KEY = "warranty_claim_key"
TARGET = "target__high_cost_claim_flag"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_complete_hardened_bundle_is_accepted(monkeypatch, tmp_path: Path) -> None:
    from warranty_analytics_model.baseline_model import validation
    from warranty_analytics_model.baseline_model.feature_sets import feature_sets_payload
    from warranty_analytics_model.baseline_model.metrics import apply_ap_lift, calculate_metrics
    from warranty_analytics_model.feature_mart.manifest import sha256_file

    specs = {
        experiment_id: FeatureSetSpec(
            experiment_id,
            ("number",),
            ("number",),
            (),
            (),
            (),
            1,
            0,
            0,
            0,
            f"feature-{experiment_id}",
        )
        for experiment_id in ("E1", "E2", "E3", "E4")
    }
    development = pd.DataFrame(
        {
            KEY: [1, 2, 3, 4, 5],
            "split": ["TRAIN", "TRAIN", "VALIDATION", "VALIDATION", "TEST"],
            "number": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    assignments = development[[KEY, "split"]].copy()
    targets = DevelopmentTargets(
        pd.DataFrame({KEY: [1, 2], TARGET: [0, 1]}),
        pd.DataFrame({KEY: [3, 4], TARGET: [0, 1]}),
        "train-digest",
        "validation-digest",
        {
            "train_target_rows_loaded": 2,
            "validation_target_rows_loaded": 2,
            "test_target_rows_loaded": 0,
            "test_predictions_created": 0,
            "test_metrics_computed": False,
            "test_target_access_allowed": False,
            "first_allowed_test_target_phase": 15,
        },
    )
    run_dir = tmp_path / "hardened"
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True)
    models: dict[str, dict[str, object]] = {}
    for experiment_id in specs:
        path = model_dir / f"{experiment_id.casefold()}.cbm"
        path.write_bytes(experiment_id.encode("ascii"))
        models[experiment_id] = {
            "status": "SUCCESS",
            "model_type": "CatBoostClassifier",
            "model_file": f"models/{path.name}",
            "model_sha256": sha256_file(path),
            "feature_set_sha256": specs[experiment_id].feature_set_sha256,
            "effective_parameters": {
                "loss_function": "Logloss",
                "iterations": 500,
                "learning_rate": 0.05,
                "depth": 6,
                "l2_leaf_reg": 3.0,
                "random_strength": 1.0,
                "bootstrap_type": "Bayesian",
                "bagging_temperature": 1.0,
                "random_seed": 20260810,
                "use_best_model": False,
                "auto_class_weights": "None",
            },
        }
    y_validation = np.array([0, 1])
    probability_by_experiment = {
        "E0": np.array([0.5, 0.5]),
        "E1": np.array([0.2, 0.8]),
        "E2": np.array([0.2, 0.8]),
        "E3": np.array([0.2, 0.8]),
        "E4": np.array([0.2, 0.8]),
    }
    metrics_by_experiment = {
        experiment_id: calculate_metrics(y_validation, probabilities)
        for experiment_id, probabilities in probability_by_experiment.items()
    }
    apply_ap_lift(metrics_by_experiment)
    prediction_rows = []
    for experiment_id, probabilities in probability_by_experiment.items():
        prediction_rows.extend(
            {
                KEY: key,
                "experiment_id": experiment_id,
                "probability": probability,
            }
            for key, probability in zip([3, 4], probabilities, strict=True)
        )
    predictions = pd.DataFrame(prediction_rows, columns=[KEY, "experiment_id", "probability"])
    predictions = predictions.sort_values(["experiment_id", KEY], kind="mergesort").reset_index(
        drop=True
    )
    predictions_path = run_dir / "validation_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    metrics_payload = {
        "primary_metric": "average_precision",
        "champion_experiment_id": "E1",
        "experiments": {
            experiment_id: {
                "status": "SUCCESS",
                "model_type": "constant_train_prevalence"
                if experiment_id == "E0"
                else "CatBoostClassifier",
                "champion_eligible": experiment_id != "E0",
                "feature_count": 0 if experiment_id == "E0" else 1,
                "metrics": metrics_by_experiment[experiment_id],
                "validation_probability_sha256": validation.probability_sha256(
                    probability_by_experiment[experiment_id]
                ),
            }
            for experiment_id in ("E0", "E1", "E2", "E3", "E4")
        },
    }
    _write_json(run_dir / "validation_metrics.json", metrics_payload)
    _write_json(run_dir / "feature_sets.json", feature_sets_payload(specs))
    _write_json(
        run_dir / "model_input_schema.json",
        {
            experiment_id: {
                "ordered_features": ["number"],
                "numeric": ["number"],
                "categorical": [],
                "boolean": [],
                "text": [],
            }
            for experiment_id in specs
        },
    )
    target_audit = {
        **targets.audit,
        "train": {
            "rows": 2,
            "positive_count": 1,
            "negative_count": 1,
            "target_content_sha256": "train-digest",
        },
        "validation": {
            "rows": 2,
            "positive_count": 1,
            "negative_count": 1,
            "target_content_sha256": "validation-digest",
        },
        "target_hashes": {"train": "train-digest", "validation": "validation-digest"},
    }
    _write_json(run_dir / "target_access_audit.json", target_audit)
    _write_json(run_dir / "model_manifest.json", {"models": models})
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    _write_json(report_dir / "baseline_model_report.json", {"champion_experiment_id": "E1"})
    artifact_files = (
        "validation_predictions.parquet",
        "validation_metrics.json",
        "feature_sets.json",
        "model_input_schema.json",
        "target_access_audit.json",
        "model_manifest.json",
    )
    runtime_versions = {
        "python_version": "3.12.8",
        "python_implementation": "CPython",
        "catboost_version": "1.2.10",
        "scikit_learn_version": "1.7.2",
        "pandas_version": "2.3.3",
        "numpy_version": "2.2.6",
        "pyarrow_version": "19.0.1",
        "platform": "fictional",
        "machine": "AMD64",
        "os": "Windows",
    }
    dependency_compatibility = validate_runtime_dependency_constraints(Path.cwd(), runtime_versions)
    assert dependency_compatibility["valid"] is True
    _write_json(
        run_dir / "experiment_manifest.json",
        {
            "hardening_version": "phase9_corrective_hardening_v1",
            "hardened_status": "HARDENED_PASS",
            "run_id": "hardened",
            "input_hashes": {
                "phase5_claim_snapshot": "p5",
                "phase6_split_assignment": "p6",
                "phase7_structured_features": "p7",
                "phase8_text_features": "p8",
            },
            "artifact_file_sha256": {name: sha256_file(run_dir / name) for name in artifact_files},
            "input_directories": {},
            "target_summary": {
                "train": {"target_content_sha256": "train-digest"},
                "validation": {"target_content_sha256": "validation-digest"},
            },
            "target_hashes": {"train": "train-digest", "validation": "validation-digest"},
            "test_seal": targets.audit,
            "runtime_versions": runtime_versions,
            "dependency_compatibility": dependency_compatibility,
            "champion_experiment_id": "E1",
            "report_directory": str(report_dir),
            "prediction_artifact": {
                "row_count": len(predictions),
                "column_count": 3,
                "canonical_content_sha256": prediction_content_sha256(predictions),
            },
        },
    )
    monkeypatch.setattr(validation, "resolve_feature_sets", lambda *args: specs)
    monkeypatch.setattr(validation, "load_development_targets", lambda *args: targets)
    monkeypatch.setattr(validation, "build_development_feature_frame", lambda *args: development)
    monkeypatch.setattr(
        validation,
        "load_model",
        lambda path: SimpleNamespace(
            get_all_params=lambda: models[
                path.stem.upper() if path.stem.upper() in models else "E1"
            ]["effective_parameters"]
        ),
    )
    monkeypatch.setattr(
        validation,
        "predict_probabilities",
        lambda model, matrix, spec: probability_by_experiment[spec.experiment_id],
    )
    result = validate_model_directory(
        run_dir,
        project_root=Path.cwd(),
        inputs=SimpleNamespace(
            **{
                "mart_dir": tmp_path,
                "assignments": assignments,
                "phase5_manifest": {"artifact_content_fingerprints": {"claim_snapshot": "p5"}},
                "phase6_manifest": {},
                "phase7_manifest": {"artifact_content_sha256": {"structured_features": "p7"}},
                "phase8_manifest": {"artifact_content_sha256": {"text_features": "p8"}},
                "frozen_membership": {"split_assignment_sha256": "p6"},
                "phase7_lineage": {},
                "phase8_lineage": {},
                "root": Path.cwd(),
            }
        ),
    )
    assert result["valid"] is True, result
    assert result["hardening_status"] == "HARDENED_PASS"
