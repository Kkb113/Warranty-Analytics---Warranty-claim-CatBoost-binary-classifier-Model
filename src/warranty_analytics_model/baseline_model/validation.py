"""Independent validation of completed Phase 9 model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..feature_mart.manifest import sha256_file
from .adapters import adapt_matrix, build_development_feature_frame, split_development_frame
from .catboost_baseline import load_model, predict_probabilities
from .config import load_baseline_settings
from .feature_sets import resolve_feature_sets
from .input import load_phase9_inputs
from .metrics import calculate_metrics, prevalence_probabilities, probability_sha256
from .models import BaselineModelError, Phase9Inputs
from .target import KEY, TARGET, load_development_targets


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BaselineModelError(f"Expected a JSON object: {path}")
    return payload


def validate_model_directory(
    model_dir: Path,
    *,
    project_root: Path | None = None,
    inputs: Phase9Inputs | None = None,
) -> dict[str, Any]:
    """Recompute validation metrics and reload every persisted trained model."""

    directory = model_dir.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    required = (
        "validation_predictions.parquet",
        "validation_metrics.json",
        "feature_sets.json",
        "model_input_schema.json",
        "target_access_audit.json",
        "experiment_manifest.json",
        "model_manifest.json",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        return {
            "status": "BLOCKED",
            "valid": False,
            "errors": ["Missing artifacts: " + ", ".join(missing)],
            "warnings": [],
        }
    try:
        experiment_manifest = _read_json(directory / "experiment_manifest.json")
        metrics_payload = _read_json(directory / "validation_metrics.json")
        audit = _read_json(directory / "target_access_audit.json")
        model_manifest = _read_json(directory / "model_manifest.json")
        if inputs is None:
            paths = experiment_manifest["input_directories"]
            inputs = load_phase9_inputs(
                Path(paths["phase5"]),
                Path(paths["phase6"]),
                Path(paths["phase7"]),
                Path(paths["phase8"]),
                project_root=project_root,
            )
        phase5_content = inputs.phase5_manifest.get(
            "artifact_content_fingerprints",
            inputs.phase5_manifest.get("artifact_content_sha256", {}),
        )
        expected_inputs = {
            "phase5_claim_snapshot": phase5_content.get("claim_snapshot"),
            "phase6_split_assignment": inputs.frozen_membership.get("split_assignment_sha256"),
            "phase7_structured_features": inputs.phase7_manifest.get(
                "artifact_content_sha256", {}
            ).get("structured_features"),
            "phase8_text_features": inputs.phase8_manifest.get("artifact_content_sha256", {}).get(
                "text_features"
            ),
        }
        if experiment_manifest.get("input_hashes") != expected_inputs or any(
            value is None for value in expected_inputs.values()
        ):
            errors.append("Phase 9 input hashes are missing or differ from locked inputs.")
        for name, expected_hash in experiment_manifest.get("artifact_file_sha256", {}).items():
            path = directory / str(name)
            if not path.is_file() or sha256_file(path) != expected_hash:
                errors.append(f"Phase 9 artifact file hash differs: {name}")
        feature_sets = resolve_feature_sets(inputs.phase7_lineage, inputs.phase8_lineage)
        settings = load_baseline_settings(project_root)
        targets = load_development_targets(
            inputs.mart_dir / "claim_snapshot.parquet", inputs.assignments
        )
        development = build_development_feature_frame(inputs, feature_sets)
        train_rows, validation_rows = split_development_frame(development)
        y_train = (
            train_rows[[KEY]]
            .merge(targets.train, on=KEY, validate="one_to_one")[TARGET]
            .to_numpy(dtype="int8")
        )
        y_validation = (
            validation_rows[[KEY]]
            .merge(targets.validation, on=KEY, validate="one_to_one")[TARGET]
            .to_numpy(dtype="int8")
        )
        predictions = pd.read_parquet(directory / "validation_predictions.parquet")
        if list(predictions.columns) != [KEY, "experiment_id", "probability"]:
            errors.append("Validation prediction schema changed or exposes an unauthorized column.")
        validation_keys = set(validation_rows[KEY].astype(int))
        test_keys = set(
            inputs.assignments.loc[inputs.assignments["split"] == "TEST", KEY].astype(int)
        )
        if set(predictions[KEY].astype(int)) != validation_keys:
            errors.append("Validation prediction membership differs from Phase 6 VALIDATION.")
        if set(predictions[KEY].astype(int)) & test_keys:
            errors.append("TEST predictions were materialized.")
        stored_metrics = metrics_payload.get("experiments", {})
        for experiment_id, item in stored_metrics.items():
            if item.get("status") != "SUCCESS":
                continue
            subset = predictions.loc[predictions["experiment_id"] == experiment_id].sort_values(KEY)
            probabilities = subset["probability"].to_numpy(dtype="float64")
            expected_keys = validation_rows[KEY].sort_values().to_numpy()
            if not np.array_equal(subset[KEY].to_numpy(), expected_keys):
                errors.append(f"{experiment_id} validation prediction keys changed.")
                continue
            if experiment_id == "E0":
                expected_probabilities = prevalence_probabilities(y_train, len(y_validation))
            else:
                record = model_manifest.get("models", {}).get(experiment_id, {})
                path = directory / str(record.get("model_file", ""))
                if not path.is_file() or sha256_file(path) != record.get("model_sha256"):
                    errors.append(f"{experiment_id} model file hash differs from its manifest.")
                    continue
                spec = feature_sets[experiment_id]
                matrix = adapt_matrix(validation_rows, spec, settings)
                expected_probabilities = predict_probabilities(load_model(path), matrix, spec)
            if not np.allclose(probabilities, expected_probabilities, rtol=0.0, atol=1.0e-12):
                errors.append(f"{experiment_id} probabilities differ after model reload.")
            recomputed = calculate_metrics(y_validation, probabilities)
            for name, value in recomputed.items():
                stored = item["metrics"].get(name)
                if isinstance(value, float):
                    if stored is None or not np.isclose(
                        float(stored), value, rtol=0.0, atol=1.0e-12
                    ):
                        errors.append(f"{experiment_id} metric differs: {name}")
                elif stored != value:
                    errors.append(f"{experiment_id} metric differs: {name}")
            if probability_sha256(probabilities) != item.get("validation_probability_sha256"):
                errors.append(f"{experiment_id} probability content hash differs.")
        if any(
            audit.get(key) != expected
            for key, expected in {
                "test_target_rows_loaded": 0,
                "test_predictions_created": 0,
                "test_metrics_computed": False,
                "test_target_access_allowed": False,
                "first_allowed_test_target_phase": 15,
            }.items()
        ):
            errors.append("TEST target-access seal is invalid.")
        if any("test_target_content_sha256" in key for key in audit):
            errors.append("A prohibited TEST target hash was persisted.")
        if (directory / "test_predictions.parquet").exists():
            errors.append("A prohibited TEST prediction artifact exists.")
    except Exception as exc:
        errors.append(str(exc))
    return {
        "status": "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS"),
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "model_reload_probability_tolerance": 1.0e-12,
        "metrics_recomputed": not errors,
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
    }
