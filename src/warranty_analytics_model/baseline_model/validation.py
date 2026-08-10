"""Independent validation of completed Phase 9 model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..feature_mart.manifest import sha256_file
from ..text_features.contract import load_text_feature_contract
from .adapters import adapt_matrix, build_development_feature_frame, split_development_frame
from .catboost_baseline import effective_parameters, load_model, predict_probabilities
from .config import load_baseline_settings
from .feature_sets import audit_phase8_sources, feature_sets_payload, resolve_feature_sets
from .input import _audit_phase7_sources, load_phase9_inputs
from .metrics import (
    calculate_metrics,
    prevalence_probabilities,
    probability_sha256,
    select_champion,
)
from .models import BaselineModelError, ExperimentResult, Phase9Inputs
from .provenance import (
    EXPERIMENT_IDS,
    HARDENING_VERSION,
    model_policy_errors,
    prediction_content_sha256,
    runtime_provenance_errors,
    validate_runtime_dependency_constraints,
)
from .target import KEY, TARGET, load_development_targets

REQUIRED_ARTIFACTS = (
    "validation_predictions.parquet",
    "validation_metrics.json",
    "feature_sets.json",
    "model_input_schema.json",
    "target_access_audit.json",
    "experiment_manifest.json",
    "model_manifest.json",
)
ARTIFACT_HASHED_FILES = (
    "validation_predictions.parquet",
    "validation_metrics.json",
    "feature_sets.json",
    "model_input_schema.json",
    "target_access_audit.json",
    "model_manifest.json",
)
TEST_SEAL = {
    "test_target_rows_loaded": 0,
    "test_predictions_created": 0,
    "test_metrics_computed": False,
    "test_target_access_allowed": False,
    "first_allowed_test_target_phase": 15,
}
TARGET_HASH_KEY_MARKERS = ("sha", "hash", "digest", "fingerprint", "content")
FORBIDDEN_FEATURE_TOKENS = {
    KEY,
    TARGET,
    "split",
    "claim_date",
    "target",
    "outcome",
    "high_cost_claim_flag",
    "total_claim_cost",
    "labor_cost",
    "parts_cost",
    "diagnostic_cost",
    "towing_cost",
    "other_cost",
    "approved_amount",
    "rejected_amount",
    "customer_paid_amount",
    "repair_end_date",
    "days_to_repair",
    "claim_status",
    "root_cause_category",
    "repeat_claim_flag",
    "potential_recall_flag",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BaselineModelError(f"Expected a JSON object: {path}")
    return payload


def _close(left: Any, right: Any) -> bool:
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1.0e-12))
    return bool(left == right)


def _expected_input_hashes(inputs: Phase9Inputs) -> dict[str, Any]:
    phase5_content = inputs.phase5_manifest.get(
        "artifact_content_fingerprints",
        inputs.phase5_manifest.get("artifact_content_sha256", {}),
    )
    return {
        "phase5_claim_snapshot": phase5_content.get("claim_snapshot"),
        "phase6_split_assignment": inputs.frozen_membership.get("split_assignment_sha256"),
        "phase7_structured_features": inputs.phase7_manifest.get("artifact_content_sha256", {}).get(
            "structured_features"
        ),
        "phase8_text_features": inputs.phase8_manifest.get("artifact_content_sha256", {}).get(
            "text_features"
        ),
    }


def _validate_input_hashes(
    directory: Path,
    experiment_manifest: dict[str, Any],
    inputs: Phase9Inputs,
    *,
    hardened: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    expected = _expected_input_hashes(inputs)
    declared = experiment_manifest.get("input_hashes")
    if declared != expected or any(value is None for value in expected.values()):
        errors.append("Phase 9 input hashes are missing or differ from locked inputs.")
    artifact_hashes = experiment_manifest.get("artifact_file_sha256", {})
    if not isinstance(artifact_hashes, dict):
        errors.append("Phase 9 artifact_file_sha256 must be an object.")
        return
    missing = sorted(set(ARTIFACT_HASHED_FILES) - set(artifact_hashes))
    unexpected = sorted(set(artifact_hashes) - set(ARTIFACT_HASHED_FILES))
    if missing and hardened:
        errors.append("Phase 9 artifact hashes are missing: " + ", ".join(missing))
    if unexpected:
        errors.append("Phase 9 artifact hashes contain unexpected files: " + ", ".join(unexpected))
    if missing and not hardened:
        warnings.append("LEGACY_VALID: pre-hardening run lacks one or more artifact hashes.")
    for name, expected_hash in artifact_hashes.items():
        path = directory / str(name)
        if not path.is_file() or sha256_file(path) != expected_hash:
            errors.append(f"Phase 9 artifact file hash differs: {name}")


def _walk_target_hashes(
    value: Any,
    path: tuple[str, ...] = (),
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Find persisted TRAIN/VALIDATION target digests and reject TEST digests."""

    found: list[tuple[str, str, str]] = []
    errors: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.casefold()
            location = (*path, lowered)
            location_text = ".".join(location)
            target_hash_map = any(
                segment in {"target_hashes", "target_digests", "target_fingerprints"}
                for segment in path
            )
            is_target_hash = (
                (
                    "target" in lowered
                    and any(marker in lowered for marker in TARGET_HASH_KEY_MARKERS)
                )
                or target_hash_map
            ) and not isinstance(item, (dict, list))
            if is_target_hash:
                if "test" in location_text:
                    errors.append("A prohibited TEST target hash was persisted.")
                elif not isinstance(item, str) or not item:
                    errors.append(
                        f"Persisted target hash is not a non-empty string: {location_text}"
                    )
                elif "train" in location_text:
                    found.append(("train", item, location_text))
                elif "validation" in location_text:
                    found.append(("validation", item, location_text))
                else:
                    errors.append(f"Persisted target hash has no authorized split: {location_text}")
            nested_found, nested_errors = _walk_target_hashes(item, location)
            found.extend(nested_found)
            errors.extend(nested_errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            nested_found, nested_errors = _walk_target_hashes(item, (*path, str(index)))
            found.extend(nested_found)
            errors.extend(nested_errors)
    return found, errors


def _validate_target_metadata(
    directory: Path,
    audit: dict[str, Any],
    experiment_manifest: dict[str, Any],
    targets: Any,
    *,
    hardened: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    expected = {
        "train": targets.train_target_content_sha256,
        "validation": targets.validation_target_content_sha256,
    }
    found: list[tuple[str, str, str]] = []
    for path in sorted(directory.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid JSON while scanning target metadata: {path.name}: {exc}")
            continue
        nested_found, nested_errors = _walk_target_hashes(payload, (path.name,))
        found.extend(nested_found)
        errors.extend(nested_errors)
    if not found:
        if hardened:
            errors.append("No persisted TRAIN/VALIDATION target hashes were found.")
        else:
            warnings.append("LEGACY_VALID: pre-hardening run lacks persisted target hash metadata.")
    for split, value, location in found:
        if value != expected[split]:
            errors.append(f"Persisted {split.upper()} target hash differs: {location}")
    for split in expected:
        values = {value for found_split, value, _ in found if found_split == split}
        if len(values) > 1:
            errors.append(f"Persisted {split.upper()} target hashes disagree.")
        if not values and hardened:
            errors.append(f"Persisted {split.upper()} target hash is missing.")
    for split, frame in (("train", targets.train), ("validation", targets.validation)):
        block = audit.get(split)
        if isinstance(block, dict):
            if block.get("rows") != len(frame):
                errors.append(f"Persisted {split.upper()} target row count differs.")
            if block.get("positive_count") != int(frame[TARGET].sum()):
                errors.append(f"Persisted {split.upper()} positive count differs.")
            if block.get("negative_count") != int(len(frame) - frame[TARGET].sum()):
                errors.append(f"Persisted {split.upper()} negative count differs.")
    target_summary = experiment_manifest.get("target_summary")
    if hardened and not isinstance(target_summary, dict):
        errors.append("Experiment manifest target_summary is missing.")


def _validate_test_seal(
    directory: Path,
    audit: dict[str, Any],
    experiment_manifest: dict[str, Any],
    errors: list[str],
) -> None:
    seals = [audit]
    manifest_seal = experiment_manifest.get("test_seal")
    if isinstance(manifest_seal, dict):
        seals.append(manifest_seal)
    for seal in seals:
        if any(seal.get(key) != expected for key, expected in TEST_SEAL.items()):
            errors.append("TEST target-access seal is invalid.")
    for name in ("test_predictions.parquet", "test_metrics.json"):
        if any(path.name == name for path in directory.rglob(name)):
            errors.append(f"A prohibited TEST artifact exists: {name}")


def _feature_name_errors(feature_sets: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for experiment_id, spec in feature_sets.items():
        names = tuple(spec.feature_names)
        if len(names) != len(set(names)):
            errors.append(f"{experiment_id} feature names are not unique.")
        for name in names:
            lowered = name.casefold()
            if name in FORBIDDEN_FEATURE_TOKENS or any(
                token in lowered
                for token in (
                    "target__",
                    "outcome__",
                    "high_cost_claim_flag",
                    "total_claim_cost",
                )
            ):
                errors.append(f"{experiment_id} contains a target/outcome/control feature: {name}")
    return errors


def _expected_model_input_schema(feature_sets: dict[str, Any]) -> dict[str, Any]:
    return {
        experiment_id: {
            "ordered_features": list(spec.feature_names),
            "numeric": list(spec.numeric_features),
            "categorical": list(spec.categorical_features),
            "boolean": list(spec.boolean_features),
            "text": list(spec.text_features),
        }
        for experiment_id, spec in feature_sets.items()
    }


def _validate_feature_sets(
    persisted: dict[str, Any],
    model_input_schema: dict[str, Any],
    model_manifest: dict[str, Any],
    expected: dict[str, Any],
    *,
    hardened: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    expected_payload = feature_sets_payload(expected)
    if persisted != expected_payload:
        if hardened or persisted:
            errors.append("Persisted feature sets differ from recomputed Phase 7/8 lineage.")
        else:
            warnings.append("LEGACY_VALID: pre-hardening run lacks feature-set metadata.")
    errors.extend(_feature_name_errors(expected))
    if model_input_schema and model_input_schema != _expected_model_input_schema(expected):
        errors.append("Persisted model_input_schema differs from recomputed feature order.")
    elif hardened and not model_input_schema:
        errors.append("Persisted model_input_schema is missing.")
    models = model_manifest.get("models", {})
    if isinstance(models, dict):
        for experiment_id, record in models.items():
            if (
                experiment_id in expected
                and isinstance(record, dict)
                and record.get("feature_set_sha256") is not None
                and record.get("feature_set_sha256") != expected[experiment_id].feature_set_sha256
            ):
                errors.append(f"{experiment_id} model feature-set hash differs from lineage.")
            elif (
                experiment_id in expected
                and isinstance(record, dict)
                and record.get("feature_set_sha256") is None
                and hardened
            ):
                errors.append(f"{experiment_id} model feature-set hash is missing.")


def _validate_experiment_inventory(
    metrics_payload: dict[str, Any],
    model_manifest: dict[str, Any],
    directory: Path,
    *,
    hardened: bool,
    errors: list[str],
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    experiments = metrics_payload.get("experiments")
    if not isinstance(experiments, dict):
        errors.append("validation_metrics.experiments must be an object.")
        return {}
    actual_ids = set(experiments)
    if actual_ids != set(EXPERIMENT_IDS):
        message = "Phase 9 experiment inventory must be exactly E0, E1, E2, E3, and E4."
        if hardened:
            errors.append(message)
        else:
            warnings.append("LEGACY_VALID: pre-hardening run lacks complete experiment inventory.")
    statuses: dict[str, dict[str, Any]] = {}
    for experiment_id, item in experiments.items():
        if not isinstance(item, dict):
            errors.append(f"{experiment_id} experiment record is not an object.")
            continue
        status = item.get("status")
        legal = (experiment_id in {"E0", "E1", "E2", "E3"} and status == "SUCCESS") or (
            experiment_id == "E4" and status in {"SUCCESS", "UNAVAILABLE_WITH_WARNING"}
        )
        if not legal:
            errors.append(f"Illegal Phase 9 experiment status: {experiment_id}={status}")
        statuses[experiment_id] = item
    models = model_manifest.get("models", {})
    if not isinstance(models, dict):
        errors.append("model_manifest.models must be an object.")
        models = {}
    expected_model_ids = {
        experiment_id
        for experiment_id, item in statuses.items()
        if experiment_id != "E0" and item.get("status") == "SUCCESS"
    }
    if set(models) != expected_model_ids:
        if hardened or models:
            errors.append("Persisted model inventory does not match successful E1-E4 experiments.")
        else:
            warnings.append("LEGACY_VALID: pre-hardening run lacks complete model inventory.")
    expected_paths: set[str] = set()
    for experiment_id in sorted(expected_model_ids):
        record = models.get(experiment_id, {})
        if not isinstance(record, dict):
            errors.append(f"{experiment_id} model record is not an object.")
            continue
        model_file = record.get("model_file")
        if not isinstance(model_file, str) or not model_file.casefold().endswith(".cbm"):
            errors.append(f"{experiment_id} model file is missing or not CBM.")
            continue
        expected_paths.add(model_file.replace("\\", "/").casefold())
        path = directory / model_file
        if not path.is_file() or sha256_file(path) != record.get("model_sha256"):
            errors.append(f"{experiment_id} model file hash differs from its manifest.")
    actual_paths = {
        path.relative_to(directory).as_posix().casefold() for path in directory.rglob("*.cbm")
    }
    if actual_paths != expected_paths:
        errors.append("Persisted .cbm inventory contains an unexpected or missing model.")
    return statuses


def _validate_predictions(
    predictions: pd.DataFrame,
    statuses: dict[str, dict[str, Any]],
    validation_keys: set[int],
    train_keys: set[int],
    test_keys: set[int],
    experiment_manifest: dict[str, Any],
    *,
    hardened: bool,
    errors: list[str],
    warnings: list[str],
) -> str | None:
    columns = [KEY, "experiment_id", "probability"]
    if list(predictions.columns) != columns:
        errors.append("Validation prediction schema changed or exposes an unauthorized column.")
        return None
    if predictions.duplicated([KEY, "experiment_id"]).any():
        errors.append("Validation predictions contain duplicate experiment/key rows.")
    try:
        key_values = set(predictions[KEY].astype(int))
    except (TypeError, ValueError):
        errors.append("Validation prediction claim keys are not integer-like.")
        key_values = set()
    if key_values & train_keys:
        errors.append("TRAIN rows were materialized in validation predictions.")
    if key_values & test_keys:
        errors.append("TEST rows were materialized in validation predictions.")
    try:
        probabilities = pd.to_numeric(predictions["probability"], errors="coerce").to_numpy(
            dtype="float64"
        )
        if (
            not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
        ):
            errors.append("Validation predictions contain invalid probabilities.")
    except (TypeError, ValueError):
        errors.append("Validation predictions contain nonnumeric probabilities.")
    successful = {
        experiment_id for experiment_id, item in statuses.items() if item.get("status") == "SUCCESS"
    }
    prediction_ids = set(predictions["experiment_id"].astype(str))
    if prediction_ids - successful:
        errors.append("Validation predictions contain an unknown or unavailable experiment.")
    if successful - prediction_ids:
        errors.append("Validation predictions are missing a successful experiment.")
    for experiment_id in sorted(successful):
        subset = predictions.loc[predictions["experiment_id"].astype(str) == experiment_id]
        subset_keys = set(subset[KEY].astype(int)) if not subset.empty else set()
        if subset_keys != validation_keys or len(subset) != len(validation_keys):
            errors.append(
                f"{experiment_id} validation prediction membership differs from VALIDATION."
            )
    expected_rows = len(successful) * len(validation_keys)
    if len(predictions) != expected_rows:
        errors.append(
            "Validation prediction row count differs from successful experiment inventory."
        )
    ordered = predictions.sort_values(["experiment_id", KEY], kind="mergesort").reset_index(
        drop=True
    )
    if not predictions.reset_index(drop=True).equals(ordered):
        errors.append("Validation predictions are not in canonical experiment/key order.")
    try:
        canonical_hash = prediction_content_sha256(predictions)
    except BaselineModelError as exc:
        errors.append(str(exc))
        canonical_hash = None
    declared = experiment_manifest.get("prediction_artifact", {})
    if isinstance(declared, dict):
        if declared and declared.get("row_count") != len(predictions):
            errors.append("Prediction artifact row_count differs from the persisted Parquet.")
        if declared and declared.get("column_count") != len(columns):
            errors.append("Prediction artifact column_count differs from the persisted Parquet.")
        declared_canonical = declared.get("canonical_content_sha256")
        if declared_canonical is not None and declared_canonical != canonical_hash:
            errors.append("Prediction artifact canonical content hash differs.")
        elif hardened and declared_canonical is None:
            errors.append("Prediction artifact canonical content hash is missing.")
        elif declared and declared_canonical is None:
            warnings.append(
                "LEGACY_VALID: pre-hardening run lacks canonical prediction hash metadata."
            )
        elif not declared and hardened:
            errors.append("Prediction artifact metadata is missing.")
        elif not declared:
            warnings.append("LEGACY_VALID: pre-hardening run lacks prediction artifact metadata.")
    return canonical_hash


def _validate_configuration_policy(
    settings: Any, contract: dict[str, Any], errors: list[str]
) -> None:
    errors.extend(
        model_policy_errors(settings.catboost_parameters, context="Phase 9 configuration")
    )
    fixed = contract.get("fixed_catboost_parameters", {})
    if not isinstance(fixed, dict):
        errors.append("Phase 9 fixed_catboost_parameters is missing.")
    else:
        for key in ("eval_set", "early_stopping"):
            if str(fixed.get(key, "")).casefold() != "none":
                errors.append(f"Phase 9 {key} policy is not disabled.")
    imbalance = contract.get("class_imbalance_policy", {})
    if not isinstance(imbalance, dict) or any(
        str(value).casefold() != "none" for value in imbalance.values()
    ):
        errors.append("Phase 9 class weighting or resampling policy is active.")


def _validate_source_audit(
    inputs: Phase9Inputs, project_root: Path | None, errors: list[str]
) -> None:
    input_root = getattr(inputs, "root", None)
    phase8_contract, _ = load_text_feature_contract(project_root or input_root or Path.cwd())
    phase8_audit = audit_phase8_sources(inputs.phase8_lineage, phase8_contract)
    errors.extend(phase8_audit.get("errors", []))
    phase7_audit = _audit_phase7_sources(inputs.phase7_lineage)
    errors.extend(phase7_audit.get("errors", []))
    for name, item in inputs.phase8_lineage.items():
        if item.get("is_model_feature") is True:
            values = item.get("value_sources")
            if values != ["prior_failure__failure_description"]:
                errors.append(f"Phase 8 feature {name} has an unauthorized value_sources list.")
            if item.get("target_dependent") is not False or item.get("is_control") is True:
                errors.append(f"Phase 8 feature {name} is target-dependent or a control.")


def _report_summary_path(
    directory: Path, experiment_manifest: dict[str, Any], project_root: Path | None
) -> Path:
    report_value = experiment_manifest.get("report_directory")
    if isinstance(report_value, str) and report_value:
        return Path(report_value) / "baseline_model_report.json"
    root = project_root or directory.parents[2]
    return (
        root / "reports" / "phase9_baseline_models" / directory.name / "baseline_model_report.json"
    )


def _validate_champion(
    directory: Path,
    metrics_payload: dict[str, Any],
    experiment_manifest: dict[str, Any],
    report_path: Path,
    feature_sets: dict[str, Any],
    statuses: dict[str, dict[str, Any]],
    *,
    hardened: bool,
    allow_report_pending: bool,
    errors: list[str],
    warnings: list[str],
) -> str | None:
    if set(statuses) != set(EXPERIMENT_IDS):
        return None
    results: list[ExperimentResult] = []
    for experiment_id in EXPERIMENT_IDS:
        item = statuses[experiment_id]
        metrics = item.get("metrics", {})
        if not isinstance(metrics, dict):
            errors.append(f"{experiment_id} metrics are not an object.")
            return None
        results.append(
            ExperimentResult(
                experiment_id=experiment_id,
                model_type=str(item.get("model_type", "")),
                status=str(item.get("status", "")),
                feature_set=feature_sets.get(experiment_id),
                metrics=metrics,
                probabilities=None,
            )
        )
    try:
        champion = select_champion(results).experiment_id
    except BaselineModelError as exc:
        errors.append(str(exc))
        return None
    declared = metrics_payload.get("champion_experiment_id")
    if declared != champion:
        errors.append("Independent champion selection differs from validation_metrics.json.")
    if experiment_manifest.get("champion_experiment_id") != champion:
        errors.append("Independent champion selection differs from experiment_manifest.json.")
    if report_path.is_file():
        try:
            report = _read_json(report_path)
            if report.get("champion_experiment_id") != champion:
                errors.append("Independent champion selection differs from the report summary.")
        except (OSError, json.JSONDecodeError, BaselineModelError) as exc:
            errors.append(f"Phase 9 report summary is invalid: {exc}")
    elif hardened and not allow_report_pending:
        errors.append(f"Phase 9 report summary is missing: {report_path}")
    elif hardened:
        warnings.append("Report summary will be written after atomic Phase 9 publication.")
    else:
        warnings.append("LEGACY_VALID: pre-hardening report summary is unavailable.")
    return champion


def _validate_ap_lift(
    statuses: dict[str, dict[str, Any]],
    *,
    hardened: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    success = {
        experiment_id: item
        for experiment_id, item in statuses.items()
        if item.get("status") == "SUCCESS"
    }
    if "E0" not in success:
        return
    baseline = float(success["E0"].get("metrics", {}).get("average_precision", 0.0))
    for experiment_id, item in success.items():
        metrics = item.get("metrics", {})
        if not isinstance(metrics, dict) or "average_precision" not in metrics:
            errors.append(f"{experiment_id} average_precision is missing.")
            continue
        expected = float(metrics["average_precision"]) / baseline if baseline > 0 else None
        stored = metrics.get("ap_lift_over_prevalence_baseline")
        if expected is None:
            if stored is not None:
                errors.append(f"{experiment_id} AP lift should be null when E0 AP is zero.")
        elif stored is None or not _close(stored, expected):
            if hardened:
                errors.append(f"{experiment_id} AP lift differs from average_precision / E0 AP.")
            else:
                warnings.append(f"LEGACY_VALID: {experiment_id} lacks a valid AP-lift value.")


def _validate_reload_and_metrics(
    directory: Path,
    statuses: dict[str, dict[str, Any]],
    model_manifest: dict[str, Any],
    feature_sets: dict[str, Any],
    settings: Any,
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    predictions: pd.DataFrame,
    *,
    hardened: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    models = model_manifest.get("models", {})
    for experiment_id, item in statuses.items():
        if item.get("status") != "SUCCESS":
            continue
        subset = predictions.loc[
            predictions["experiment_id"].astype(str) == experiment_id
        ].sort_values(KEY, kind="mergesort")
        expected_keys = validation_rows[KEY].sort_values(kind="mergesort").to_numpy()
        if not np.array_equal(subset[KEY].to_numpy(), expected_keys):
            errors.append(f"{experiment_id} validation prediction keys changed.")
            continue
        probabilities = subset["probability"].to_numpy(dtype="float64")
        if experiment_id == "E0":
            expected_probabilities = prevalence_probabilities(y_train, len(y_validation))
        else:
            record = models.get(experiment_id, {})
            path = directory / str(record.get("model_file", ""))
            if not path.is_file() or sha256_file(path) != record.get("model_sha256"):
                errors.append(f"{experiment_id} model file hash differs from its manifest.")
                continue
            persisted_effective = record.get("effective_parameters")
            if isinstance(persisted_effective, dict):
                errors.extend(
                    model_policy_errors(
                        persisted_effective, context=f"{experiment_id} persisted model"
                    )
                )
            elif hardened:
                errors.append(f"{experiment_id} persisted effective_parameters are missing.")
            else:
                warnings.append(
                    f"LEGACY_VALID: {experiment_id} lacks effective parameter metadata."
                )
            spec = feature_sets[experiment_id]
            matrix = adapt_matrix(validation_rows, spec, settings)
            model = load_model(path)
            try:
                reloaded_effective = effective_parameters(model)
            except (AttributeError, TypeError):
                reloaded_effective = {}
            if reloaded_effective:
                errors.extend(
                    model_policy_errors(
                        reloaded_effective, context=f"{experiment_id} reloaded model"
                    )
                )
            elif hardened:
                errors.append(f"{experiment_id} reloaded effective parameters are unavailable.")
            expected_probabilities = predict_probabilities(model, matrix, spec)
        if not np.allclose(probabilities, expected_probabilities, rtol=0.0, atol=1.0e-12):
            errors.append(f"{experiment_id} probabilities differ after model reload.")
        recomputed = calculate_metrics(y_validation, probabilities)
        stored_metrics = item.get("metrics", {})
        for name, value in recomputed.items():
            stored = stored_metrics.get(name) if isinstance(stored_metrics, dict) else None
            if not _close(stored, value):
                errors.append(f"{experiment_id} metric differs: {name}")
        if probability_sha256(probabilities) != item.get("validation_probability_sha256"):
            errors.append(f"{experiment_id} probability content hash differs.")


def validate_model_directory(
    model_dir: Path,
    *,
    project_root: Path | None = None,
    inputs: Phase9Inputs | None = None,
    allow_report_pending: bool = False,
) -> dict[str, Any]:
    """Recompute validation metrics and independently audit every persisted artifact."""

    directory = model_dir.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    missing = [name for name in REQUIRED_ARTIFACTS if not (directory / name).is_file()]
    if missing:
        return {
            "status": "BLOCKED",
            "valid": False,
            "errors": ["Missing artifacts: " + ", ".join(missing)],
            "warnings": [],
            "hardening_status": "BLOCKED",
        }
    hardened = False
    canonical_prediction_hash: str | None = None
    champion: str | None = None
    try:
        experiment_manifest = _read_json(directory / "experiment_manifest.json")
        metrics_payload = _read_json(directory / "validation_metrics.json")
        persisted_feature_sets = _read_json(directory / "feature_sets.json")
        model_input_schema = _read_json(directory / "model_input_schema.json")
        audit = _read_json(directory / "target_access_audit.json")
        model_manifest = _read_json(directory / "model_manifest.json")
        hardened = experiment_manifest.get("hardening_version") == HARDENING_VERSION
        if hardened:
            runtime_versions = experiment_manifest.get("runtime_versions", {})
            errors.extend(runtime_provenance_errors(runtime_versions))
            dependency_compatibility = validate_runtime_dependency_constraints(
                Path(project_root) if project_root is not None else Path.cwd(), runtime_versions
            )
            errors.extend(dependency_compatibility["errors"])
            if experiment_manifest.get("dependency_compatibility") != dependency_compatibility:
                errors.append(
                    "Persisted Phase 9 dependency compatibility metadata differs from "
                    "the declared project requirements."
                )
            if experiment_manifest.get("hardened_status") != "HARDENED_PASS":
                errors.append("Hardened run does not declare HARDENED_PASS.")
        else:
            warnings.append(
                "LEGACY_VALID: pre-hardening Phase 9 run validated; missing corrective metadata is not corruption."
            )
            errors.extend(
                runtime_provenance_errors(
                    experiment_manifest.get("runtime_versions", {}), required=False
                )
            )
        if inputs is None:
            paths = experiment_manifest["input_directories"]
            inputs = load_phase9_inputs(
                Path(paths["phase5"]),
                Path(paths["phase6"]),
                Path(paths["phase7"]),
                Path(paths["phase8"]),
                project_root=project_root,
            )
        _validate_input_hashes(
            directory,
            experiment_manifest,
            inputs,
            hardened=hardened,
            errors=errors,
            warnings=warnings,
        )
        feature_sets = resolve_feature_sets(inputs.phase7_lineage, inputs.phase8_lineage)
        _validate_source_audit(inputs, project_root, errors)
        _validate_feature_sets(
            persisted_feature_sets,
            model_input_schema,
            model_manifest,
            feature_sets,
            hardened=hardened,
            errors=errors,
            warnings=warnings,
        )
        statuses = _validate_experiment_inventory(
            metrics_payload,
            model_manifest,
            directory,
            hardened=hardened,
            errors=errors,
            warnings=warnings,
        )
        settings = load_baseline_settings(project_root)
        try:
            from .contract import load_baseline_contract

            contract, _ = load_baseline_contract(project_root)
            _validate_configuration_policy(settings, contract, errors)
        except Exception as exc:
            errors.append(f"Phase 9 configuration policy could not be loaded: {exc}")
        targets = load_development_targets(
            inputs.mart_dir / "claim_snapshot.parquet", inputs.assignments
        )
        _validate_target_metadata(
            directory,
            audit,
            experiment_manifest,
            targets,
            hardened=hardened,
            errors=errors,
            warnings=warnings,
        )
        _validate_test_seal(directory, audit, experiment_manifest, errors)
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
        train_keys = set(train_rows[KEY].astype(int))
        validation_keys = set(validation_rows[KEY].astype(int))
        test_keys = set(
            inputs.assignments.loc[inputs.assignments["split"] == "TEST", KEY].astype(int)
        )
        canonical_prediction_hash = _validate_predictions(
            predictions,
            statuses,
            validation_keys,
            train_keys,
            test_keys,
            experiment_manifest,
            hardened=hardened,
            errors=errors,
            warnings=warnings,
        )
        _validate_reload_and_metrics(
            directory,
            statuses,
            model_manifest,
            feature_sets,
            settings,
            train_rows,
            validation_rows,
            y_train,
            y_validation,
            predictions,
            hardened=hardened,
            errors=errors,
            warnings=warnings,
        )
        _validate_ap_lift(statuses, hardened=hardened, errors=errors, warnings=warnings)
        report_path = _report_summary_path(directory, experiment_manifest, project_root)
        champion = _validate_champion(
            directory,
            metrics_payload,
            experiment_manifest,
            report_path,
            feature_sets,
            statuses,
            hardened=hardened,
            allow_report_pending=allow_report_pending,
            errors=errors,
            warnings=warnings,
        )
    except Exception as exc:
        errors.append(str(exc))
    unique_errors = list(dict.fromkeys(errors))
    unique_warnings = list(dict.fromkeys(warnings))
    if unique_errors:
        status = "BLOCKED"
        hardening_status = "BLOCKED"
    elif hardened:
        status = "PASS WITH WARNINGS" if unique_warnings else "PASS"
        hardening_status = "HARDENED_PASS"
    else:
        status = "PASS WITH WARNINGS" if unique_warnings else "PASS"
        hardening_status = "LEGACY_VALID"
    return {
        "status": status,
        "valid": not unique_errors,
        "hardening_status": hardening_status,
        "errors": unique_errors,
        "warnings": unique_warnings,
        "champion_experiment_id": champion,
        "canonical_prediction_sha256": canonical_prediction_hash,
        "model_reload_probability_tolerance": 1.0e-12,
        "metrics_recomputed": not unique_errors,
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
    }


def compare_phase9_runs(before_dir: Path, after_dir: Path) -> dict[str, Any]:
    """Compare locked populations, features, predictions, and metrics across runs."""

    before_manifest = _read_json(before_dir / "experiment_manifest.json")
    after_manifest = _read_json(after_dir / "experiment_manifest.json")
    before_metrics = _read_json(before_dir / "validation_metrics.json")
    after_metrics = _read_json(after_dir / "validation_metrics.json")
    before_features = _read_json(before_dir / "feature_sets.json")
    after_features = _read_json(after_dir / "feature_sets.json")
    before_predictions = pd.read_parquet(before_dir / "validation_predictions.parquet")
    after_predictions = pd.read_parquet(after_dir / "validation_predictions.parquet")

    errors: list[str] = []
    before_membership = before_manifest.get("frozen_membership", {})
    after_membership = after_manifest.get("frozen_membership", {})
    if before_membership.get("counts") != after_membership.get("counts"):
        errors.append("Frozen population counts changed.")
    before_targets = before_manifest.get("target_summary", {})
    after_targets = after_manifest.get("target_summary", {})
    target_hashes = {
        "before": {
            split: before_targets.get(split, {}).get("target_content_sha256")
            for split in ("train", "validation")
        },
        "after": {
            split: after_targets.get(split, {}).get("target_content_sha256")
            for split in ("train", "validation")
        },
    }
    if target_hashes["before"] != target_hashes["after"]:
        errors.append("TRAIN/VALIDATION target hashes changed.")
    feature_inventory = {
        "before": {
            key: {"count": value.get("feature_count"), "sha256": value.get("feature_set_sha256")}
            for key, value in before_features.items()
        },
        "after": {
            key: {"count": value.get("feature_count"), "sha256": value.get("feature_set_sha256")}
            for key, value in after_features.items()
        },
    }
    if feature_inventory["before"] != feature_inventory["after"]:
        errors.append("Feature-set counts or hashes changed.")
    prediction_hashes = {
        "before": prediction_content_sha256(before_predictions),
        "after": prediction_content_sha256(after_predictions),
    }
    if prediction_hashes["before"] != prediction_hashes["after"]:
        errors.append("Canonical validation prediction hash changed.")
    before_experiments = before_metrics.get("experiments", {})
    after_experiments = after_metrics.get("experiments", {})
    metric_deltas: dict[str, dict[str, float]] = {}
    for experiment_id in sorted(set(before_experiments) | set(after_experiments)):
        before_item = before_experiments.get(experiment_id, {})
        after_item = after_experiments.get(experiment_id, {})
        if before_item.get("status") != after_item.get("status"):
            errors.append(f"{experiment_id} experiment status changed.")
        before_metric = before_item.get("metrics", {})
        after_metric = after_item.get("metrics", {})
        metric_deltas[experiment_id] = {}
        for name in sorted(set(before_metric) | set(after_metric)):
            left = before_metric.get(name)
            right = after_metric.get(name)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                delta = abs(float(left) - float(right))
                metric_deltas[experiment_id][name] = delta
                if delta > 1.0e-12:
                    errors.append(f"{experiment_id} metric changed beyond tolerance: {name}")
            elif left != right:
                errors.append(f"{experiment_id} metric changed: {name}")
    if before_manifest.get("champion_experiment_id") != after_manifest.get(
        "champion_experiment_id"
    ):
        errors.append("Champion experiment changed.")
    model_hashes = {
        "before": {
            key: value.get("model_sha256")
            for key, value in _read_json(before_dir / "model_manifest.json")
            .get("models", {})
            .items()
        },
        "after": {
            key: value.get("model_sha256")
            for key, value in _read_json(after_dir / "model_manifest.json")
            .get("models", {})
            .items()
        },
    }
    return {
        "status": "PASS" if not errors else "BLOCKED",
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "before_run_id": before_manifest.get("run_id"),
        "after_run_id": after_manifest.get("run_id"),
        "population_counts": {
            "before": before_membership.get("counts"),
            "after": after_membership.get("counts"),
        },
        "target_hashes": target_hashes,
        "feature_sets": feature_inventory,
        "prediction_hashes": prediction_hashes,
        "metric_deltas": metric_deltas,
        "champion": {
            "before": before_manifest.get("champion_experiment_id"),
            "after": after_manifest.get("champion_experiment_id"),
        },
        "model_binary_hashes": model_hashes,
        "model_binary_hashes_required_to_match": False,
    }
