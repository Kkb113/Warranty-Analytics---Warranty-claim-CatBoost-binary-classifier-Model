"""Independent fail-closed validator for a completed Phase 12 bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..baseline_model.catboost_baseline import effective_parameters, load_model
from ..catboost_optimization.input import (
    load_train_targets_for_optimization,
    load_validation_targets_after_freeze,
)
from ..catboost_optimization.provenance import canonical_json_sha256, sha256_file
from ..paths import discover_repository_root
from .config import STRATEGY_IDS, TRACKS, ImbalanceThresholdError, load_imbalance_threshold_settings
from .contract import validate_imbalance_threshold_contract
from .input import load_locked_phase11_inputs
from .metrics import (
    aggregate_strategy_metrics,
    ranking_metrics,
    threshold_metrics,
)
from .selection import select_phase12_champion, select_strategy
from .strategies import StrategyDefinition, build_strategy_definitions, strategy_parameters
from .thresholds import build_threshold_curve, threshold_summary

KEY = "warranty_claim_key"
TARGET = "target__high_cost_claim_flag"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ImbalanceThresholdError(f"Expected JSON object: {path}")
    return payload


def _finite_equal(left: Any, right: Any, tolerance: float = 1.0e-10) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=tolerance))
    except (TypeError, ValueError):
        return bool(left == right)


def _compare_frame(
    actual: pd.DataFrame, expected: pd.DataFrame, columns: list[str], tolerance: float = 1.0e-10
) -> list[str]:
    errors: list[str] = []
    if list(actual.columns) != columns or list(expected.columns) != columns:
        return ["Phase 12 persisted table schema differs from the locked schema."]
    if len(actual) != len(expected):
        return [f"Phase 12 persisted table row count differs: {len(actual)} != {len(expected)}."]
    for column in columns:
        if pd.api.types.is_numeric_dtype(actual[column]) and pd.api.types.is_numeric_dtype(
            expected[column]
        ):
            left = pd.to_numeric(actual[column], errors="coerce").to_numpy(dtype="float64")
            right = pd.to_numeric(expected[column], errors="coerce").to_numpy(dtype="float64")
            if not np.allclose(left, right, atol=tolerance, rtol=0.0, equal_nan=False):
                errors.append(f"Phase 12 table column differs: {column}.")
        elif not actual[column].astype(str).equals(expected[column].astype(str)):
            errors.append(f"Phase 12 table column differs: {column}.")
    return errors


def _validate_model_parameters(
    model_path: Path,
    parent_parameters: dict[str, Any],
    strategy: StrategyDefinition,
) -> None:
    model = load_model(model_path)
    actual = effective_parameters(model)
    expected = strategy_parameters(
        {
            **parent_parameters,
            "thread_count": actual.get("thread_count", 1),
            "allow_writing_files": False,
            "verbose": False,
            "use_best_model": False,
        },
        strategy,
    )
    forbidden = {
        "thread_count",
        "allow_writing_files",
        "verbose",
        "use_best_model",
        "class_weights",
        "auto_class_weights",
        "scale_pos_weight",
    }
    for key, value in expected.items():
        if key in forbidden:
            continue
        if key not in actual:
            raise ImbalanceThresholdError(
                f"CatBoost parameter mismatch in {model_path.name}: {key}."
            )
        actual_value = actual[key]
        if isinstance(value, (int, float)) and isinstance(actual_value, (int, float)):
            if abs(float(actual_value) - float(value)) > 1.0e-6:
                raise ImbalanceThresholdError(
                    f"CatBoost parameter mismatch in {model_path.name}: {key}."
                )
        elif actual_value != value:
            raise ImbalanceThresholdError(
                f"CatBoost parameter mismatch in {model_path.name}: {key}."
            )
    actual_scale = actual.get("scale_pos_weight")
    actual_auto = actual.get("auto_class_weights")
    actual_classes = actual.get("class_weights")
    class_weight_value = None
    if isinstance(actual_classes, (list, tuple)) and len(actual_classes) == 2:
        try:
            class_weight_value = float(actual_classes[1]) / float(actual_classes[0])
        except (TypeError, ValueError, ZeroDivisionError):
            class_weight_value = None
    active = {
        key
        for key, value in {
            "scale_pos_weight": actual_scale,
            "auto_class_weights": actual_auto,
            "class_weights": class_weight_value,
        }.items()
        if value not in (None, 1, 1.0, "None")
    }
    if strategy.strategy_type == "none" and active:
        raise ImbalanceThresholdError(
            f"S0_NONE model contains weighting parameters: {sorted(active)}"
        )
    if strategy.strategy_type == "scale_pos_weight":
        if not isinstance(strategy.parameter, (int, float)):
            raise ImbalanceThresholdError("Scale-positive strategy has no numeric value.")
        expected_value = float(strategy.parameter)
        if not (
            (
                isinstance(actual_scale, (int, float))
                and abs(float(actual_scale) - expected_value) <= 1.0e-6
            )
            or (
                class_weight_value is not None
                and abs(class_weight_value - expected_value) <= 1.0e-6
            )
        ):
            raise ImbalanceThresholdError(
                "Scale-positive CatBoost parameter differs from strategy."
            )
        if actual_auto not in (None, "None"):
            raise ImbalanceThresholdError(
                "Scale-positive model also contains auto class weighting."
            )
    if strategy.strategy_type == "auto_class_weights":
        expected_policy = strategy.parameter
        auto_expected_value = strategy.resolved_parameter
        policy_ok = actual_auto in (expected_policy, None, "None")
        value_ok = auto_expected_value is None or (
            isinstance(auto_expected_value, (int, float))
            and class_weight_value is not None
            and abs(class_weight_value - float(auto_expected_value)) <= 1.0e-6
        )
        if not (policy_ok and value_ok):
            raise ImbalanceThresholdError("Auto CatBoost parameter differs from strategy.")


def validate_existing_phase12(
    phase12_dir: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    directory = phase12_dir.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        root = discover_repository_root(project_root or directory)
        required = (
            "phase12_manifest.json",
            "phase11_parent_resolution.json",
            "strategy_definitions.json",
            "strategy_fold_metrics.parquet",
            "strategy_summary.parquet",
            "strategy_oof_predictions.parquet",
            "threshold_curve.parquet",
            "threshold_summary.json",
            "phase12_freeze.json",
            "validation_predictions.parquet",
            "validation_metrics.json",
            "effective_model_manifest.json",
            "threshold_policy.json",
            "model_manifest.json",
            "target_access_audit.json",
            "compute_manifest.json",
            "validation.json",
        )
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise ImbalanceThresholdError("Phase 12 artifacts missing: " + ", ".join(missing))
        manifest = _read_json(directory / "phase12_manifest.json")
        if manifest.get("phase") != 12:
            raise ImbalanceThresholdError("Phase 12 manifest phase is invalid.")
        contract = validate_imbalance_threshold_contract(root)
        if not contract.get("valid"):
            raise ImbalanceThresholdError(
                "Phase 12 contract is invalid: " + "; ".join(contract.get("errors", []))
            )
        if manifest.get("contract_sha256") != contract.get("contract_checksum"):
            raise ImbalanceThresholdError(
                "Phase 12 contract checksum differs from current repository."
            )
        phase11_run_id = str(manifest.get("phase11_run_id"))
        phase11_dir = root / "artifacts" / "feature_selection" / phase11_run_id
        inputs = load_locked_phase11_inputs(phase11_dir, project_root=root)
        if manifest.get("phase11_manifest_sha256") != inputs.phase11_manifest_sha256:
            raise ImbalanceThresholdError("Phase 11 manifest hash binding changed.")
        if manifest.get("phase11_validation_sha256") != inputs.phase11_validation_sha256:
            raise ImbalanceThresholdError("Phase 11 validation hash binding changed.")
        if manifest.get("phase11_model_manifest_sha256") != inputs.phase11_model_manifest_sha256:
            raise ImbalanceThresholdError("Phase 11 model manifest hash binding changed.")
        if manifest.get("phase10_inner_fold_sha256") != inputs.fold_plan.content_sha256:
            raise ImbalanceThresholdError("Phase 10 fold hash binding changed.")
        resolution = _read_json(directory / "phase11_parent_resolution.json")
        if resolution.get("phase11_manifest_sha256") != inputs.phase11_manifest_sha256:
            raise ImbalanceThresholdError("Phase 12 parent-resolution binding changed.")
        audit = _read_json(directory / "target_access_audit.json")
        for key, expected in {
            "test_target_rows_loaded": 0,
            "test_predictions_created": 0,
            "test_metrics_computed": False,
            "test_target_access_allowed": False,
            "first_allowed_test_target_phase": 15,
        }.items():
            if audit.get(key) != expected:
                raise ImbalanceThresholdError(f"Phase 12 TEST seal changed: {key}.")
        freeze = _read_json(directory / "phase12_freeze.json")
        freeze_copy = dict(freeze)
        freeze_hash = freeze_copy.pop("phase12_freeze_sha256", None)
        if (
            canonical_json_sha256(freeze_copy) != freeze_hash
            or manifest.get("phase12_freeze_sha256") != freeze_hash
        ):
            raise ImbalanceThresholdError("Phase 12 freeze hash is invalid.")
        if (
            freeze.get("outer_validation_accessed") is not False
            or freeze.get("test_target_accessed") is not False
        ):
            raise ImbalanceThresholdError("Phase 12 freeze is not a pre-validation freeze.")
        settings = load_imbalance_threshold_settings(root)
        train_targets, _ = load_train_targets_for_optimization(inputs.phase10_inputs)
        positive = int(train_targets[TARGET].sum())
        negative = int((train_targets[TARGET] == 0).sum())
        strategies = build_strategy_definitions(positive, negative)
        definitions = _read_json(directory / "strategy_definitions.json")
        if [item.get("strategy_id") for item in definitions.get("strategies", [])] != list(
            STRATEGY_IDS
        ):
            raise ImbalanceThresholdError(
                "Phase 12 strategy inventory is not exactly the locked eight strategies."
            )
        if any(
            item.get("parameter_sha256") != strategy.parameter_sha256
            for item, strategy in zip(definitions["strategies"], strategies, strict=True)
        ):
            raise ImbalanceThresholdError("Phase 12 strategy parameter hash changed.")
        fold_keys_by_id = {
            int(fold.fold_id): set(fold.validation_keys) for fold in inputs.fold_plan.folds
        }
        fold_frame = pd.read_parquet(directory / "strategy_fold_metrics.parquet")
        expected_fold_columns = [
            "track",
            "strategy_id",
            "fold_id",
            "average_precision",
            "roc_auc",
            "log_loss",
            "brier_score",
            "train_positive_count",
            "train_negative_count",
            "validation_positive_count",
            "validation_negative_count",
            "weighting_parameters",
            "prediction_minimum",
            "prediction_maximum",
            "prediction_mean",
            "prediction_median",
            "training_seconds",
        ]
        if list(fold_frame.columns) != expected_fold_columns:
            raise ImbalanceThresholdError("Phase 12 fold metric schema changed.")
        if (
            len(fold_frame) != 48
            or fold_frame.duplicated(["track", "strategy_id", "fold_id"]).any()
        ):
            raise ImbalanceThresholdError(
                "Phase 12 does not contain exactly 48 unique strategy-fold rows."
            )
        if set(fold_frame["track"]) != set(TRACKS) or set(fold_frame["strategy_id"]) != set(
            STRATEGY_IDS
        ):
            raise ImbalanceThresholdError("Phase 12 fold metric track/strategy inventory changed.")
        if any(
            len(group) != 3 or set(group["fold_id"]) != {1, 2, 3}
            for _, group in fold_frame.groupby(["track", "strategy_id"])
        ):
            raise ImbalanceThresholdError("Phase 12 strategy folds are incomplete.")
        oof = pd.read_parquet(directory / "strategy_oof_predictions.parquet")
        expected_oof_columns = [KEY, "track", "strategy_id", "fold_id", "high_cost_probability"]
        if list(oof.columns) != expected_oof_columns:
            raise ImbalanceThresholdError("Phase 12 OOF schema changed.")
        if oof.duplicated([KEY, "track", "strategy_id"]).any():
            raise ImbalanceThresholdError("Phase 12 OOF predictions contain duplicates.")
        target_by_key = train_targets.set_index(KEY)[TARGET]
        if not set(oof[KEY].astype(int)).issubset(set(target_by_key.index.astype(int))):
            raise ImbalanceThresholdError("Phase 12 OOF includes non-TRAIN claims.")
        for (track, strategy_id), group in oof.groupby(["track", "strategy_id"]):
            if len(group) != sum(len(fold.validation_keys) for fold in inputs.fold_plan.folds):
                raise ImbalanceThresholdError(
                    f"Phase 12 OOF row count changed for {track}/{strategy_id}."
                )
            for fold_id, fold_group in group.groupby("fold_id"):
                if set(fold_group[KEY].astype(int)) != fold_keys_by_id[int(fold_id)]:
                    raise ImbalanceThresholdError(
                        f"Phase 12 OOF fold membership changed for {track}/{strategy_id}/{fold_id}."
                    )
        summary = pd.read_parquet(directory / "strategy_summary.parquet")
        expected_summary_rows: list[dict[str, Any]] = []
        for (_track, _strategy_id), group in fold_frame.groupby(
            ["track", "strategy_id"], sort=True
        ):
            rows = group.to_dict("records")
            expected_summary_rows.append(aggregate_strategy_metrics(rows))
        expected_summary = (
            pd.DataFrame(expected_summary_rows)
            .sort_values(["track", "strategy_id"], kind="mergesort")
            .reset_index(drop=True)
        )
        summary = summary.sort_values(["track", "strategy_id"], kind="mergesort").reset_index(
            drop=True
        )
        summary_columns = list(expected_summary.columns)
        errors.extend(_compare_frame(summary, expected_summary, summary_columns, tolerance=1.0e-10))
        curves = pd.read_parquet(directory / "threshold_curve.parquet")
        persisted_threshold_summaries = _read_json(directory / "threshold_summary.json")
        expected_curves: list[pd.DataFrame] = []
        expected_thresholds: dict[str, dict[str, dict[str, Any]]] = {}
        for track in TRACKS:
            expected_thresholds[track] = {}
            for strategy_id in STRATEGY_IDS:
                group = oof.loc[
                    (oof["track"] == track) & (oof["strategy_id"] == strategy_id)
                ].sort_values(KEY, kind="mergesort")
                y = target_by_key.loc[group[KEY].tolist()].to_numpy(dtype="int8")
                expected_curve = build_threshold_curve(
                    y,
                    group["high_cost_probability"].to_numpy(dtype="float64"),
                    track=track,
                    strategy_id=strategy_id,
                )
                expected_curves.append(expected_curve)
                expected_thresholds[track][strategy_id] = threshold_summary(expected_curve)
        if persisted_threshold_summaries != expected_thresholds:
            errors.append(
                "Phase 12 persisted threshold summary differs from the recomputed summary."
            )
        expected_curve_frame = (
            pd.concat(expected_curves, ignore_index=True)
            .sort_values(["track", "strategy_id", "threshold"], kind="mergesort")
            .reset_index(drop=True)
        )
        curves = curves.sort_values(
            ["track", "strategy_id", "threshold"], kind="mergesort"
        ).reset_index(drop=True)
        errors.extend(
            _compare_frame(
                curves, expected_curve_frame, list(expected_curve_frame.columns), tolerance=1.0e-12
            )
        )
        selected: dict[str, dict[str, Any]] = {}
        for track in TRACKS:
            selected[track] = select_strategy(
                summary.loc[summary["track"] == track],
                expected_thresholds[track],
                max_ap_tolerance=settings.max_ap_tolerance,
                max_min_ap_drop=settings.max_min_ap_drop,
                max_roc_auc_drop=settings.max_roc_auc_drop,
                prefer_none_mcc_tolerance=settings.prefer_none_mcc_tolerance,
            )
            if (
                manifest.get("selected_strategies", {}).get(track)
                != selected[track]["selected_strategy_id"]
            ):
                errors.append(f"Phase 12 selected strategy changed for {track}.")
        validation_targets, _ = load_validation_targets_after_freeze(
            inputs.phase10_inputs, study_frozen=True
        )
        validation_by_key = validation_targets.set_index(KEY)[TARGET]
        validation_predictions = pd.read_parquet(directory / "validation_predictions.parquet")
        if list(validation_predictions.columns) != [
            KEY,
            "track",
            "candidate_id",
            "high_cost_probability",
        ]:
            raise ImbalanceThresholdError("Phase 12 validation prediction schema changed.")
        model_manifest = _read_json(directory / "model_manifest.json")
        model_entries = model_manifest.get("models", [])
        if not isinstance(model_entries, list):
            raise ImbalanceThresholdError("Phase 12 model manifest schema changed.")
        # Verify actual serialized parameters and reload predictions for every persisted model.
        for entry in model_entries:
            track = str(entry["track"])
            strategy_id = str(entry.get("imbalance_strategy", {}).get("strategy_id", "S0_NONE"))
            strategy = next(item for item in strategies if item.strategy_id == strategy_id)
            model_path = directory / str(entry["model_file"])
            if not model_path.is_file() or sha256_file(model_path) != entry.get("model_sha256"):
                raise ImbalanceThresholdError(
                    f"Phase 12 model hash changed: {entry.get('candidate_id')}."
                )
            _validate_model_parameters(
                model_path, inputs.parents[track].statistical_parameters, strategy
            )
            model = load_model(model_path)
            validation_frame = inputs.development.loc[
                inputs.development["split"] == "VALIDATION"
            ].sort_values(KEY, kind="mergesort")
            from dataclasses import replace

            from ..baseline_model.adapters import adapt_matrix

            settings_model = __import__(
                "warranty_analytics_model.baseline_model.config",
                fromlist=["load_baseline_settings"],
            ).load_baseline_settings(root)
            adapted = adapt_matrix(
                validation_frame.drop(columns=[KEY]),
                inputs.parents[track].feature_set,
                replace(settings_model, catboost_parameters={}),
            )
            probabilities = np.asarray(
                model.predict_proba(
                    __import__(
                        "warranty_analytics_model.baseline_model.catboost_baseline",
                        fromlist=["build_pool"],
                    ).build_pool(adapted, inputs.parents[track].feature_set)
                )[:, 1],
                dtype="float64",
            )
            persisted = validation_predictions.loc[
                (validation_predictions["track"] == track)
                & (validation_predictions["candidate_id"] == entry["candidate_id"])
            ].sort_values(KEY, kind="mergesort")
            if len(persisted) == len(probabilities) and not np.allclose(
                persisted["high_cost_probability"].to_numpy(), probabilities, atol=1.0e-10, rtol=0.0
            ):
                errors.append(f"Phase 12 reload probability mismatch: {entry['candidate_id']}.")
        validation_metrics = _read_json(directory / "validation_metrics.json")
        effective = _read_json(directory / "effective_model_manifest.json").get("models", [])
        champion_candidates: list[dict[str, Any]] = []
        for entry in effective:
            track = str(entry["track"])
            candidate_id = str(entry["candidate_id"])
            row = validation_predictions.loc[
                (validation_predictions["track"] == track)
                & (validation_predictions["candidate_id"] == candidate_id)
            ].sort_values(KEY, kind="mergesort")
            threshold = float(entry["technical_threshold"])
            y = validation_by_key.loc[row[KEY].tolist()].to_numpy(dtype="int8")
            metrics = ranking_metrics(y, row["high_cost_probability"].to_numpy(dtype="float64"))
            metrics.update(
                threshold_metrics(
                    y, row["high_cost_probability"].to_numpy(dtype="float64"), threshold
                )
            )
            persisted_metrics = entry.get("validation_metrics", {})
            for key in (
                "average_precision",
                "roc_auc",
                "log_loss",
                "brier_score",
                "mcc",
                "f1",
                "f2",
            ):
                if key in persisted_metrics and not _finite_equal(
                    metrics.get(key), persisted_metrics[key]
                ):
                    errors.append(f"Phase 12 validation metric differs for {candidate_id}: {key}.")
            champion_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "feature_count": entry.get("feature_count", 0),
                    "complexity_order": 0,
                    "validation_metrics": metrics,
                }
            )
        if champion_candidates:
            expected_champion = select_phase12_champion(champion_candidates)
            if validation_metrics.get("development_champion") != expected_champion:
                errors.append("Phase 12 development champion changed.")
        status = "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS")
        return {
            "status": status,
            "valid": not errors,
            "hardening_status": "HARDENED_PASS" if not errors else "BLOCKED",
            "errors": list(dict.fromkeys(errors)),
            "warnings": list(dict.fromkeys(warnings)),
            "run_id": manifest.get("run_id"),
            "phase12_development_champion": validation_metrics.get("development_champion"),
            "test_seal": {
                key: audit.get(key)
                for key in (
                    "test_target_rows_loaded",
                    "test_predictions_created",
                    "test_metrics_computed",
                    "test_target_access_allowed",
                    "first_allowed_test_target_phase",
                )
            },
        }
    except Exception as exc:
        errors.append(str(exc))
        return {
            "status": "BLOCKED",
            "valid": False,
            "hardening_status": "BLOCKED",
            "errors": list(dict.fromkeys(errors)),
            "warnings": list(dict.fromkeys(warnings)),
        }


__all__ = ["validate_existing_phase12"]
