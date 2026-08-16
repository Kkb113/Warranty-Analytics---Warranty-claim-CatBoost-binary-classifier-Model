"""Independent Phase 13 artifact validation.

The validator reconstructs the deterministic calibration, ensemble, and threshold
steps from the locked Phase 12 inputs.  It intentionally does not accept the
runner's status field as evidence of correctness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..baseline_model.adapters import adapt_matrix
from ..baseline_model.catboost_baseline import load_model, predict_probabilities
from ..baseline_model.config import load_baseline_settings
from ..catboost_optimization.config import TRACK_TO_EXPERIMENT
from ..catboost_optimization.input import load_validation_targets_after_freeze
from ..catboost_optimization.provenance import canonical_json_sha256, sha256_file
from .calibration_folds import calibration_fold_assignments, calibration_fold_manifest
from .calibration_metrics import probability_metrics
from .calibrators import apply_calibrator, calibrator_sha, fit_calibrator
from .config import CALIBRATION_METHODS, TRACKS, load_calibration_ensemble_settings
from .ensemble import evaluate_ensemble_weights
from .input import KEY, load_phase12_lock
from .selection import select_calibration_method, select_ensemble
from .thresholds import build_threshold_curve, select_mcc_threshold


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _close(left: Any, right: Any, tolerance: float = 1.0e-9) -> bool:
    try:
        if not np.isfinite(float(left)) or not np.isfinite(float(right)):
            return bool(left == right)
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return bool(left == right)


def _calibration_input_sha(frame: pd.DataFrame) -> str:
    columns = [KEY, "fold_id", "high_cost_probability", "target"]
    records = []
    for row in (
        frame.loc[:, columns].sort_values(["fold_id", KEY], kind="mergesort").to_dict("records")
    ):
        records.append(
            [
                int(row[KEY]),
                int(row["fold_id"]),
                float(row["high_cost_probability"]),
                int(row["target"]),
            ]
        )
    return canonical_json_sha256({"columns": columns, "rows": records})


def _compare_frame(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    columns: list[str],
    *,
    tolerance: float = 1.0e-9,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if list(expected.columns) != columns or list(actual.columns) != columns:
        return [f"{label} schema changed."]
    if len(expected) != len(actual):
        return [f"{label} row count changed."]
    for index, (left, right) in enumerate(
        zip(expected.to_dict("records"), actual.to_dict("records"), strict=True)
    ):
        for column in columns:
            if isinstance(left[column], (float, int)) and not isinstance(left[column], bool):
                differs = not _close(left[column], right[column], tolerance)
            else:
                differs = left[column] != right[column]
            if differs:
                errors.append(f"{label} differs at row {index}, column {column}.")
                return errors
    return errors


def _validate_artifact_hashes(directory: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    declared = manifest.get("artifact_file_sha256", {})
    if not isinstance(declared, dict):
        return ["Phase 13 artifact_file_sha256 is missing."]
    for name, digest in declared.items():
        path = directory / str(name)
        if not path.is_file():
            errors.append(f"Phase 13 declared artifact is missing: {name}.")
        elif sha256_file(path) != str(digest):
            errors.append(f"Phase 13 artifact hash differs: {name}.")
    return errors


def _validate_test_seal(payload: dict[str, Any], label: str) -> list[str]:
    expected = {
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
    }
    return [
        f"{label} TEST seal changed: {key}."
        for key, value in expected.items()
        if payload.get(key) != value
    ]


def _validate_calibration(
    directory: Path,
    lock: Any,
    settings: Any,
    errors: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    source = lock.source_oof
    expected_assignment = calibration_fold_assignments(
        source.loc[
            :, [KEY, "track", "strategy_id", "fold_id", "high_cost_probability", "claim_date"]
        ]
    )
    persisted_assignment = pd.read_parquet(directory / "calibration_fold_assignments.parquet")
    if not persisted_assignment.equals(expected_assignment):
        errors.append("Phase 13 calibration fold assignments cannot be reproduced.")
    fold_manifest, fold_sha = calibration_fold_manifest(expected_assignment)
    persisted_fold_manifest = _read_json(directory / "calibration_fold_manifest.json")
    if persisted_fold_manifest.get("calibration_fold_content_sha256") != fold_sha:
        errors.append("Phase 13 calibration fold content hash differs.")
    if persisted_fold_manifest.get("folds") != fold_manifest.get("folds"):
        errors.append("Phase 13 calibration chronology manifest differs.")
    crossfit = pd.read_parquet(directory / "calibration_crossfit_predictions.parquet")
    crossfit_columns = [
        KEY,
        "source_fold_id",
        "calibration_fold_id",
        "track",
        "calibration_method",
        "raw_probability",
        "calibrated_probability",
    ]
    if list(crossfit.columns) != crossfit_columns:
        errors.append("Phase 13 calibration cross-fit schema changed.")
        return {}, {}
    if set(crossfit["source_fold_id"].astype(int)) - {2, 3}:
        errors.append("Phase 13 calibration predictions include a non-evaluation source fold.")
    if crossfit.duplicated([KEY, "track", "calibration_method", "calibration_fold_id"]).any():
        errors.append("Phase 13 calibration predictions contain duplicate rows.")
    selections = _read_json(directory / "calibration_selection.json").get("tracks", {})
    summary = pd.read_parquet(directory / "calibration_summary.parquet")
    final_payloads: dict[str, dict[str, Any]] = {}
    selected_frames: dict[str, pd.DataFrame] = {}
    for track in TRACKS:
        track_source = source.loc[source["track"] == track].copy()
        summaries: list[dict[str, Any]] = []
        for method in CALIBRATION_METHODS:
            predictions: list[pd.DataFrame] = []
            eligible = True
            reason = "ELIGIBLE"
            for fold_id, train_folds, validation_fold in (("C1", (1,), 2), ("C2", (1, 2), 3)):
                train = track_source.loc[track_source["fold_id"].isin(train_folds)]
                validation = track_source.loc[track_source["fold_id"] == validation_fold]
                payload = fit_calibrator(
                    method,
                    train["high_cost_probability"],
                    train["target"],
                    epsilon=settings.sigmoid_epsilon,
                    isotonic_y_min=settings.isotonic_y_min,
                    isotonic_y_max=settings.isotonic_y_max,
                    isotonic_out_of_bounds=settings.isotonic_out_of_bounds,
                    isotonic_minimum_positive=settings.isotonic_minimum_training_positives,
                    isotonic_minimum_negative=settings.isotonic_minimum_training_negatives,
                    isotonic_minimum_unique=settings.isotonic_minimum_unique_probabilities,
                    input_sha=_calibration_input_sha(train),
                )
                if payload.get("calibrator_sha") != _read_json(
                    directory / "calibration_candidates.json"
                )["tracks"][track][method]["folds"][fold_id]["calibrator"].get("calibrator_sha"):
                    errors.append(f"Phase 13 {track}/{method}/{fold_id} calibrator changed.")
                eligible = eligible and bool(payload.get("eligible", True))
                reason = (
                    str(payload.get("eligibility_reason", reason))
                    if not payload.get("eligible", True)
                    else reason
                )
                if payload.get("eligible", True):
                    calibrated = apply_calibrator(payload, validation["high_cost_probability"])
                    predictions.append(
                        pd.DataFrame(
                            {
                                KEY: validation[KEY].astype(int).to_numpy(),
                                "source_fold_id": validation["fold_id"].astype(int).to_numpy(),
                                "calibration_fold_id": fold_id,
                                "track": track,
                                "calibration_method": method,
                                "raw_probability": validation["high_cost_probability"]
                                .astype(float)
                                .to_numpy(),
                                "calibrated_probability": calibrated,
                                "target": validation["target"].astype(int).to_numpy(),
                            }
                        )
                    )
            if predictions:
                combined = pd.concat(predictions, ignore_index=True)
                pooled = probability_metrics(
                    combined["target"],
                    combined["calibrated_probability"],
                    bins=settings.reliability_bins,
                    keys=combined[KEY],
                )
                aps = [
                    probability_metrics(
                        part["target"],
                        part["calibrated_probability"],
                        bins=settings.reliability_bins,
                        keys=part[KEY],
                    )["average_precision"]
                    for _, part in combined.groupby("calibration_fold_id", sort=True)
                ]
                summaries.append(
                    {
                        "track": track,
                        "calibration_method": method,
                        "pooled_average_precision": pooled["average_precision"],
                        "mean_fold_average_precision": float(np.mean(aps)),
                        "min_fold_average_precision": float(min(aps)),
                        "pooled_roc_auc": pooled["roc_auc"],
                        "pooled_log_loss": pooled["log_loss"],
                        "pooled_brier_score": pooled["brier_score"],
                        "pooled_ece": pooled["ece_10"],
                        "pooled_mce": pooled["mce_10"],
                        "row_count": pooled["row_count"],
                        "positive_count": pooled["positive_count"],
                        "negative_count": pooled["negative_count"],
                        "eligible": eligible,
                        "eligibility_reason": reason,
                    }
                )
            selected_method = str(selections.get(track, {}).get("selected_calibration_method", ""))
            if method == selected_method and predictions:
                selected_frames[track] = pd.concat(predictions, ignore_index=True)
                final_payload = fit_calibrator(
                    method,
                    track_source["high_cost_probability"],
                    track_source["target"],
                    epsilon=settings.sigmoid_epsilon,
                    isotonic_y_min=settings.isotonic_y_min,
                    isotonic_y_max=settings.isotonic_y_max,
                    isotonic_out_of_bounds=settings.isotonic_out_of_bounds,
                    isotonic_minimum_positive=settings.isotonic_minimum_training_positives,
                    isotonic_minimum_negative=settings.isotonic_minimum_training_negatives,
                    isotonic_minimum_unique=settings.isotonic_minimum_unique_probabilities,
                    input_sha=_calibration_input_sha(track_source),
                )
                final_payloads[track] = final_payload
        expected_selection = select_calibration_method(pd.DataFrame(summaries))
        if expected_selection.get("selected_calibration_method") != selections.get(track, {}).get(
            "selected_calibration_method"
        ):
            errors.append(f"Phase 13 calibration selection differs for {track}.")
        persisted = (
            summary.loc[summary["track"] == track]
            .sort_values("calibration_method")
            .reset_index(drop=True)
        )
        expected = pd.DataFrame(summaries).sort_values("calibration_method").reset_index(drop=True)
        for column in [
            "pooled_average_precision",
            "mean_fold_average_precision",
            "min_fold_average_precision",
            "pooled_roc_auc",
            "pooled_log_loss",
            "pooled_brier_score",
            "pooled_ece",
            "pooled_mce",
        ]:
            if column not in persisted or any(
                not _close(a, b, 1.0e-8)
                for a, b in zip(persisted[column], expected[column], strict=True)
            ):
                errors.append(f"Phase 13 calibration summary differs for {track}/{column}.")
    return selected_frames, final_payloads


def validate_existing_phase13(
    phase13_dir: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    directory = phase13_dir.expanduser().resolve()
    root = (project_root or Path.cwd()).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    required = (
        "phase13_manifest.json",
        "phase12_parent_resolution.json",
        "calibration_fold_manifest.json",
        "calibration_fold_assignments.parquet",
        "calibration_candidates.json",
        "calibration_fold_metrics.parquet",
        "calibration_crossfit_predictions.parquet",
        "calibration_summary.parquet",
        "calibration_selection.json",
        "selected_calibrated_oof_predictions.parquet",
        "ensemble_candidates.json",
        "ensemble_fold_metrics.parquet",
        "ensemble_summary.parquet",
        "ensemble_crossfit_predictions.parquet",
        "ensemble_selection.json",
        "threshold_curve.parquet",
        "threshold_policy.json",
        "phase13_freeze.json",
        "validation_predictions.parquet",
        "validation_metrics.json",
        "effective_model_manifest.json",
        "target_access_audit.json",
        "compute_manifest.json",
        "calibrators/t1.json",
        "calibrators/t3.json",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        return {
            "phase": 13,
            "valid": False,
            "hardening_status": "BLOCKED",
            "errors": ["Phase 13 artifacts missing: " + ", ".join(missing)],
            "warnings": [],
        }
    try:
        manifest = _read_json(directory / "phase13_manifest.json")
        if manifest.get("phase") != 13 or not manifest.get("run_id"):
            errors.append("Phase 13 manifest is not a valid run.")
        errors.extend(_validate_artifact_hashes(directory, manifest))
        errors.extend(_validate_test_seal(manifest, "Phase 13 manifest"))
        audit = _read_json(directory / "target_access_audit.json")
        errors.extend(_validate_test_seal(audit, "Phase 13 target audit"))
        if audit.get("validation_target_rows_loaded_before_phase13_freeze") != 0:
            errors.append("Phase 13 accessed VALIDATION targets before freeze.")
        freeze = _read_json(directory / "phase13_freeze.json")
        freeze_copy = {
            key: value for key, value in freeze.items() if key != "phase13_freeze_sha256"
        }
        if canonical_json_sha256(freeze_copy) != freeze.get("phase13_freeze_sha256"):
            errors.append("Phase 13 freeze hash differs.")
        if freeze.get("outer_validation_accessed") is not False:
            errors.append("Phase 13 freeze outer-validation flag was changed.")
        errors.extend(_validate_test_seal(freeze, "Phase 13 freeze"))
        lock = load_phase12_lock(
            Path(str(manifest.get("phase12_dir", "")))
            if manifest.get("phase12_dir")
            else directory.parent.parent
            / "imbalance_threshold"
            / str(manifest.get("phase12_run_id", "")),
            project_root=root,
        )
        settings = load_calibration_ensemble_settings(root)
        selected_frames, final_payloads = _validate_calibration(directory, lock, settings, errors)
        if set(selected_frames) != set(TRACKS):
            errors.append("Phase 13 selected calibrated OOF tracks are incomplete.")
        selected_persisted = pd.read_parquet(
            directory / "selected_calibrated_oof_predictions.parquet"
        )
        selected_columns = [
            KEY,
            "source_fold_id",
            "calibration_fold_id",
            "track",
            "raw_probability",
            "calibrated_probability",
        ]
        if list(selected_persisted.columns) != selected_columns:
            errors.append("Phase 13 selected calibrated OOF schema changed.")
        for track, frame in selected_frames.items():
            expected = (
                frame.loc[:, selected_columns]
                .sort_values(["track", "calibration_fold_id", KEY], kind="mergesort")
                .reset_index(drop=True)
            )
            actual = (
                selected_persisted.loc[selected_persisted["track"] == track]
                .sort_values(["track", "calibration_fold_id", KEY], kind="mergesort")
                .reset_index(drop=True)
            )
            errors.extend(
                _compare_frame(expected, actual, selected_columns, label=f"selected OOF {track}")
            )

        source_targets = (
            lock.source_oof.loc[:, [KEY, "target"]].drop_duplicates(KEY).set_index(KEY)["target"]
        )
        t1 = selected_persisted.loc[selected_persisted["track"] == "T1"].copy()
        t3 = selected_persisted.loc[selected_persisted["track"] == "T3"].copy()
        for frame in (t1, t3):
            frame["target"] = frame[KEY].map(source_targets).astype("int8")
        from .ensemble import align_selected_tracks

        aligned = align_selected_tracks(t1, t3)
        expected_predictions, expected_summary = evaluate_ensemble_weights(
            aligned, weights=settings.ensemble_weights, bins=settings.reliability_bins
        )
        expected_summary = expected_summary.rename(
            columns={
                "average_precision": "pooled_average_precision",
                "roc_auc": "pooled_roc_auc",
                "log_loss": "pooled_log_loss",
                "brier_score": "pooled_brier_score",
                "ece_10": "pooled_ece",
                "mce_10": "pooled_mce",
            }
        )
        expected_summary["candidate_id"] = expected_summary["t1_weight"].map(
            lambda weight: f"P13_ENSEMBLE_W{int(round(float(weight) * 10)):02d}"
        )
        persisted_summary = (
            pd.read_parquet(directory / "ensemble_summary.parquet")
            .sort_values("t1_weight")
            .reset_index(drop=True)
        )
        expected_summary = expected_summary.sort_values("t1_weight").reset_index(drop=True)
        for column in [
            "pooled_average_precision",
            "pooled_roc_auc",
            "pooled_log_loss",
            "pooled_brier_score",
            "pooled_ece",
            "pooled_mce",
            "min_fold_average_precision",
        ]:
            if column not in persisted_summary or any(
                not _close(a, b, 1.0e-8)
                for a, b in zip(persisted_summary[column], expected_summary[column], strict=True)
            ):
                errors.append(f"Phase 13 ensemble summary differs: {column}.")
        selection = _read_json(directory / "ensemble_selection.json")
        expected_selection = select_ensemble(expected_summary)
        if selection.get("selected_policy") != expected_selection.get(
            "selected_policy"
        ) or not _close(
            selection.get("selected_weight"), expected_selection.get("selected_weight"), 1.0e-12
        ):
            errors.append("Phase 13 ensemble selection differs.")
        for track, payload in final_payloads.items():
            persisted = _read_json(directory / "calibrators" / f"{track.lower()}.json")
            if persisted.get("calibrator_sha") != payload.get("calibrator_sha") or calibrator_sha(
                persisted
            ) != persisted.get("calibrator_sha"):
                errors.append(f"Phase 13 final calibrator differs for {track}.")

        policy = _read_json(directory / "threshold_policy.json")
        curve = pd.read_parquet(directory / "threshold_curve.parquet")
        for track, frame in selected_frames.items():
            y = frame[KEY].map(source_targets).astype("int8")
            candidate_id = f"P13_{track}_CALIBRATED"
            expected_curve = build_threshold_curve(
                y,
                frame["calibrated_probability"],
                candidate_id=candidate_id,
                score_space="CALIBRATED_PROBABILITY",
                start=settings.threshold_start,
                stop=settings.threshold_stop,
                step=settings.threshold_step,
            )
            actual_curve = curve.loc[curve["candidate_id"] == candidate_id].reset_index(drop=True)
            errors.extend(
                _compare_frame(
                    expected_curve,
                    actual_curve,
                    list(expected_curve.columns),
                    tolerance=1.0e-12,
                    label=f"threshold curve {track}",
                )
            )
            selected = select_mcc_threshold(expected_curve, settings.threshold_tie_tolerance)
            if policy.get("candidates", {}).get(track, {}).get("threshold") != selected.get(
                "threshold"
            ):
                errors.append(f"Phase 13 threshold selection differs for {track}.")

        # Reproduce outer validation probabilities after the freeze gate.  This also
        # verifies that persisted validation rows were not fabricated or reordered.
        phase10 = lock.phase12_inputs.phase10_inputs
        validation_targets, _ = load_validation_targets_after_freeze(phase10, study_frozen=True)
        validation_frame = phase10.development.loc[
            phase10.development["split"] == "VALIDATION"
        ].copy()
        baseline_settings = load_baseline_settings(root)
        persisted_validation = pd.read_parquet(directory / "validation_predictions.parquet")
        for track in TRACKS:
            experiment = TRACK_TO_EXPERIMENT[track]
            feature_set = phase10.feature_sets[experiment]
            raw = predict_probabilities(
                load_model(lock.phase12_dir / str(lock.effective_models[track]["model_file"])),
                adapt_matrix(validation_frame, feature_set, baseline_settings),
                feature_set,
            )
            payload = _read_json(directory / "calibrators" / f"{track.lower()}.json")
            calibrated = apply_calibrator(payload, raw)
            actual = (
                persisted_validation.loc[persisted_validation["track"] == track]
                .sort_values(KEY)
                .reset_index(drop=True)
            )
            expected = (
                pd.DataFrame(
                    {
                        KEY: validation_frame[KEY].astype(int).to_numpy(),
                        "track": track,
                        "candidate_id": actual["candidate_id"].to_numpy()
                        if len(actual) == len(validation_frame)
                        else "",
                        "raw_probability": raw,
                        "calibrated_probability": calibrated,
                        "effective_probability": actual["effective_probability"].to_numpy()
                        if len(actual) == len(validation_frame)
                        else calibrated,
                    }
                )
                .sort_values(KEY)
                .reset_index(drop=True)
            )
            if len(actual) != len(expected):
                errors.append(f"Phase 13 validation row count differs for {track}.")
            else:
                for column in [KEY, "raw_probability", "calibrated_probability"]:
                    if column == KEY:
                        if not actual[column].equals(expected[column]):
                            errors.append(f"Phase 13 validation keys differ for {track}.")
                    elif not np.allclose(
                        actual[column].to_numpy(float),
                        expected[column].to_numpy(float),
                        atol=1.0e-10,
                        rtol=0,
                    ):
                        errors.append(f"Phase 13 validation probabilities differ for {track}.")
        if len(validation_targets) != int(
            audit.get("validation_target_rows_loaded_after_phase13_freeze", -1)
        ):
            errors.append("Phase 13 validation target access audit count differs.")
    except Exception as exc:
        errors.append(str(exc))
    return {
        "phase": 13,
        "valid": not errors,
        "hardening_status": "HARDENED_PASS" if not errors else "BLOCKED",
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "run_id": _read_json(directory / "phase13_manifest.json").get("run_id")
        if (directory / "phase13_manifest.json").is_file()
        else None,
        "test_seal": _read_json(directory / "target_access_audit.json")
        if (directory / "target_access_audit.json").is_file()
        else {},
    }


__all__ = ["validate_existing_phase13"]
