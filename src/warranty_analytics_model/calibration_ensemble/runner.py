"""Phase 13 calibration, controlled ensembling, and atomic publication."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
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
from ..feature_mart.manifest import git_commit_sha, write_json
from .calibration_folds import calibration_fold_assignments, calibration_fold_manifest
from .calibration_metrics import probability_metrics
from .calibrators import apply_calibrator, fit_calibrator
from .checkpoint import write_calibration_checkpoint
from .config import (
    CALIBRATION_METHODS,
    TRACKS,
    CalibrationEnsembleError,
    CalibrationEnsembleSettings,
    load_calibration_ensemble_settings,
    settings_payload,
)
from .contract import load_calibration_ensemble_contract, validate_calibration_ensemble_contract
from .ensemble import align_selected_tracks, evaluate_ensemble_weights
from .input import (
    KEY,
    TARGET,
    Phase12Lock,
    load_phase12_lock,
    write_phase12_parent_resolution,
)
from .planner import build_compute_plan
from .reporting import write_phase13_reports
from .selection import (
    accept_ensemble,
    accept_track_calibration,
    select_calibration_method,
    select_ensemble,
    select_phase13_champion,
)
from .thresholds import build_threshold_curve, select_mcc_threshold


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationEnsembleError(f"Invalid Phase 13 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise CalibrationEnsembleError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)


def _canonical_sha(value: Any) -> str:
    return canonical_json_sha256(value)


def _frame_sha(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    selected = frame if columns is None else frame.loc[:, columns]
    return _canonical_sha(selected.to_dict("records"))


def _artifact_hashes(directory: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative in {"phase13_manifest.json", "validation.json"}:
            continue
        result[relative] = sha256_file(path)
    return result


def phase13_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S_PHASE13")


def phase13_contract_check(*, project_root: Path | None = None) -> dict[str, Any]:
    return validate_calibration_ensemble_contract(project_root)


def phase13_plan_check(
    phase12_dir: Path,
    *,
    project_root: Path | None = None,
    max_workers: int | None = None,
    catboost_replay_threads: int | None = None,
) -> dict[str, Any]:
    root = (project_root or Path.cwd()).resolve()
    contract = validate_calibration_ensemble_contract(root)
    if not contract.get("valid"):
        return {"valid": False, "errors": contract.get("errors", []), "contract": contract}
    try:
        settings = load_calibration_ensemble_settings(root)
        lock = load_phase12_lock(phase12_dir, project_root=root)
        plan = build_compute_plan(
            settings,
            max_workers=max_workers,
            catboost_replay_threads=catboost_replay_threads,
        )
        return {
            "valid": True,
            "phase": 13,
            "phase12_run_id": lock.run_id,
            "compute_plan": plan.as_dict(),
            "settings": settings_payload(settings),
            "errors": [],
            "warnings": [],
        }
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)], "warnings": []}


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
    return _canonical_sha({"columns": columns, "rows": records})


def _metric_row(
    *,
    track: str,
    method: str,
    calibration_fold: str,
    role: str,
    metrics: dict[str, Any],
    eligible: bool,
    eligibility_reason: str,
) -> dict[str, Any]:
    return {
        "track": track,
        "calibration_method": method,
        "calibration_fold_id": calibration_fold,
        "evaluation_role": role,
        "eligible": bool(eligible),
        "eligibility_reason": eligibility_reason,
        **metrics,
    }


def _calibration_stage(
    lock: Phase12Lock,
    settings: CalibrationEnsembleSettings,
    work_dir: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    source = lock.source_oof.copy()
    source_for_assignments = source.loc[
        :, [KEY, "track", "strategy_id", "fold_id", "high_cost_probability", "claim_date"]
    ].copy()
    assignments = calibration_fold_assignments(source_for_assignments)
    fold_manifest, fold_sha = calibration_fold_manifest(assignments)
    assignments.to_parquet(work_dir / "calibration_fold_assignments.parquet", index=False)
    _write_json(work_dir / "calibration_fold_manifest.json", fold_manifest)

    all_crossfit: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    candidate_payload: dict[str, Any] = {"phase": 13, "tracks": {}}
    summary_rows: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}

    for track in TRACKS:
        track_source = source.loc[source["track"] == track].copy()
        candidate_payload["tracks"][track] = {}
        method_rows: list[dict[str, Any]] = []
        for method in CALIBRATION_METHODS:
            candidate_payload["tracks"][track][method] = {"folds": {}}
            predictions: list[pd.DataFrame] = []
            method_fold_rows: list[dict[str, Any]] = []
            method_eligible = True
            method_reason = "ELIGIBLE"
            definitions: tuple[dict[str, Any], ...] = (
                {"id": "C1", "train": (1,), "validation": 2},
                {"id": "C2", "train": (1, 2), "validation": 3},
            )
            for definition in definitions:
                calibration_fold = str(definition["id"])
                train = track_source.loc[track_source["fold_id"].isin(definition["train"])].copy()
                validation = track_source.loc[
                    track_source["fold_id"] == int(definition["validation"])
                ].copy()
                train_sha = _calibration_input_sha(train)
                validation_sha = _calibration_input_sha(validation)
                payload: dict[str, Any]
                if method == "C0_NONE":
                    payload = fit_calibrator(
                        method,
                        train["high_cost_probability"],
                        train["target"],
                        input_sha=train_sha,
                    )
                else:
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
                        input_sha=train_sha,
                    )
                eligible = bool(payload.get("eligible", True))
                reason = str(payload.get("eligibility_reason", "ELIGIBLE"))
                method_eligible = method_eligible and eligible
                if not eligible:
                    method_reason = reason
                candidate_payload["tracks"][track][method]["folds"][calibration_fold] = {
                    "calibrator": payload,
                    "training_input_sha": train_sha,
                    "validation_input_sha": validation_sha,
                }
                if not eligible:
                    continue
                calibrated = apply_calibrator(payload, validation["high_cost_probability"])
                internal = pd.DataFrame(
                    {
                        KEY: validation[KEY].astype(int).to_numpy(),
                        "source_fold_id": validation["fold_id"].astype(int).to_numpy(),
                        "calibration_fold_id": calibration_fold,
                        "track": track,
                        "calibration_method": method,
                        "raw_probability": validation["high_cost_probability"]
                        .astype(float)
                        .to_numpy(),
                        "calibrated_probability": calibrated,
                        "target": validation["target"].astype(int).to_numpy(),
                    }
                )
                predictions.append(internal)
                metrics = probability_metrics(
                    internal["target"],
                    internal["calibrated_probability"],
                    bins=settings.reliability_bins,
                    keys=internal[KEY],
                )
                method_fold_rows.append(
                    _metric_row(
                        track=track,
                        method=method,
                        calibration_fold=calibration_fold,
                        role="CALIBRATION_VALIDATION",
                        metrics=metrics,
                        eligible=eligible,
                        eligibility_reason=reason,
                    )
                )
                from .reliability import reliability_bins

                bins = reliability_bins(
                    internal["target"],
                    internal["calibrated_probability"],
                    bins=settings.reliability_bins,
                    keys=internal[KEY],
                )
                if not bins.empty:
                    bins.insert(0, "track", track)
                    bins.insert(1, "calibration_method", method)
                    bins.insert(2, "calibration_fold_id", calibration_fold)
                    reliability_rows.extend(bins.to_dict("records"))
                if settings.checkpoint_each_calibration_fold:
                    write_calibration_checkpoint(
                        work_dir,
                        track=track,
                        calibration_method=method,
                        calibration_fold=calibration_fold,
                        training_input_sha=train_sha,
                        validation_input_sha=validation_sha,
                        calibrator_sha=str(payload["calibrator_sha"]),
                        metrics=metrics,
                        prediction_sha=_frame_sha(internal),
                    )
            if predictions:
                method_predictions = pd.concat(predictions, ignore_index=True).sort_values(
                    ["calibration_fold_id", KEY], kind="mergesort"
                )
                pooled = probability_metrics(
                    method_predictions["target"],
                    method_predictions["calibrated_probability"],
                    bins=settings.reliability_bins,
                    keys=method_predictions[KEY],
                )
                aps = [float(row["average_precision"]) for row in method_fold_rows]
                summary = {
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
                    "eligible": bool(method_eligible),
                    "eligibility_reason": method_reason,
                }
                all_crossfit.append(method_predictions)
            else:
                summary = {
                    "track": track,
                    "calibration_method": method,
                    "pooled_average_precision": 0.0,
                    "mean_fold_average_precision": 0.0,
                    "min_fold_average_precision": 0.0,
                    "pooled_roc_auc": 0.0,
                    "pooled_log_loss": float("inf"),
                    "pooled_brier_score": float("inf"),
                    "pooled_ece": float("inf"),
                    "pooled_mce": float("inf"),
                    "row_count": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "eligible": False,
                    "eligibility_reason": method_reason,
                }
            method_rows.append(summary)
            fold_metric_rows.extend(method_fold_rows)
        method_summary = pd.DataFrame(method_rows)
        selection = select_calibration_method(method_summary)
        selections[track] = selection
        summary_rows.extend(method_rows)

    crossfit = pd.concat(all_crossfit, ignore_index=True) if all_crossfit else pd.DataFrame()
    crossfit_columns = [
        KEY,
        "source_fold_id",
        "calibration_fold_id",
        "track",
        "calibration_method",
        "raw_probability",
        "calibrated_probability",
    ]
    crossfit.loc[:, crossfit_columns].sort_values(
        ["track", "calibration_method", "calibration_fold_id", KEY], kind="mergesort"
    ).to_parquet(work_dir / "calibration_crossfit_predictions.parquet", index=False)
    pd.DataFrame(fold_metric_rows).to_parquet(
        work_dir / "calibration_fold_metrics.parquet", index=False
    )
    pd.DataFrame(summary_rows).to_parquet(work_dir / "calibration_summary.parquet", index=False)
    pd.DataFrame(reliability_rows).to_parquet(work_dir / "reliability_bins.parquet", index=False)
    _write_json(work_dir / "calibration_candidates.json", candidate_payload)
    _write_json(
        work_dir / "calibration_selection.json",
        {"phase": 13, "tracks": selections, "selection_sha256": _canonical_sha(selections)},
    )
    return {
        "assignments": assignments,
        "fold_manifest": fold_manifest,
        "fold_sha": fold_sha,
        "crossfit": crossfit,
        "fold_metrics": pd.DataFrame(fold_metric_rows),
        "summary": pd.DataFrame(summary_rows),
        "selections": selections,
        "candidate_payload": candidate_payload,
    }


def _selected_oof(
    calibration: dict[str, Any],
    settings: CalibrationEnsembleSettings,
    lock: Phase12Lock,
    work_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    selected: dict[str, pd.DataFrame] = {}
    final_payloads: dict[str, dict[str, Any]] = {}
    source = lock.source_oof
    for track in TRACKS:
        method = str(calibration["selections"][track]["selected_calibration_method"])
        track_source = source.loc[source["track"] == track].copy()
        payload = fit_calibrator(
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
        if not payload.get("eligible", True):
            raise CalibrationEnsembleError(
                f"Selected calibration is ineligible for {track}: {method}"
            )
        final_payloads[track] = payload
        fit = (
            calibration["crossfit"]
            .loc[
                (calibration["crossfit"]["track"] == track)
                & (calibration["crossfit"]["calibration_method"] == method)
            ]
            .copy()
        )
        if fit.empty:
            raise CalibrationEnsembleError(
                f"Selected calibration has no cross-fit rows for {track}."
            )
        selected[track] = fit.sort_values(
            ["calibration_fold_id", KEY], kind="mergesort"
        ).reset_index(drop=True)
    selected_frame = pd.concat(
        [
            frame.loc[
                :,
                [
                    KEY,
                    "source_fold_id",
                    "calibration_fold_id",
                    "track",
                    "raw_probability",
                    "calibrated_probability",
                ],
            ]
            for frame in selected.values()
        ],
        ignore_index=True,
    )
    selected_frame.to_parquet(work_dir / "selected_calibrated_oof_predictions.parquet", index=False)
    calibrator_dir = work_dir / "calibrators"
    for track, payload in final_payloads.items():
        _write_json(calibrator_dir / f"{track.lower()}.json", payload)
    return selected, final_payloads


def _ensemble_stage_with_source(
    selected: dict[str, pd.DataFrame],
    source: pd.DataFrame,
    settings: CalibrationEnsembleSettings,
    work_dir: Path,
) -> dict[str, Any]:
    source_targets = source.loc[:, [KEY, "target"]].drop_duplicates(KEY).set_index(KEY)["target"]
    frames: dict[str, pd.DataFrame] = {}
    for track, frame in selected.items():
        current = frame.copy()
        current["target"] = current[KEY].map(source_targets).astype("int8")
        frames[track] = current
    aligned = align_selected_tracks(frames["T1"], frames["T3"])
    prediction_frame, summary = evaluate_ensemble_weights(
        aligned, weights=settings.ensemble_weights, bins=settings.reliability_bins
    )
    summary = summary.rename(
        columns={
            "average_precision": "pooled_average_precision",
            "roc_auc": "pooled_roc_auc",
            "log_loss": "pooled_log_loss",
            "brier_score": "pooled_brier_score",
            "ece_10": "pooled_ece",
            "mce_10": "pooled_mce",
        }
    )
    summary["candidate_id"] = summary["t1_weight"].map(
        lambda weight: f"P13_ENSEMBLE_W{int(round(float(weight) * 10)):02d}"
    )
    fold_rows: list[dict[str, Any]] = []
    for row in prediction_frame.to_dict("records"):
        fold_rows.append(row)
    fold_metrics: list[dict[str, Any]] = []
    for (weight, fold), part in prediction_frame.groupby(
        ["t1_weight", "calibration_fold_id"], sort=True
    ):
        metric = probability_metrics(
            part["target"],
            part["ensemble_probability"],
            bins=settings.reliability_bins,
            keys=part[KEY],
        )
        fold_metrics.append(
            {
                "candidate_id": f"P13_ENSEMBLE_W{int(round(float(weight) * 10)):02d}",
                "t1_weight": float(weight),
                "t3_weight": round(1.0 - float(weight), 1),
                "calibration_fold_id": str(fold),
                **metric,
            }
        )
    prediction_columns = [
        KEY,
        "source_fold_id",
        "calibration_fold_id",
        "t1_weight",
        "t3_weight",
        "p_t1",
        "p_t3",
        "ensemble_probability",
    ]
    prediction_frame.loc[:, prediction_columns].to_parquet(
        work_dir / "ensemble_crossfit_predictions.parquet", index=False
    )
    summary.to_parquet(work_dir / "ensemble_summary.parquet", index=False)
    pd.DataFrame(fold_metrics).to_parquet(work_dir / "ensemble_fold_metrics.parquet", index=False)
    selection = select_ensemble(summary)
    _write_json(work_dir / "ensemble_selection.json", selection)
    _write_json(
        work_dir / "ensemble_candidates.json",
        {"phase": 13, "weights": summary.to_dict("records"), "selection": selection},
    )
    return {
        "aligned": aligned,
        "predictions": prediction_frame,
        "summary": summary,
        "fold_metrics": pd.DataFrame(fold_metrics),
        "selection": selection,
    }


def _threshold_stage(
    selected: dict[str, pd.DataFrame],
    ensemble: dict[str, Any],
    source: pd.DataFrame,
    settings: CalibrationEnsembleSettings,
    work_dir: Path,
) -> dict[str, Any]:
    source_targets = source.loc[:, [KEY, "target"]].drop_duplicates(KEY).set_index(KEY)["target"]
    curves: list[pd.DataFrame] = []
    policy: dict[str, Any] = {"phase": 13, "objective": "MCC_MAX", "candidates": {}}
    for track in TRACKS:
        frame = selected[track]
        y = frame[KEY].map(source_targets).astype("int8")
        candidate_id = f"P13_{track}_CALIBRATED"
        curve = build_threshold_curve(
            y,
            frame["calibrated_probability"],
            candidate_id=candidate_id,
            score_space="CALIBRATED_PROBABILITY",
            start=settings.threshold_start,
            stop=settings.threshold_stop,
            step=settings.threshold_step,
        )
        curves.append(curve)
        policy["candidates"][track] = select_mcc_threshold(curve, settings.threshold_tie_tolerance)
    selection = ensemble["selection"]
    if selection.get("selected_policy") == "TRUE_BLEND":
        weight = float(selection["selected_weight"])
        row = ensemble["predictions"].loc[ensemble["predictions"]["t1_weight"] == weight]
        y = row[KEY].map(source_targets).astype("int8")
        candidate_id = f"P13_ENSEMBLE_W{int(round(weight * 10)):02d}"
        curve = build_threshold_curve(
            y,
            row["ensemble_probability"],
            candidate_id=candidate_id,
            score_space="CALIBRATED_ENSEMBLE_PROBABILITY",
            start=settings.threshold_start,
            stop=settings.threshold_stop,
            step=settings.threshold_step,
        )
        curves.append(curve)
        policy["candidates"]["ENSEMBLE"] = select_mcc_threshold(
            curve, settings.threshold_tie_tolerance
        )
    curve_frame = pd.concat(curves, ignore_index=True)
    curve_frame.to_parquet(work_dir / "threshold_curve.parquet", index=False)
    policy["threshold_curve_sha256"] = _frame_sha(curve_frame)
    _write_json(work_dir / "threshold_policy.json", policy)
    return {"curve": curve_frame, "policy": policy}


def _validation_stage(
    lock: Phase12Lock,
    settings: CalibrationEnsembleSettings,
    selected: dict[str, pd.DataFrame],
    final_calibrators: dict[str, dict[str, Any]],
    calibration: dict[str, Any],
    ensemble: dict[str, Any],
    thresholds: dict[str, Any],
    work_dir: Path,
) -> dict[str, Any]:
    """Score outer VALIDATION only after the immutable Phase 13 freeze exists."""

    phase10 = lock.phase12_inputs.phase10_inputs
    validation_targets, validation_audit = load_validation_targets_after_freeze(
        phase10, study_frozen=True
    )
    settings_baseline = load_baseline_settings(lock.root)
    validation_frame = phase10.development.loc[phase10.development["split"] == "VALIDATION"].copy()
    target_map = validation_targets.set_index(KEY)[TARGET]
    prediction_rows: list[dict[str, Any]] = []
    metric_payload: dict[str, Any] = {"phase": 13, "tracks": {}, "ensemble": None}
    effective_track_probs: dict[str, np.ndarray] = {}
    effective_track_meta: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for track in TRACKS:
        experiment = TRACK_TO_EXPERIMENT[track]
        feature_set = phase10.feature_sets[experiment]
        matrix = adapt_matrix(validation_frame, feature_set, settings_baseline)
        model_path = lock.phase12_dir / str(lock.effective_models[track]["model_file"])
        raw = predict_probabilities(load_model(model_path), matrix, feature_set)
        keys = validation_frame[KEY].astype(int).to_numpy()
        y = validation_frame[KEY].map(target_map).astype("int8").to_numpy()
        raw_metrics = probability_metrics(y, raw, bins=settings.reliability_bins, keys=keys)
        raw_threshold = float(str(lock.effective_models[track].get("technical_threshold")))
        from ..imbalance_threshold.metrics import threshold_metrics

        raw_metrics.update(threshold_metrics(y, raw, raw_threshold))
        calibrated = apply_calibrator(final_calibrators[track], raw)
        calibrated_metrics = probability_metrics(
            y, calibrated, bins=settings.reliability_bins, keys=keys
        )
        selected_threshold = float(thresholds["policy"]["candidates"][track]["threshold"])
        calibrated_metrics.update(threshold_metrics(y, calibrated, selected_threshold))
        calibration_acceptance = accept_track_calibration(raw_metrics, calibrated_metrics, settings)
        accepted = bool(calibration_acceptance.get("accepted"))
        if not accepted:
            warnings.append("CALIBRATION_REJECTED_ON_VALIDATION")
        effective = calibrated if accepted else raw
        effective_threshold = selected_threshold if accepted else raw_threshold
        effective_space = "CALIBRATED_PROBABILITY" if accepted else "RAW_UNCALIBRATED_PROBABILITY"
        effective_metrics = calibrated_metrics if accepted else raw_metrics
        effective_track_probs[track] = effective
        effective_track_meta[track] = {
            "candidate_id": (
                f"P13_{track}_CALIBRATED_{final_calibrators[track]['method']}"
                if accepted and final_calibrators[track]["method"] != "NONE"
                else str(lock.effective_models[track]["candidate_id"])
            ),
            "accepted": accepted,
            "calibration_acceptance": calibration_acceptance,
            "threshold": effective_threshold,
            "score_space": effective_space,
            "raw_metrics": raw_metrics,
            "calibrated_metrics": calibrated_metrics,
            "effective_metrics": effective_metrics,
        }
        metric_payload["tracks"][track] = effective_track_meta[track]
        for key, raw_value, calibrated_value, effective_value in zip(
            keys, raw, calibrated, effective, strict=True
        ):
            prediction_rows.append(
                {
                    KEY: int(key),
                    "track": track,
                    "candidate_id": effective_track_meta[track]["candidate_id"],
                    "raw_probability": float(raw_value),
                    "calibrated_probability": float(calibrated_value),
                    "effective_probability": float(effective_value),
                }
            )
    effective_single_rows: list[dict[str, Any]] = []
    for track in TRACKS:
        item = effective_track_meta[track]
        effective_single_rows.append(
            {
                "candidate_id": item["candidate_id"],
                "validation_metrics": item["effective_metrics"],
                "complexity_order": 1 if item["accepted"] else 0,
                "feature_count": int(lock.effective_models[track].get("feature_count", 0)),
            }
        )

    ensemble_candidate: dict[str, Any] | None = None
    ensemble_acceptance: dict[str, Any] | None = None
    if ensemble["selection"].get("selected_policy") == "TRUE_BLEND":
        weight = float(ensemble["selection"]["selected_weight"])
        p1 = effective_track_probs["T1"]
        p3 = effective_track_probs["T3"]
        p = weight * p1 + (1.0 - weight) * p3
        y = validation_frame[KEY].map(target_map).astype("int8").to_numpy()
        metrics = probability_metrics(
            y, p, bins=settings.reliability_bins, keys=validation_frame[KEY]
        )
        from ..imbalance_threshold.metrics import threshold_metrics

        selected_threshold = float(thresholds["policy"]["candidates"]["ENSEMBLE"]["threshold"])
        metrics.update(threshold_metrics(y, p, selected_threshold))
        best_single = sorted(
            effective_single_rows,
            key=lambda item: (
                -float(item["validation_metrics"]["average_precision"]),
                str(item["candidate_id"]),
            ),
        )[0]
        ensemble_acceptance = accept_ensemble(metrics, best_single["validation_metrics"], settings)
        if not ensemble_acceptance["accepted"]:
            warnings.append("ENSEMBLE_REJECTED_ON_VALIDATION")
        else:
            ensemble_candidate = {
                "candidate_id": f"P13_ENSEMBLE_W{int(round(weight * 10)):02d}",
                "validation_metrics": metrics,
                "complexity_order": 2,
                "feature_count": int(lock.effective_models["T1"].get("feature_count", 0))
                + int(lock.effective_models["T3"].get("feature_count", 0)),
                "t1_weight": weight,
                "t3_weight": round(1.0 - weight, 1),
                "threshold": selected_threshold,
                "score_space": "CALIBRATED_ENSEMBLE_PROBABILITY",
            }
        metric_payload["ensemble"] = {
            "selected_policy": ensemble["selection"].get("selected_policy"),
            "selected_weight": weight,
            "metrics": metrics,
            "acceptance": ensemble_acceptance,
        }

    validation_predictions = pd.DataFrame(prediction_rows)
    validation_predictions.to_parquet(work_dir / "validation_predictions.parquet", index=False)
    candidates = effective_single_rows + ([ensemble_candidate] if ensemble_candidate else [])
    champion = select_phase13_champion(candidates)
    metric_payload["phase13_development_champion"] = champion
    metric_payload["warnings"] = sorted(set(warnings))
    _write_json(work_dir / "validation_metrics.json", metric_payload)
    return {
        "validation_targets": validation_targets,
        "validation_audit": validation_audit,
        "predictions": validation_predictions,
        "metrics": metric_payload,
        "track_meta": effective_track_meta,
        "ensemble_candidate": ensemble_candidate,
        "ensemble_acceptance": ensemble_acceptance,
        "champion": champion,
        "warnings": sorted(set(warnings)),
    }


def _effective_manifest(
    lock: Phase12Lock,
    final_calibrators: dict[str, dict[str, Any]],
    validation: dict[str, Any],
    ensemble: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for track in TRACKS:
        source = lock.effective_models[track]
        meta = validation["track_meta"][track]
        entries.append(
            {
                "candidate_id": meta["candidate_id"],
                "candidate_type": "SINGLE_TRACK",
                "track": track,
                "source_phase12_candidate_id": source.get("candidate_id"),
                "source_model_sha256": source.get("model_sha256"),
                "model_file": source.get("model_file"),
                "feature_count": source.get("feature_count"),
                "feature_set_sha256": source.get("feature_set_sha256"),
                "feature_list_sha256": source.get("feature_list_sha256"),
                "imbalance_strategy": source.get("selected_imbalance_strategy"),
                "calibration_method": final_calibrators[track].get("method"),
                "calibrator_sha": final_calibrators[track].get("calibrator_sha"),
                "score_space": meta["score_space"],
                "technical_threshold": meta["threshold"],
                "validation_metrics": meta["effective_metrics"],
                "acceptance": meta["calibration_acceptance"],
            }
        )
    if validation.get("ensemble_candidate"):
        candidate = validation["ensemble_candidate"]
        entries.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_type": "ENSEMBLE",
                "track": "T1_T3",
                "source_phase12_candidate_ids": {
                    track: lock.effective_models[track].get("candidate_id") for track in TRACKS
                },
                "source_model_sha256": {
                    track: lock.effective_models[track].get("model_sha256") for track in TRACKS
                },
                "t1_weight": candidate["t1_weight"],
                "t3_weight": candidate["t3_weight"],
                "score_space": candidate["score_space"],
                "technical_threshold": candidate["threshold"],
                "validation_metrics": candidate["validation_metrics"],
                "acceptance": validation["ensemble_acceptance"],
            }
        )
    return {
        "phase": 13,
        "phase12_run_id": lock.run_id,
        "phase12_dir": str(lock.phase12_dir),
        "models": entries,
        "selected_ensemble_policy": ensemble["selection"].get("selected_policy"),
        "selected_ensemble_weight": ensemble["selection"].get("selected_weight"),
        "threshold_policy_sha256": _canonical_sha(thresholds["policy"]),
    }


def build_phase13(
    phase12_dir: Path,
    *,
    project_root: Path | None = None,
    output_dir: Path | None = None,
    report_dir: Path | None = None,
    run_id: str | None = None,
    resume: bool = False,
    max_workers: int | None = None,
    catboost_replay_threads: int | None = None,
) -> dict[str, Any]:
    root = (project_root or Path.cwd()).expanduser().resolve()
    settings = load_calibration_ensemble_settings(root)
    contract, contract_sha = load_calibration_ensemble_contract(root)
    contract_result = validate_calibration_ensemble_contract(root)
    if not contract_result.get("valid"):
        raise CalibrationEnsembleError(
            "Phase 13 contract is blocked: " + "; ".join(contract_result["errors"])
        )
    lock = load_phase12_lock(phase12_dir, project_root=root)
    plan = build_compute_plan(
        settings,
        max_workers=max_workers,
        catboost_replay_threads=catboost_replay_threads,
    )
    output_root = (output_dir or root / settings.output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_root = (report_dir or root / settings.report_directory).expanduser().resolve()
    selected_run_id = str(run_id or phase13_run_id())
    final_dir = output_root / selected_run_id
    if final_dir.exists():
        raise CalibrationEnsembleError(f"Published Phase 13 run is immutable: {final_dir}")
    work_dir = output_root / f".phase13_{selected_run_id}.work"
    if work_dir.exists() and not resume:
        raise CalibrationEnsembleError(
            f"Phase 13 work directory already exists; use --resume or choose a new run id: {work_dir}"
        )
    work_dir.mkdir(parents=True, exist_ok=True)

    parent = write_phase12_parent_resolution(work_dir / "phase12_parent_resolution.json", lock)
    calibration = _calibration_stage(lock, settings, work_dir, resume=resume)
    selected, final_calibrators = _selected_oof(calibration, settings, lock, work_dir)
    ensemble = _ensemble_stage_with_source(selected, lock.source_oof, settings, work_dir)
    thresholds = _threshold_stage(selected, ensemble, lock.source_oof, settings, work_dir)

    freeze_without_hash: dict[str, Any] = {
        "phase": 13,
        "phase13_run_id": selected_run_id,
        "phase12_run_id": lock.run_id,
        "phase12_manifest_sha256": lock.phase12_manifest_sha256,
        "phase12_validation_sha256": lock.phase12_validation_sha256,
        "phase12_effective_model_manifest_sha256": lock.phase12_effective_model_manifest_sha256,
        "phase12_freeze_sha256": lock.phase12_freeze_sha256,
        "phase12_test_seal": {
            "test_target_rows_loaded": 0,
            "test_predictions_created": 0,
            "test_metrics_computed": False,
            "test_target_access_allowed": False,
            "first_allowed_test_target_phase": 15,
        },
        "effective_models": {
            track: lock.effective_models[track].get("candidate_id") for track in TRACKS
        },
        "selected_calibration": {
            track: {
                "method": calibration["selections"][track]["selected_calibration_method"],
                "calibrator_sha": final_calibrators[track]["calibrator_sha"],
            }
            for track in TRACKS
        },
        "calibration_fold_sha256": calibration["fold_sha"],
        "calibration_selection_evidence_sha256": _canonical_sha(calibration["selections"]),
        "selected_ensemble_policy": ensemble["selection"].get("selected_policy"),
        "ensemble_t1_weight": ensemble["selection"].get("selected_weight"),
        "ensemble_evidence_sha256": _canonical_sha(ensemble["summary"].to_dict("records")),
        "calibrated_thresholds": thresholds["policy"]["candidates"],
        "threshold_evidence_sha256": thresholds["policy"]["threshold_curve_sha256"],
        "outer_validation_accessed": False,
        "test_target_rows_loaded": 0,
        "test_target_access_allowed": False,
        "test_target_accessed": False,
        "test_predictions_created": False,
        "test_metrics_computed": False,
        "first_allowed_test_target_phase": 15,
    }
    freeze = {**freeze_without_hash, "phase13_freeze_sha256": _canonical_sha(freeze_without_hash)}
    _write_json(work_dir / "phase13_freeze.json", freeze)

    validation = _validation_stage(
        lock,
        settings,
        selected,
        final_calibrators,
        calibration,
        ensemble,
        thresholds,
        work_dir,
    )
    effective_manifest = _effective_manifest(
        lock, final_calibrators, validation, ensemble, thresholds
    )
    _write_json(work_dir / "effective_model_manifest.json", effective_manifest)

    train_targets = lock.train_targets
    audit = {
        "phase": 13,
        "train_target_rows_loaded": int(len(train_targets)),
        "train_positive_rows_loaded": int((train_targets[TARGET] == 1).sum()),
        "train_negative_rows_loaded": int((train_targets[TARGET] == 0).sum()),
        "validation_target_rows_loaded_before_phase13_freeze": 0,
        "validation_target_rows_loaded_after_phase13_freeze": int(
            len(validation["validation_targets"])
        ),
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
    }
    _write_json(work_dir / "target_access_audit.json", audit)
    _write_json(work_dir / "compute_manifest.json", plan.as_dict())

    manifest_without_hash: dict[str, Any] = {
        "phase": 13,
        "run_id": selected_run_id,
        "created_at_utc": _now(),
        "git_commit_sha": git_commit_sha(root),
        "contract_version": contract["phase13"].get("version"),
        "contract_sha256": contract_sha,
        "phase12_run_id": lock.run_id,
        "phase12_dir": str(lock.phase12_dir),
        "phase12_manifest_sha256": lock.phase12_manifest_sha256,
        "phase12_validation_sha256": lock.phase12_validation_sha256,
        "phase12_effective_model_manifest_sha256": lock.phase12_effective_model_manifest_sha256,
        "phase12_freeze_sha256": lock.phase12_freeze_sha256,
        "calibration_fold_sha256": calibration["fold_sha"],
        "selected_calibration": {
            track: {
                "method": calibration["selections"][track]["selected_calibration_method"],
                "calibrator_sha": final_calibrators[track]["calibrator_sha"],
            }
            for track in TRACKS
        },
        "selected_ensemble_policy": ensemble["selection"].get("selected_policy"),
        "ensemble_t1_weight": ensemble["selection"].get("selected_weight"),
        "phase13_technical_thresholds": thresholds["policy"]["candidates"],
        "phase13_freeze_sha256": sha256_file(work_dir / "phase13_freeze.json"),
        "phase13_development_champion": validation["champion"],
        "effective_candidates": effective_manifest["models"],
        "compute_plan": plan.as_dict(),
        "target_access_audit": audit,
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
        "warnings": validation["warnings"],
    }
    manifest_without_hash["artifact_file_sha256"] = _artifact_hashes(work_dir)
    _write_json(work_dir / "phase13_manifest.json", manifest_without_hash)

    # The independent validator deliberately runs against the temporary bundle before
    # publication.  It does not trust the runner's status declaration.
    from .validation import validate_existing_phase13

    independent = validate_existing_phase13(work_dir, project_root=root)
    if not independent.get("valid") or independent.get("hardening_status") != "HARDENED_PASS":
        raise CalibrationEnsembleError(
            "Independent Phase 13 validation failed: " + "; ".join(independent.get("errors", []))
        )
    validation_json = {
        **independent,
        "phase": 13,
        "run_id": selected_run_id,
        "valid": True,
        "hardening_status": "HARDENED_PASS",
        "test_seal": audit,
    }
    _write_json(work_dir / "validation.json", validation_json)
    if final_dir.exists():
        raise CalibrationEnsembleError(
            f"Published Phase 13 run appeared during publication: {final_dir}"
        )
    os.replace(work_dir, final_dir)

    report_payload = {
        "run_id": selected_run_id,
        "phase13_dir": str(final_dir),
        "validation": validation_json,
        "validation_metrics": validation["metrics"],
        "calibration_summary": calibration["summary"].to_dict("records"),
        "ensemble_selection": ensemble["selection"],
        "threshold_policy": thresholds["policy"],
        "parent_resolution": parent,
    }
    write_phase13_reports(report_root, selected_run_id, report_payload)
    return {
        "phase": 13,
        "run_id": selected_run_id,
        "phase13_dir": str(final_dir),
        "report_dir": str(report_root / selected_run_id),
        "validation": validation_json,
        "phase13_development_champion": validation["champion"],
        "selected_calibration": {
            track: calibration["selections"][track]["selected_calibration_method"]
            for track in TRACKS
        },
        "selected_ensemble_policy": ensemble["selection"].get("selected_policy"),
        "selected_ensemble_weight": ensemble["selection"].get("selected_weight"),
        "warnings": validation["warnings"],
        "compute_plan": plan.as_dict(),
    }


__all__ = [
    "build_phase13",
    "phase13_contract_check",
    "phase13_plan_check",
    "phase13_run_id",
]
