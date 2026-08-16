"""Focused Phase 13 calibration, ensemble, threshold, and contract tests."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import warranty_analytics_model.calibration_ensemble.checkpoint as phase13_checkpoint
import warranty_analytics_model.calibration_ensemble.config as phase13_config
import warranty_analytics_model.calibration_ensemble.input as phase13_input
import warranty_analytics_model.calibration_ensemble.runner as phase13_runner
import warranty_analytics_model.calibration_ensemble.validation as phase13_validation
from warranty_analytics_model.calibration_ensemble.calibration_folds import (
    calibration_fold_assignments,
    calibration_fold_manifest,
)
from warranty_analytics_model.calibration_ensemble.calibration_metrics import probability_metrics
from warranty_analytics_model.calibration_ensemble.calibrators import (
    apply_calibrator,
    calibrator_sha,
    fit_calibrator,
    isotonic_eligibility,
    sigmoid_logit,
)
from warranty_analytics_model.calibration_ensemble.checkpoint import (
    load_valid_calibration_checkpoint,
    write_calibration_checkpoint,
)
from warranty_analytics_model.calibration_ensemble.config import (
    CALIBRATION_METHODS,
    ENSEMBLE_WEIGHTS,
    CalibrationEnsembleError,
    load_calibration_ensemble_settings,
    locked_configuration_sha256,
    settings_payload,
)
from warranty_analytics_model.calibration_ensemble.contract import (
    validate_calibration_ensemble_contract,
)
from warranty_analytics_model.calibration_ensemble.ensemble import (
    align_selected_tracks,
    blend_probability,
    evaluate_ensemble_weights,
)
from warranty_analytics_model.calibration_ensemble.planner import build_compute_plan
from warranty_analytics_model.calibration_ensemble.reliability import ece_mce, reliability_bins
from warranty_analytics_model.calibration_ensemble.reporting import write_phase13_reports
from warranty_analytics_model.calibration_ensemble.runner import (
    _artifact_hashes,
    _calibration_stage,
    _ensemble_stage_with_source,
    _frame_sha,
    _metric_row,
    _read_json,
    _selected_oof,
    _threshold_stage,
    _validation_stage,
    build_phase13,
    phase13_contract_check,
    phase13_plan_check,
)
from warranty_analytics_model.calibration_ensemble.selection import (
    accept_ensemble,
    accept_track_calibration,
    compare_champion_candidates,
    select_ensemble,
    select_phase13_champion,
)
from warranty_analytics_model.calibration_ensemble.thresholds import (
    build_threshold_curve,
    select_mcc_threshold,
    threshold_grid,
)
from warranty_analytics_model.calibration_ensemble.validation import validate_existing_phase13
from warranty_analytics_model.cli import build_parser


def _source_oof() -> pd.DataFrame:
    rows = []
    key = 1
    for fold, date in ((1, "2024-01-01"), (2, "2024-02-01"), (3, "2024-03-01")):
        for offset in range(4):
            rows.append(
                {
                    "warranty_claim_key": key,
                    "track": "T1" if offset % 2 == 0 else "T3",
                    "strategy_id": "S0_NONE",
                    "fold_id": fold,
                    "high_cost_probability": 0.05 + 0.1 * offset + 0.01 * fold,
                    "claim_date": date,
                }
            )
            key += 1
    return pd.DataFrame(rows)


def _large_source_oof() -> pd.DataFrame:
    """A balanced, identical T1/T3 population for exercising Stage A."""

    rows = []
    for track in ("T1", "T3"):
        for fold, date in ((1, "2024-01-01"), (2, "2024-02-01"), (3, "2024-03-01")):
            for offset in range(90):
                rows.append(
                    {
                        "warranty_claim_key": (fold - 1) * 90 + offset + 1,
                        "track": track,
                        "strategy_id": "S0_NONE",
                        "fold_id": fold,
                        "high_cost_probability": 0.01 + 0.98 * offset / 89.0,
                        "claim_date": date,
                        "target": int(offset < 30),
                    }
                )
    return pd.DataFrame(rows)


def test_phase13_contract_configuration_and_cli_are_locked() -> None:
    settings = load_calibration_ensemble_settings()
    assert settings.calibration_methods == CALIBRATION_METHODS
    assert settings.ensemble_weights == ENSEMBLE_WEIGHTS
    assert validate_calibration_ensemble_contract()["valid"] is True
    parsed = build_parser().parse_args(
        [
            "phase13-calibrate",
            "--phase12-dir",
            "phase12",
            "--max-workers",
            "4",
            "--catboost-replay-threads",
            "8",
        ]
    )
    assert parsed.phase12_dir == Path("phase12")
    assert parsed.max_workers == 4
    assert parsed.catboost_replay_threads == 8


def test_phase13_configuration_rejects_any_experimental_drift(monkeypatch) -> None:
    original_loader = phase13_config.yaml.safe_load

    def drifted_loader(stream):
        payload = original_loader(stream)
        payload["phase13_calibration_ensemble"]["validation"][
            "ensemble_max_log_loss_regression"
        ] = 0.004
        return payload

    monkeypatch.setattr(phase13_config.yaml, "safe_load", drifted_loader)
    with pytest.raises(CalibrationEnsembleError, match="experimental payload"):
        phase13_config.load_calibration_ensemble_settings()


def test_calibrators_are_transparent_and_guarded() -> None:
    raw = np.linspace(0.01, 0.99, 150)
    target = np.array([1] * 30 + [0] * 120, dtype="int8")
    assert np.allclose(sigmoid_logit([0.0, 1.0]), [-13.81550956, 13.81550956])
    sigmoid = fit_calibrator("C1_SIGMOID", raw, target)
    assert sigmoid["method"] == "SIGMOID"
    assert sigmoid["calibrator_sha"] == calibrator_sha(sigmoid)
    assert np.isfinite(apply_calibrator(sigmoid, raw)).all()
    isotonic = fit_calibrator("C2_ISOTONIC", raw, target)
    assert isotonic["method"] == "ISOTONIC"
    assert len(isotonic["X_thresholds"]) == len(isotonic["y_thresholds"])
    assert isotonic_eligibility(target[:10], raw[:10])[0] is False
    ineligible = fit_calibrator("C2_ISOTONIC", raw[:10], target[:10])
    assert ineligible["eligible"] is False
    assert ineligible["calibrator_sha"] == calibrator_sha(ineligible)
    assert np.allclose(apply_calibrator({"method": "NONE"}, raw), raw)
    with pytest.raises(ValueError):
        fit_calibrator("C1_SIGMOID", raw, np.zeros_like(target))


def test_reliability_metrics_and_temporal_folds_are_deterministic() -> None:
    y = np.array([0, 1, 0, 1, 0, 1], dtype="int8")
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    bins = reliability_bins(y, p, bins=3, keys=[6, 5, 4, 3, 2, 1])
    assert bins["row_count"].tolist() == [2, 2, 2]
    assert ece_mce(y, p, bins=3)[0] >= 0
    metrics = probability_metrics(y, p, bins=3, keys=np.arange(6))
    assert metrics["row_count"] == 6
    source = _source_oof()
    assignments = calibration_fold_assignments(source)
    manifest, digest = calibration_fold_manifest(assignments)
    assert manifest["source_folds_used_for_evaluation"] == [2, 3]
    assert manifest["calibration_fold_content_sha256"] == digest
    assert set(assignments["calibration_fold_id"]) == {"C1", "C2"}
    with pytest.raises(ValueError):
        calibration_fold_assignments(source.drop(columns="claim_date"))


def test_controlled_ensemble_selection_and_threshold_ties() -> None:
    keys = np.arange(1, 7)
    t1 = pd.DataFrame(
        {
            "warranty_claim_key": keys,
            "source_fold_id": [1, 1, 2, 2, 3, 3],
            "calibration_fold_id": ["C1", "C1", "C1", "C1", "C2", "C2"],
            "track": "T1",
            "calibrated_probability": [0.1, 0.8, 0.2, 0.7, 0.3, 0.6],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    t3 = t1.copy()
    t3["track"] = "T3"
    t3["calibrated_probability"] = [0.2, 0.7, 0.3, 0.6, 0.4, 0.5]
    aligned = align_selected_tracks(t1, t3)
    predictions, summary = evaluate_ensemble_weights(aligned)
    assert set(summary["t1_weight"]) == set(ENSEMBLE_WEIGHTS)
    assert len(predictions) == len(keys) * 11
    assert np.allclose(blend_probability([0.2], [0.8], 0.25), [0.65])
    with pytest.raises(ValueError):
        blend_probability([0.2], [0.8], 1.1)
    policy_summary = pd.DataFrame(
        [
            {
                "t1_weight": weight,
                "pooled_average_precision": 0.5 + (0.1 if weight == 0.5 else 0),
                "min_fold_average_precision": 0.4,
                "pooled_roc_auc": 0.7,
                "pooled_log_loss": 0.2,
                "pooled_brier_score": 0.1,
            }
            for weight in ENSEMBLE_WEIGHTS
        ]
    )
    assert select_ensemble(policy_summary)["selected_policy"] == "TRUE_BLEND"
    curve = build_threshold_curve(
        y_true=[0, 1, 0, 1],
        probabilities=[0.1, 0.9, 0.2, 0.8],
        candidate_id="x",
        score_space="CALIBRATED_PROBABILITY",
    )
    assert len(threshold_grid()) == 999
    assert select_mcc_threshold(curve)["threshold"] < 1
    assert (
        compare_champion_candidates(
            {"candidate_id": "a", "validation_metrics": {"average_precision": 0.5}},
            {"candidate_id": "b", "validation_metrics": {"average_precision": 0.4}},
        )
        < 0
    )
    assert (
        select_phase13_champion(
            [
                {"candidate_id": "raw", "validation_metrics": {"average_precision": 0.5}},
                {
                    "candidate_id": "cal",
                    "validation_metrics": {"average_precision": 0.5},
                    "complexity_order": 1,
                },
            ]
        )
        == "raw"
    )


def test_stage_a_calibration_ensemble_and_threshold_artifacts(monkeypatch, tmp_path: Path) -> None:
    source = _large_source_oof()
    settings = load_calibration_ensemble_settings()
    lock = SimpleNamespace(
        source_oof=source, train_targets=source.drop_duplicates("warranty_claim_key")
    )
    work_dir = tmp_path / "phase13-work"
    work_dir.mkdir()

    calibration = _calibration_stage(lock, settings, work_dir, resume=False)
    selected, final_calibrators = _selected_oof(calibration, settings, lock, work_dir)
    ensemble = _ensemble_stage_with_source(selected, source, settings, work_dir)
    thresholds = _threshold_stage(selected, ensemble, source, settings, work_dir)

    assert set(calibration["selections"]) == {"T1", "T3"}
    assert set(final_calibrators) == {"T1", "T3"}
    assert all(not frame.empty for frame in selected.values())
    assert len(ensemble["summary"]) == 11
    assert not thresholds["curve"].empty
    assert (work_dir / "calibration_fold_assignments.parquet").is_file()
    assert (work_dir / "calibrators" / "t1.json").is_file()
    assert (work_dir / "ensemble_candidates.json").is_file()
    assert (work_dir / "threshold_policy.json").is_file()
    validation_errors: list[str] = []
    selected_again, payloads_again = phase13_validation._validate_calibration(
        work_dir, lock, settings, validation_errors
    )
    assert validation_errors == []
    assert set(selected_again) == {"T1", "T3"}
    assert set(payloads_again) == {"T1", "T3"}

    # Exercise the independent validator against the complete Stage A bundle,
    # while stubbing only the locked Phase 12 model replay boundary.
    source_targets = source[["warranty_claim_key", "target"]].drop_duplicates("warranty_claim_key")
    selected_for_validator = {
        track: frame.assign(
            target=frame["warranty_claim_key"].map(
                source_targets.set_index("warranty_claim_key")["target"]
            )
        )
        for track, frame in selected.items()
    }
    raw_summary = ensemble["summary"].rename(
        columns={
            "pooled_average_precision": "average_precision",
            "pooled_roc_auc": "roc_auc",
            "pooled_log_loss": "log_loss",
            "pooled_brier_score": "brier_score",
            "pooled_ece": "ece_10",
            "pooled_mce": "mce_10",
        }
    )
    validation_keys = np.arange(1, 31)
    validation_frame = pd.DataFrame({"warranty_claim_key": validation_keys, "split": "VALIDATION"})
    validation_targets = pd.DataFrame(
        {
            "warranty_claim_key": validation_keys,
            "target__high_cost_claim_flag": (validation_keys % 5 == 0).astype("int8"),
        }
    )
    phase10 = SimpleNamespace(
        development=validation_frame,
        feature_sets={"E1": object(), "E3": object()},
    )
    lock.phase12_inputs = SimpleNamespace(phase10_inputs=phase10)
    lock.phase12_dir = work_dir
    lock.effective_models = {
        track: {
            "candidate_id": f"P10_{track}",
            "model_file": f"{track.lower()}.cbm",
            "technical_threshold": "0.5",
            "feature_count": 3,
        }
        for track in ("T1", "T3")
    }
    monkeypatch.setattr(
        phase13_validation,
        "_validate_calibration",
        lambda _directory, _lock, _settings, _errors: (
            selected_for_validator,
            final_calibrators,
        ),
    )
    monkeypatch.setattr(phase13_validation, "load_phase12_lock", lambda *_args, **_kwargs: lock)
    monkeypatch.setattr(
        phase13_validation,
        "evaluate_ensemble_weights",
        lambda _aligned, weights, bins: (ensemble["predictions"], raw_summary.copy()),
    )
    monkeypatch.setattr(
        phase13_validation,
        "load_validation_targets_after_freeze",
        lambda _phase10, study_frozen: (
            validation_targets,
            {"target_rows": len(validation_targets)},
        ),
    )
    monkeypatch.setattr(phase13_validation, "load_baseline_settings", lambda _root: object())
    monkeypatch.setattr(
        phase13_validation, "adapt_matrix", lambda frame, _features, _settings: frame
    )
    monkeypatch.setattr(phase13_validation, "load_model", lambda _path: object())
    monkeypatch.setattr(
        phase13_validation,
        "predict_probabilities",
        lambda _model, matrix, _features: np.clip(
            matrix["warranty_claim_key"].to_numpy() / 35.0, 0.01, 0.99
        ),
    )
    validation_rows = []
    raw = validation_keys / 35.0
    for track in ("T1", "T3"):
        calibrated = apply_calibrator(final_calibrators[track], raw)
        validation_rows.extend(
            {
                "warranty_claim_key": int(key),
                "track": track,
                "candidate_id": f"P10_{track}",
                "raw_probability": float(raw_value),
                "calibrated_probability": float(calibrated_value),
                "effective_probability": float(raw_value),
            }
            for key, raw_value, calibrated_value in zip(
                validation_keys, raw, calibrated, strict=True
            )
        )
    pd.DataFrame(validation_rows).to_parquet(
        work_dir / "validation_predictions.parquet", index=False
    )
    (work_dir / "phase12_parent_resolution.json").write_text("{}", encoding="utf-8")
    (work_dir / "compute_manifest.json").write_text("{}", encoding="utf-8")
    audit = {
        "validation_target_rows_loaded_before_phase13_freeze": 0,
        "validation_target_rows_loaded_after_phase13_freeze": len(validation_targets),
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
    }
    (work_dir / "target_access_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    lock.run_id = "P12"
    expected_outer = phase13_validation._reconstruct_outer_validation(
        work_dir,
        lock,
        load_calibration_ensemble_settings(),
        final_calibrators,
        _read_json(work_dir / "threshold_policy.json"),
        _read_json(work_dir / "ensemble_selection.json"),
        Path.cwd(),
    )
    (work_dir / "validation_metrics.json").write_text(
        json.dumps(expected_outer["metrics"]), encoding="utf-8"
    )
    (work_dir / "effective_model_manifest.json").write_text(
        json.dumps(expected_outer["effective_manifest"]), encoding="utf-8"
    )
    freeze_without_hash = {
        "phase": 13,
        "phase13_run_id": "P13_TEST",
        "phase12_run_id": "P12",
        "selected_calibration": {
            track: {
                "method": _read_json(work_dir / "calibration_selection.json")["tracks"][track][
                    "selected_calibration_method"
                ],
                "calibrator_sha": final_calibrators[track]["calibrator_sha"],
            }
            for track in ("T1", "T3")
        },
        "calibration_fold_sha256": calibration["fold_sha"],
        "calibration_selection_evidence_sha256": phase13_runner._canonical_sha(
            _read_json(work_dir / "calibration_selection.json")["tracks"]
        ),
        "selected_ensemble_policy": _read_json(work_dir / "ensemble_selection.json")[
            "selected_policy"
        ],
        "ensemble_t1_weight": _read_json(work_dir / "ensemble_selection.json")["selected_weight"],
        "frozen_ensemble_components": {
            track: {
                "method": final_calibrators[track].get("method"),
                "calibrator_sha": final_calibrators[track].get("calibrator_sha"),
            }
            for track in ("T1", "T3")
        },
        "ensemble_evidence_sha256": phase13_runner._canonical_sha(
            {
                "summary": ensemble["summary"].to_dict("records"),
                "selection": _read_json(work_dir / "ensemble_selection.json"),
            }
        ),
        "calibrated_thresholds": _read_json(work_dir / "threshold_policy.json")["candidates"],
        "threshold_evidence_sha256": _read_json(work_dir / "threshold_policy.json")[
            "threshold_curve_sha256"
        ],
        "outer_validation_accessed": False,
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
    }
    freeze = {
        **freeze_without_hash,
        "phase13_freeze_content_sha256": phase13_runner._canonical_sha(freeze_without_hash),
    }
    (work_dir / "phase13_freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
    (work_dir / "phase13_manifest.json").write_text(
        json.dumps(
            {
                "phase": 13,
                "run_id": "P13_TEST",
                "phase12_dir": str(work_dir),
                "phase12_run_id": "P12",
                "configuration_sha256": locked_configuration_sha256(),
                "phase13_freeze_content_sha256": freeze["phase13_freeze_content_sha256"],
                "phase13_freeze_file_sha256": sha256(
                    (work_dir / "phase13_freeze.json").read_bytes()
                ).hexdigest(),
                "phase13_development_champion": expected_outer["champion"],
                "artifact_file_sha256": {},
                "test_target_rows_loaded": 0,
                "test_predictions_created": 0,
                "test_metrics_computed": False,
                "test_target_access_allowed": False,
                "first_allowed_test_target_phase": 15,
            }
        ),
        encoding="utf-8",
    )
    validation_result = validate_existing_phase13(work_dir, project_root=Path.cwd())
    assert validation_result["valid"] is True
    assert validation_result["hardening_status"] == "HARDENED_PASS"
    tampered = pd.read_parquet(work_dir / "validation_predictions.parquet")
    tampered.loc[0, "effective_probability"] = float(tampered.loc[0, "calibrated_probability"])
    tampered.to_parquet(work_dir / "validation_predictions.parquet", index=False)
    tampered_result = validate_existing_phase13(work_dir, project_root=Path.cwd())
    assert tampered_result["valid"] is False
    assert any("outer validation predictions" in error for error in tampered_result["errors"])


def test_calibration_stage_reuses_valid_checkpoints_and_refits_corruption(
    monkeypatch, tmp_path: Path
) -> None:
    source = _large_source_oof()
    settings = load_calibration_ensemble_settings()
    lock = SimpleNamespace(
        source_oof=source, train_targets=source.drop_duplicates("warranty_claim_key")
    )
    work_dir = tmp_path / "phase13-resume"
    work_dir.mkdir()
    first = _calibration_stage(lock, settings, work_dir, resume=False)
    assert first["execution"]["jobs_refit"] == 12
    calls: list[str] = []
    original = phase13_runner.fit_calibrator

    def counted_fit(*args, **kwargs):
        calls.append(str(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(phase13_runner, "fit_calibrator", counted_fit)
    resumed = _calibration_stage(lock, settings, work_dir, resume=True)
    assert calls == []
    assert resumed["execution"]["jobs_reused_from_checkpoint"] == 12
    assert resumed["execution"]["jobs_refit"] == 0

    corrupt = work_dir / "checkpoints" / "T1_C1_SIGMOID_C1.json"
    corrupt.write_text("{", encoding="utf-8")
    repaired = _calibration_stage(lock, settings, work_dir, resume=True)
    assert len(calls) == 1
    assert repaired["execution"]["jobs_reused_from_checkpoint"] == 11
    assert repaired["execution"]["jobs_refit"] == 1


def test_validation_stage_replays_frozen_models_and_fails_closed(
    monkeypatch, tmp_path: Path
) -> None:
    settings = load_calibration_ensemble_settings()
    keys = np.arange(1, 31)
    validation_frame = pd.DataFrame({"warranty_claim_key": keys, "split": "VALIDATION"})
    validation_targets = pd.DataFrame(
        {"warranty_claim_key": keys, "target__high_cost_claim_flag": (keys % 5 == 0).astype("int8")}
    )
    phase10 = SimpleNamespace(
        development=validation_frame,
        feature_sets={"E1": object(), "E3": object()},
    )
    lock = SimpleNamespace(
        root=tmp_path,
        phase12_dir=tmp_path,
        phase12_inputs=SimpleNamespace(phase10_inputs=phase10),
        effective_models={
            "T1": {
                "model_file": "t1.cbm",
                "technical_threshold": "0.5",
                "candidate_id": "P10_T1",
                "feature_count": 3,
            },
            "T3": {
                "model_file": "t3.cbm",
                "technical_threshold": "0.5",
                "candidate_id": "P10_T3",
                "feature_count": 4,
            },
        },
    )
    monkeypatch.setattr(
        phase13_runner,
        "load_validation_targets_after_freeze",
        lambda _phase10, study_frozen: (
            validation_targets,
            {"target_rows": len(validation_targets)},
        ),
    )
    monkeypatch.setattr(phase13_runner, "load_baseline_settings", lambda _root: object())
    monkeypatch.setattr(phase13_runner, "adapt_matrix", lambda frame, _features, _settings: frame)
    monkeypatch.setattr(phase13_runner, "load_model", lambda _path: object())
    monkeypatch.setattr(
        phase13_runner,
        "predict_probabilities",
        lambda _model, matrix, _features: np.clip(
            matrix["warranty_claim_key"].to_numpy() / 35.0, 0.01, 0.99
        ),
    )
    rejected_calibrator = fit_calibrator(
        "C1_SIGMOID",
        np.linspace(0.01, 0.99, len(keys)),
        (keys % 5 == 0).astype("int8"),
    )
    monkeypatch.setattr(
        phase13_runner,
        "accept_track_calibration",
        lambda _raw, _calibrated, _settings: {
            "accepted": False,
            "reason": "CALIBRATION_REJECTED_ON_VALIDATION",
            "effective": "RAW_PHASE12",
        },
    )
    ensemble = {"selection": {"selected_policy": "TRUE_BLEND", "selected_weight": 0.5}}
    thresholds = {
        "policy": {
            "candidates": {
                "T1": {"threshold": 0.5},
                "T3": {"threshold": 0.5},
                "ENSEMBLE": {"threshold": 0.5},
            }
        }
    }
    result = _validation_stage(
        lock,
        settings,
        selected={},
        final_calibrators={"T1": rejected_calibrator, "T3": rejected_calibrator},
        calibration={},
        ensemble=ensemble,
        thresholds=thresholds,
        work_dir=tmp_path,
    )
    assert result["champion"] in {"P10_T1", "P10_T3"}
    assert result["warnings"] == [
        "CALIBRATION_REJECTED_ON_VALIDATION",
        "ENSEMBLE_REJECTED_ON_VALIDATION",
    ]
    assert result["ensemble_candidate"] is None
    assert result["ensemble_acceptance"]["reason"] == "COMPONENT_CALIBRATION_REJECTED"
    assert len(result["predictions"]) == 60
    assert (tmp_path / "validation_metrics.json").is_file()


def test_phase12_lock_resolution_and_validation_helpers(monkeypatch, tmp_path: Path) -> None:
    phase12 = tmp_path / "phase12"
    phase12.mkdir()
    full_source = _large_source_oof()
    source = full_source.drop(columns=["claim_date", "target"])
    source["strategy_id"] = "S0_NONE"
    source["fold_id"] = source["fold_id"].astype("int8")
    source = source.drop_duplicates(["warranty_claim_key", "track"])
    assignment = pd.DataFrame(
        {
            "warranty_claim_key": np.arange(1, 271),
            "fold_id": np.repeat([1, 2, 3], 90),
            "claim_date": np.repeat(["2024-01-01", "2024-02-01", "2024-03-01"], 90),
            "role": "VALIDATION",
        }
    )
    source.to_parquet(phase12 / "strategy_oof_predictions.parquet", index=False)
    for name in (
        "model_manifest.json",
        "threshold_policy.json",
        "strategy_summary.parquet",
    ):
        path = phase12 / name
        if path.suffix == ".parquet":
            pd.DataFrame().to_parquet(path, index=False)
        else:
            path.write_text("{}", encoding="utf-8")
    models = {}
    for track in ("T1", "T3"):
        model_path = phase12 / f"{track.lower()}.cbm"
        model_path.write_bytes(track.encode())
        models[track] = {
            "track": track,
            "candidate_id": f"P10_{track}",
            "model_file": model_path.name,
            "model_sha256": sha256(model_path.read_bytes()).hexdigest(),
            "feature_set_sha256": "feature-set",
            "feature_list_sha256": "feature-list",
            "selected_imbalance_strategy": "S0_NONE",
        }
    (phase12 / "effective_model_manifest.json").write_text(
        json.dumps({"models": list(models.values())}), encoding="utf-8"
    )
    (phase12 / "phase12_manifest.json").write_text(
        json.dumps(
            {
                "phase": 12,
                "run_id": "P12",
                "phase11_run_id": "P11",
                "test_target_rows_loaded": 0,
                "test_predictions_created": 0,
                "test_metrics_computed": False,
                "first_allowed_test_target_phase": 15,
            }
        ),
        encoding="utf-8",
    )
    (phase12 / "phase11_parent_resolution.json").write_text(
        json.dumps({"phase11_run_id": "P11"}), encoding="utf-8"
    )
    (phase12 / "validation.json").write_text(
        json.dumps({"valid": True, "hardening_status": "HARDENED_PASS"}), encoding="utf-8"
    )
    (phase12 / "phase12_freeze.json").write_text(
        json.dumps({"outer_validation_accessed": False}), encoding="utf-8"
    )
    (phase12 / "target_access_audit.json").write_text(
        json.dumps(
            {
                "test_target_rows_loaded": 0,
                "test_predictions_created": 0,
                "test_metrics_computed": False,
                "test_target_access_allowed": False,
                "first_allowed_test_target_phase": 15,
            }
        ),
        encoding="utf-8",
    )
    train_targets = (
        full_source[["warranty_claim_key", "target"]]
        .drop_duplicates("warranty_claim_key")
        .rename(columns={"target": "target__high_cost_claim_flag"})
    )
    phase10_inputs = object()
    phase12_inputs = SimpleNamespace(
        phase10_inputs=phase10_inputs,
        fold_plan=SimpleNamespace(assignments=assignment),
    )
    monkeypatch.setattr(
        phase13_input,
        "validate_existing_phase12",
        lambda *_args, **_kwargs: {"valid": True, "hardening_status": "HARDENED_PASS"},
    )
    monkeypatch.setattr(
        phase13_input,
        "load_locked_phase11_inputs",
        lambda *_args, **_kwargs: phase12_inputs,
    )
    monkeypatch.setattr(
        phase13_input,
        "load_train_targets_for_optimization",
        lambda _inputs: (train_targets, None),
    )
    lock = phase13_input.load_phase12_lock(phase12, project_root=Path.cwd())
    parent = phase13_input.write_phase12_parent_resolution(tmp_path / "parent.json", lock)
    assert lock.run_id == "P12"
    assert parent["tracks"]["T1"]["effective_candidate_id"] == "P10_T1"
    with pytest.raises(ValueError):
        phase13_input._phase12_test_seal({"test_target_rows_loaded": 1}, "seal")
    assert (
        phase13_input._effective_models({"models": list(models.values())})["T3"]["candidate_id"]
        == "P10_T3"
    )

    assert phase13_validation._close(float("inf"), float("inf")) is True
    assert phase13_validation._close("x", "x") is True
    frame = pd.DataFrame({"a": [1, 2], "b": [0.1, 0.2]})
    assert phase13_validation._compare_frame(frame, frame.copy(), ["a", "b"], label="frame") == []
    bad = frame.copy()
    bad.loc[0, "b"] = 0.9
    assert phase13_validation._compare_frame(frame, bad, ["a", "b"], label="frame")
    assert (
        phase13_validation._validate_test_seal(
            {
                "test_target_rows_loaded": 0,
                "test_predictions_created": 0,
                "test_metrics_computed": False,
                "test_target_access_allowed": False,
                "first_allowed_test_target_phase": 15,
            },
            "seal",
        )
        == []
    )


def test_selection_guardrails_and_claim_free_reports(tmp_path: Path) -> None:
    settings = load_calibration_ensemble_settings()
    raw = {"average_precision": 0.4, "roc_auc": 0.7, "log_loss": 0.4, "brier_score": 0.2}
    accepted = {
        "calibration_method": "C1_SIGMOID",
        "average_precision": 0.4,
        "roc_auc": 0.7,
        "log_loss": 0.35,
        "brier_score": 0.19,
    }
    rejected = {**accepted, "average_precision": 0.1, "log_loss": 0.8, "brier_score": 0.7}
    assert accept_track_calibration(raw, accepted, settings)["accepted"] is True
    assert accept_track_calibration(raw, rejected, settings)["accepted"] is False
    assert (
        accept_track_calibration(raw, {"calibration_method": "C0_NONE", **raw}, settings)["reason"]
        == "NONE"
    )
    best = {"average_precision": 0.4, "roc_auc": 0.7, "log_loss": 0.4, "brier_score": 0.2}
    assert (
        accept_ensemble(
            {"average_precision": 0.41, "roc_auc": 0.7, "log_loss": 0.39, "brier_score": 0.19},
            best,
            settings,
        )["accepted"]
        is True
    )
    assert (
        accept_ensemble(
            {"average_precision": 0.1, "roc_auc": 0.2, "log_loss": 0.8, "brier_score": 0.7},
            best,
            settings,
        )["accepted"]
        is False
    )
    report_dir = write_phase13_reports(
        tmp_path / "reports",
        "run",
        {
            "validation": {"hardening_status": "HARDENED_PASS"},
            "validation_metrics": {"phase13_development_champion": "P13_T1"},
            "calibration_summary": [{"score": np.float64(0.5)}],
            "ensemble_selection": {"selected_policy": "BEST_SINGLE"},
            "parent_resolution": {"phase12_run_id": "p12"},
            "threshold_policy": {"objective": "MCC"},
        },
    )
    assert (report_dir / "phase_13_summary.md").is_file()
    assert (report_dir / "validation.json").is_file()


def test_runner_provenance_helpers_and_fail_closed_start(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text('{"value": 1}', encoding="utf-8")
    assert _read_json(payload_path)["value"] == 1
    with pytest.raises(CalibrationEnsembleError):
        _read_json(tmp_path / "missing.json")
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("[]", encoding="utf-8")
    with pytest.raises(CalibrationEnsembleError):
        _read_json(invalid_json)
    frame = pd.DataFrame({"key": [2, 1], "score": [0.2, 0.1]})
    assert _frame_sha(frame) == _frame_sha(frame, ["key", "score"])
    metric = _metric_row(
        track="T1",
        method="C0_NONE",
        calibration_fold="C1",
        role="CALIBRATION_VALIDATION",
        metrics={"average_precision": 0.2},
        eligible=True,
        eligibility_reason="ELIGIBLE",
    )
    assert metric["track"] == "T1" and metric["average_precision"] == 0.2
    (tmp_path / "phase13_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "validation.json").write_text("{}", encoding="utf-8")
    (tmp_path / "nested.txt").write_text("artifact", encoding="utf-8")
    artifact_names = set(_artifact_hashes(tmp_path))
    assert {"nested.txt", "payload.json", "invalid.json"}.issubset(artifact_names)
    assert "phase13_manifest.json" not in artifact_names
    assert "validation.json" not in artifact_names
    assert phase13_contract_check()["valid"] is True
    with pytest.raises(CalibrationEnsembleError):
        build_phase13(tmp_path / "missing-phase12", project_root=Path.cwd())

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(CalibrationEnsembleError):
        phase13_input._read_json(malformed)
    non_object = tmp_path / "array.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(CalibrationEnsembleError):
        phase13_input._read_json(non_object)
    assert phase13_input.current_git_commit(tmp_path) == "unknown"
    for invalid in ({}, {"models": []}, {"models": [{"track": "T2"}]}):
        with pytest.raises(CalibrationEnsembleError):
            phase13_input._effective_models(invalid)
    with pytest.raises(CalibrationEnsembleError):
        phase13_input._effective_models({"models": [{"track": "T1"}, {"track": "T1"}]})
    with pytest.raises(CalibrationEnsembleError):
        phase13_input._source_oof(tmp_path, SimpleNamespace(), {}, pd.DataFrame())
    bad_oof = tmp_path / "strategy_oof_predictions.parquet"
    pd.DataFrame({"bad": [1]}).to_parquet(bad_oof, index=False)
    with pytest.raises(CalibrationEnsembleError):
        phase13_input._source_oof(tmp_path, SimpleNamespace(), {}, pd.DataFrame())

    assert phase13_validation._validate_artifact_hashes(tmp_path, {}) == []
    assert phase13_validation._validate_artifact_hashes(
        tmp_path, {"artifact_file_sha256": {"missing.bin": "sha"}}
    )
    assert phase13_validation._validate_artifact_hashes(
        tmp_path, {"artifact_file_sha256": {"nested.txt": "wrong"}}
    )
    assert phase13_validation._close("not-a-number", "not-a-number") is True
    assert phase13_validation._compare_frame(frame, frame.iloc[:1], ["key", "score"], label="frame")
    assert phase13_validation._compare_frame(frame, frame, ["wrong"], label="frame")
    settings = load_calibration_ensemble_settings()
    assert settings_payload(settings)["configuration_sha256"] == locked_configuration_sha256()
    assert phase13_validation._compare_payload({"a": [1]}, {"a": [2]}, label="payload")
    assert phase13_validation._compare_payload({"a": 1}, {"b": 1}, label="payload")
    assert phase13_validation._compare_payload([1], [1, 2], label="payload")
    assert phase13_validation._compare_payload(True, 1, label="payload")


def test_checkpoint_planner_and_validator_fail_closed(tmp_path: Path) -> None:
    settings = load_calibration_ensemble_settings()
    plan = build_compute_plan(
        settings, logical_processors=3, max_workers=99, catboost_replay_threads=99
    )
    assert plan.effective_cpu_budget == 1
    assert plan.calibration_worker_count == 1
    checkpoint = write_calibration_checkpoint(
        tmp_path,
        track="T1",
        calibration_method="C0_NONE",
        calibration_fold="C1",
        training_input_sha="train",
        validation_input_sha="validation",
        calibrator_sha="cal",
        metrics={"average_precision": 0.1},
        prediction_sha="pred",
    )
    assert checkpoint.is_file()
    assert (
        load_valid_calibration_checkpoint(
            tmp_path,
            track="T1",
            calibration_method="C0_NONE",
            calibration_fold="C1",
            training_input_sha="train",
            validation_input_sha="validation",
        )
        is not None
    )
    checkpoint.write_text("{", encoding="utf-8")
    assert (
        load_valid_calibration_checkpoint(
            tmp_path,
            track="T1",
            calibration_method="C0_NONE",
            calibration_fold="C1",
            training_input_sha="train",
            validation_input_sha="validation",
        )
        is None
    )
    assert phase13_plan_check(tmp_path / "missing-phase12")["valid"] is False
    result = validate_existing_phase13(tmp_path / "missing-phase13")
    assert result["valid"] is False
    assert result["hardening_status"] == "BLOCKED"


def test_checkpoint_loader_rejects_each_stale_or_malformed_payload(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    path = checkpoint_dir / "T1_C0_NONE_C1.json"
    base = {
        "track": "T1",
        "calibration_method": "C0_NONE",
        "calibration_fold": "C1",
        "training_input_sha": "train",
        "validation_input_sha": "validation",
        "calibrator_sha": "cal",
        "metrics": {"average_precision": 0.1},
        "prediction_sha": "pred",
    }

    def write_payload(payload: dict[str, object], declared: str | None = None) -> None:
        checkpoint = dict(payload)
        checkpoint["checkpoint_sha"] = declared or phase13_checkpoint._sha(payload)
        path.write_text(json.dumps(checkpoint), encoding="utf-8")

    kwargs = {
        "work_dir": tmp_path,
        "track": "T1",
        "calibration_method": "C0_NONE",
        "calibration_fold": "C1",
        "training_input_sha": "train",
        "validation_input_sha": "validation",
    }
    assert load_valid_calibration_checkpoint(**kwargs | {"work_dir": tmp_path / "missing"}) is None
    invalid_declared = dict(base)
    invalid_declared["checkpoint_sha"] = 1
    path.write_text(json.dumps(invalid_declared), encoding="utf-8")
    assert load_valid_calibration_checkpoint(**kwargs) is None

    for field, value in (
        ("track", "T3"),
        ("calibration_fold", "C2"),
        ("training_input_sha", "stale-train"),
        ("validation_input_sha", "stale-validation"),
    ):
        stale = dict(base)
        stale[field] = value
        write_payload(stale)
        assert load_valid_calibration_checkpoint(**kwargs) is None

    write_payload(base, declared="wrong")
    assert load_valid_calibration_checkpoint(**kwargs) is None
    malformed_calibrator = dict(base)
    malformed_calibrator["calibrator"] = []
    write_payload(malformed_calibrator)
    assert load_valid_calibration_checkpoint(**kwargs) is None
    mismatched_calibrator = dict(base)
    mismatched_calibrator["calibrator"] = {"calibrator_sha": "different"}
    write_payload(mismatched_calibrator)
    assert load_valid_calibration_checkpoint(**kwargs) is None
