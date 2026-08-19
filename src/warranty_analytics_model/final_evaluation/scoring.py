"""Frozen serving-policy scoring for the authoritative TEST population."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..robustness_analysis.input import KEY, prepare_scorer
from .input import Phase15InputError, Phase15Resolved


def score_test_in_batches(
    resolved: Phase15Resolved,
    frame: pd.DataFrame,
    *,
    inference_threads: int,
    batch_size: int = 4096,
) -> pd.DataFrame:
    scorer = prepare_scorer(resolved.phase13, threads=int(inference_threads))
    pieces = [
        scorer(frame.iloc[start : start + int(batch_size)])
        for start in range(0, len(frame), int(batch_size))
    ]
    if not pieces:
        raise Phase15InputError("Cannot score an empty TEST feature frame.")
    scored = pd.concat(pieces, ignore_index=True)
    if KEY not in scored.columns or scored[KEY].duplicated().any():
        raise Phase15InputError("Frozen serving policy did not return one score per TEST claim.")
    expected = set(frame[KEY].astype(int))
    if set(scored[KEY].astype(int)) != expected or len(scored) != len(frame):
        raise Phase15InputError("Frozen serving policy changed TEST population membership.")
    if "probability" not in scored.columns:
        raise Phase15InputError("Frozen serving policy did not return probability.")
    probabilities = pd.to_numeric(scored["probability"], errors="coerce")
    if probabilities.isna().any() or not np.isfinite(probabilities.to_numpy()).all():
        raise Phase15InputError("TEST probabilities contain non-finite values.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise Phase15InputError("TEST probabilities are outside [0, 1].")
    scored["probability"] = probabilities.to_numpy(dtype="float64")
    result = scored.sort_values(KEY, kind="mergesort").reset_index(drop=True)
    result["frozen_threshold"] = float(resolved.threshold)
    result["predicted_class"] = (result["probability"] >= float(resolved.threshold)).astype("int8")
    return result


def build_final_model_policy(resolved: Phase15Resolved) -> dict[str, Any]:
    return {
        "phase": 15,
        "policy": "REUSE_FROZEN_PHASE14_CHAMPION",
        "model_retraining": False,
        "train_validation_refit": False,
        "alternative_candidates_evaluated": False,
        "phase14_run_id": resolved.phase14_manifest.get("run_id"),
        "phase13_run_id": resolved.phase13.phase13_manifest.get("run_id"),
        "champion_id": resolved.champion_id,
        "candidate_type": resolved.champion_type,
        "score_space": resolved.score_space,
        "frozen_threshold": float(resolved.threshold),
        "model_sha256": {item.track: item.model_sha256 for item in resolved.components},
        "calibrator_sha256": {item.track: item.calibrator_sha256 for item in resolved.components},
        "feature_list_sha256": {
            item.track: item.feature_list_sha256 for item in resolved.components
        },
        "feature_schema_sha256": resolved.feature_schema_sha256,
        "test_policy_flags": {
            "model_selection": False,
            "threshold_tuning": False,
            "calibration_tuning": False,
            "ensemble_tuning": False,
            "feature_selection": False,
            "class_weight_tuning": False,
        },
    }


__all__ = ["build_final_model_policy", "score_test_in_batches"]
