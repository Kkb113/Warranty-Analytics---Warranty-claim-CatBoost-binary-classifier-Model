"""Atomic Phase 12 fold checkpoints and strict resume validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

from ..catboost_optimization.provenance import canonical_json_sha256


def checkpoint_path(work_dir: Path, track: str, strategy_id: str, fold_id: int) -> Path:
    return work_dir / "checkpoints" / track / strategy_id / f"fold_{int(fold_id)}.json"


def write_checkpoint(
    work_dir: Path,
    *,
    track: str,
    strategy_id: str,
    fold_id: int,
    feature_set_sha256: str,
    parent_parameter_sha256: str,
    strategy_parameter_sha256: str,
    fold_membership_sha256: str,
    metrics: dict[str, Any],
    prediction_sha256: str,
    training_seconds: float,
    prediction_keys: list[int] | None = None,
    prediction_values: list[float] | None = None,
) -> Path:
    path = checkpoint_path(work_dir, track, strategy_id, fold_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "track": track,
        "strategy_id": strategy_id,
        "fold_id": int(fold_id),
        "feature_set_sha256": feature_set_sha256,
        "parent_parameter_sha256": parent_parameter_sha256,
        "strategy_parameter_sha256": strategy_parameter_sha256,
        "fold_membership_sha256": fold_membership_sha256,
        "metrics": metrics,
        "prediction_sha256": prediction_sha256,
        "training_seconds": float(training_seconds),
        "completed": True,
    }
    if prediction_keys is not None and prediction_values is not None:
        if len(prediction_keys) != len(prediction_values):
            raise ValueError("Checkpoint prediction keys and values must have equal length.")
        payload["prediction_keys"] = [int(value) for value in prediction_keys]
        payload["prediction_values"] = [float(value) for value in prediction_values]
    payload["checkpoint_sha256"] = canonical_json_sha256(payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def load_valid_checkpoint(
    work_dir: Path,
    *,
    track: str,
    strategy_id: str,
    fold_id: int,
    feature_set_sha256: str,
    parent_parameter_sha256: str,
    strategy_parameter_sha256: str,
    fold_membership_sha256: str,
) -> dict[str, Any] | None:
    path = checkpoint_path(work_dir, track, strategy_id, fold_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_hash = payload.pop("checkpoint_sha256")
        if canonical_json_sha256(payload) != expected_hash:
            return None
        if not payload.get("completed"):
            return None
        expected = {
            "track": track,
            "strategy_id": strategy_id,
            "fold_id": int(fold_id),
            "feature_set_sha256": feature_set_sha256,
            "parent_parameter_sha256": parent_parameter_sha256,
            "strategy_parameter_sha256": strategy_parameter_sha256,
            "fold_membership_sha256": fold_membership_sha256,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            return None
        return cast(dict[str, Any], payload)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


__all__ = ["checkpoint_path", "load_valid_checkpoint", "write_checkpoint"]
