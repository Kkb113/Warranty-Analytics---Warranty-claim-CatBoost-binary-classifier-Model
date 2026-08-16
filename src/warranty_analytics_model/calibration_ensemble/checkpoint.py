"""Immutable, hash-checked Phase 13 checkpoint helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def write_calibration_checkpoint(
    work_dir: Path,
    *,
    track: str,
    calibration_method: str,
    calibration_fold: str,
    training_input_sha: str,
    validation_input_sha: str,
    calibrator_sha: str,
    metrics: dict[str, Any],
    prediction_sha: str,
) -> Path:
    directory = work_dir / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "track": track,
        "calibration_method": calibration_method,
        "calibration_fold": calibration_fold,
        "training_input_sha": training_input_sha,
        "validation_input_sha": validation_input_sha,
        "calibrator_sha": calibrator_sha,
        "metrics": metrics,
        "prediction_sha": prediction_sha,
    }
    payload["checkpoint_sha"] = _sha(payload)
    path = directory / f"{track}_{calibration_method}_{calibration_fold}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_valid_calibration_checkpoint(
    work_dir: Path,
    *,
    track: str,
    calibration_method: str,
    calibration_fold: str,
    training_input_sha: str,
    validation_input_sha: str,
) -> dict[str, Any] | None:
    path = work_dir / "checkpoints" / f"{track}_{calibration_method}_{calibration_fold}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        declared = payload.pop("checkpoint_sha")
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if payload.get("track") != track or payload.get("calibration_method") != calibration_method:
        return None
    if payload.get("calibration_fold") != calibration_fold:
        return None
    if payload.get("training_input_sha") != training_input_sha:
        return None
    if payload.get("validation_input_sha") != validation_input_sha:
        return None
    if declared != _sha(payload):
        return None
    payload["checkpoint_sha"] = declared
    return dict(payload)


__all__ = ["load_valid_calibration_checkpoint", "write_calibration_checkpoint"]
