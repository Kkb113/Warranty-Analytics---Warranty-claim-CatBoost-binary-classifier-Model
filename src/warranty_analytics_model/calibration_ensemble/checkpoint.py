"""Immutable, hash-checked Phase 13 checkpoint helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
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
    calibrator: dict[str, Any] | None = None,
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
    if calibrator is not None:
        payload["calibrator"] = calibrator
    payload["checkpoint_sha"] = _sha(payload)
    path = directory / f"{track}_{calibration_method}_{calibration_fold}.json"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
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
        declared = payload.get("checkpoint_sha")
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(payload, dict) or not isinstance(declared, str):
        return None
    body = {key: value for key, value in payload.items() if key != "checkpoint_sha"}
    if payload.get("track") != track or payload.get("calibration_method") != calibration_method:
        return None
    if payload.get("calibration_fold") != calibration_fold:
        return None
    if payload.get("training_input_sha") != training_input_sha:
        return None
    if payload.get("validation_input_sha") != validation_input_sha:
        return None
    if declared != _sha(body):
        return None
    calibrator = body.get("calibrator")
    if calibrator is not None and not isinstance(calibrator, dict):
        return None
    if isinstance(calibrator, dict) and calibrator.get("calibrator_sha") != body.get(
        "calibrator_sha"
    ):
        return None
    return {**body, "checkpoint_sha": declared}


__all__ = ["load_valid_calibration_checkpoint", "write_calibration_checkpoint"]
