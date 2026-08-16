"""Atomic per-experiment/per-fold checkpoints with stale-work rejection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def checkpoint_path(work_dir: Path, experiment_id: str, fold_id: int) -> Path:
    return work_dir / "checkpoints" / experiment_id / f"fold_{fold_id}.json"


def write_checkpoint(work_dir: Path, payload: dict[str, Any]) -> Path:
    path = checkpoint_path(work_dir, str(payload["experiment_id"]), int(payload["fold_id"]))
    _atomic_json(path, payload)
    return path


def load_valid_checkpoint(
    work_dir: Path,
    *,
    experiment_id: str,
    experiment_spec_sha256: str,
    track: str,
    feature_set_sha256: str,
    parameter_sha256: str,
    fold_id: int,
) -> dict[str, Any] | None:
    path = checkpoint_path(work_dir, experiment_id, fold_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Corrupt Phase 11 checkpoint: {path}") from exc
    expected = {
        "experiment_id": experiment_id,
        "experiment_spec_sha256": experiment_spec_sha256,
        "track": track,
        "feature_set_sha256": feature_set_sha256,
        "parameter_sha256": parameter_sha256,
        "fold_id": fold_id,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Stale Phase 11 checkpoint rejected: {path}")
    if not isinstance(payload.get("metrics"), dict) or payload.get("completed_at") is None:
        raise ValueError(f"Incomplete Phase 11 checkpoint rejected: {path}")
    return cast(dict[str, Any], payload)

