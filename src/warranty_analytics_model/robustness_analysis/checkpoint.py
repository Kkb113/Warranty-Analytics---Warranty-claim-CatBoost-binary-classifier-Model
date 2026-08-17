"""Atomic, provenance-bound Phase 14 checkpoint helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def checkpoint_sha(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def write_checkpoint(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body["checkpoint_sha256"] = checkpoint_sha(
        {key: value for key, value in body.items() if key != "checkpoint_sha256"}
    )
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(body, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    with temporary.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return body


def load_checkpoint(path: Path, expected_bindings: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    declared = payload.get("checkpoint_sha256")
    actual = checkpoint_sha(
        {key: value for key, value in payload.items() if key != "checkpoint_sha256"}
    )
    if declared != actual or any(
        payload.get(key) != value for key, value in expected_bindings.items()
    ):
        return None
    return payload


__all__ = ["checkpoint_sha", "load_checkpoint", "write_checkpoint"]
