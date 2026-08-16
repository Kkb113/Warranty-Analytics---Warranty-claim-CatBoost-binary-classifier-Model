"""Phase 10 artifact metadata and canonical manifest helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..feature_mart.manifest import sha256_file, write_json, write_parquet
from .provenance import canonical_json_sha256


def write_table(frame: pd.DataFrame, path: Path, compression: str) -> dict[str, Any]:
    return dict(write_parquet(frame, path, compression=compression))


def freeze_payload_sha256(payload: dict[str, Any]) -> str:
    content = {key: value for key, value in payload.items() if key != "study_freeze_sha256"}
    return canonical_json_sha256(content)


def artifact_hashes(directory: Path, names: list[str] | tuple[str, ...]) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in names}


__all__ = ["artifact_hashes", "freeze_payload_sha256", "sha256_file", "write_json", "write_table"]
