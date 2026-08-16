"""Phase 10 hashes, runtime provenance, and policy checks."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pandas as pd

from ..baseline_model.provenance import runtime_provenance as phase9_runtime_provenance
from ..feature_mart.manifest import content_sha256, sha256_file
from .models import OptimizationError


def runtime_provenance() -> dict[str, str]:
    """Extend Phase 9 runtime metadata with the optimizer version."""

    payload = dict(phase9_runtime_provenance(include_optimization=True))
    if payload.get("optuna_version") is None:
        try:
            payload["optuna_version"] = version("optuna")
        except PackageNotFoundError:
            payload["optuna_version"] = "unavailable"
    return payload


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def inner_membership_sha256(keys: Any) -> str:
    values = sorted(int(value) for value in keys)
    return canonical_json_sha256(values)


def fold_content_sha256(frame: pd.DataFrame) -> str:
    required = ["warranty_claim_key", "claim_date", "fold_id", "role"]
    if list(frame.columns) != required:
        raise OptimizationError("Inner fold hashing requires the exact persisted fold schema.")
    ordered = frame.sort_values(["fold_id", "role", "warranty_claim_key"], kind="mergesort")
    records = [
        [int(key), str(date), int(fold), str(role)]
        for key, date, fold, role in ordered.itertuples(index=False, name=None)
    ]
    return canonical_json_sha256({"columns": required, "rows": records})


def prediction_sha256(frame: pd.DataFrame) -> str:
    required = ["warranty_claim_key", "candidate_id", "high_cost_probability"]
    if list(frame.columns) != required:
        raise OptimizationError("Phase 10 prediction hashing requires the exact schema.")
    ordered = frame.sort_values(["candidate_id", "warranty_claim_key"], kind="mergesort")
    probabilities = pd.to_numeric(ordered["high_cost_probability"], errors="coerce").to_numpy(
        dtype="float64"
    )
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise OptimizationError("Phase 10 predictions must be finite probabilities in [0, 1].")
    rows = [
        [int(key), str(candidate), format(float(probability), ".17g")]
        for key, candidate, probability in ordered.itertuples(index=False, name=None)
    ]
    return canonical_json_sha256({"columns": required, "rows": rows})


__all__ = [
    "canonical_json_sha256",
    "content_sha256",
    "fold_content_sha256",
    "inner_membership_sha256",
    "prediction_sha256",
    "runtime_provenance",
    "sha256_file",
]
