"""Phase 10 hashes, runtime provenance, and policy checks."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..baseline_model.provenance import runtime_provenance as phase9_runtime_provenance
from ..feature_mart.manifest import content_sha256, sha256_file, write_json
from .models import OptimizationError

ACCEPTANCE_OVERLAY_FILENAME = "phase10_acceptance_overlay.json"
"""Stable filename for post-run Phase 10 provenance acceptance evidence."""

_V2_MANIFEST_FIELDS = (
    "contract_version",
    "contract_checksum",
    "contract_policy_snapshot",
    "phase9_target_hashes",
    "phase9_feature_set_hashes",
    "trials_per_track",
    "objective_metric",
)


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OptimizationError(f"Expected a JSON object: {path}")
    return payload


def build_acceptance_overlay(
    optimization_dir: Path,
    *,
    validation_result: dict[str, Any],
    validator_commit_sha: str = "unknown",
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build immutable evidence for hardening an already-computed Phase 10 run.

    The overlay deliberately keeps the current manifest separate from any
    pre-v2 manifest. If the original manifest was not preserved, the overlay
    records that fact and leaves its hash null instead of treating a later
    v2 rewrite as original provenance.
    """

    directory = optimization_dir.expanduser().resolve()
    manifest_path = directory / "optimization_manifest.json"
    if not manifest_path.is_file():
        raise OptimizationError(f"Phase 10 optimization manifest is missing: {manifest_path}")
    manifest = _read_object(manifest_path)
    validation_path = directory / "validation.json"
    legacy_candidates = (
        directory / "optimization_manifest.legacy.json",
        directory / "optimization_manifest.pre_v2.json",
        directory / "optimization_manifest.original.json",
    )
    preserved_legacy = next((path for path in legacy_candidates if path.is_file()), None)
    v2_fields_present = [field for field in _V2_MANIFEST_FIELDS if field in manifest]
    artifact_hashes = manifest.get("artifact_file_sha256", {})
    if not isinstance(artifact_hashes, dict):
        artifact_hashes = {}
    model_hashes: dict[str, str] = {}
    model_manifest_path = directory / "model_manifest.json"
    if model_manifest_path.is_file():
        model_manifest = _read_object(model_manifest_path)
        for candidate_id, entry in model_manifest.get("models", {}).items():
            if isinstance(entry, dict) and entry.get("model_sha256"):
                model_hashes[str(candidate_id)] = str(entry["model_sha256"])

    legacy_evidence: dict[str, Any]
    if preserved_legacy is None:
        legacy_evidence = {
            "preserved": False,
            "path": None,
            "sha256": None,
            "status": "UNAVAILABLE",
            "note": (
                "No pre-v2 optimization manifest copy was present when this overlay was "
                "created. The current v2 manifest is recorded as post-hardening evidence "
                "only; its hash is not claimed as the original manifest hash."
            ),
        }
    else:
        legacy_evidence = {
            "preserved": True,
            "path": preserved_legacy.name,
            "sha256": sha256_file(preserved_legacy),
            "status": "PRESERVED",
            "note": "The pre-v2 manifest copy was retained beside the run bundle.",
        }

    validation_hash = sha256_file(validation_path) if validation_path.is_file() else None
    test_seal = {
        key: manifest.get(key)
        for key in (
            "test_target_rows_loaded",
            "test_predictions_created",
            "test_metrics_computed",
            "test_target_access_allowed",
            "first_allowed_test_target_phase",
        )
    }
    return {
        "overlay_version": "phase10_acceptance_overlay_v1",
        "run_id": manifest.get("run_id"),
        "created_at_utc": created_at_utc or datetime.now(UTC).isoformat(),
        "optimization_run": {
            "git_commit_sha": manifest.get("git_commit_sha"),
            "manifest_created_at_utc": manifest.get("created_at_utc"),
            "phase9_run_id": manifest.get("phase9_run_id"),
        },
        "source_manifest": {
            "path": "optimization_manifest.json",
            "sha256": sha256_file(manifest_path),
            "contract_version": manifest.get("contract_version"),
            "contract_checksum": manifest.get("contract_checksum"),
            "v2_fields_present": v2_fields_present,
            "v2_fields_complete": len(v2_fields_present) == len(_V2_MANIFEST_FIELDS),
        },
        "legacy_manifest": legacy_evidence,
        "hardening": {
            "mode": "standalone_acceptance_overlay",
            "validator_commit_sha": validator_commit_sha,
            "validation_status": validation_result.get("status"),
            "validation_hardening_status": validation_result.get("hardening_status"),
            "validation_sha256": validation_hash,
        },
        "artifact_evidence": {
            "manifest_artifact_file_sha256": {
                str(name): str(digest) for name, digest in sorted(artifact_hashes.items())
            },
            "finalist_model_sha256": dict(sorted(model_hashes.items())),
        },
        "test_seal": test_seal,
    }


def write_acceptance_overlay(
    optimization_dir: Path,
    *,
    validation_result: dict[str, Any],
    validator_commit_sha: str = "unknown",
) -> tuple[Path, bool]:
    """Write the one-time acceptance overlay without overwriting prior evidence."""

    directory = optimization_dir.expanduser().resolve()
    overlay_path = directory / ACCEPTANCE_OVERLAY_FILENAME
    if overlay_path.is_file():
        _read_object(overlay_path)
        return overlay_path, False
    overlay = build_acceptance_overlay(
        directory,
        validation_result=validation_result,
        validator_commit_sha=validator_commit_sha,
    )
    write_json(overlay_path, overlay)
    return overlay_path, True


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
    "ACCEPTANCE_OVERLAY_FILENAME",
    "build_acceptance_overlay",
    "canonical_json_sha256",
    "content_sha256",
    "fold_content_sha256",
    "inner_membership_sha256",
    "prediction_sha256",
    "runtime_provenance",
    "sha256_file",
    "write_acceptance_overlay",
]
