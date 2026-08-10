"""Canonical Phase 8 ordering, manifests, and content fingerprints."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .. import __version__
from ..feature_mart.manifest import (
    content_sha256,
    git_commit_sha,
    sha256_file,
    write_json,
    write_parquet,
)
from .config import settings_payload
from .models import Phase8Inputs, TextBuildResult, TextFeatureSettings

CONTROL_COLUMNS = ("warranty_claim_key", "split", "claim__claim_date")
DOCUMENT_COLUMNS = tuple(
    f"prior_failure_text__{window}__document" for window in ("6m", "12m", "24m", "all")
)


def ordered_text_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return canonical claim/key and stable column order without an index."""

    controls = [column for column in CONTROL_COLUMNS if column in frame]
    documents = [column for column in DOCUMENT_COLUMNS if column in frame]
    lexical = sorted(column for column in frame if column.startswith("text__"))
    ordered = controls + documents + lexical
    if set(ordered) != set(frame.columns):
        missing = sorted(set(frame.columns) - set(ordered))
        raise ValueError("Phase 8 frame has unrecognized columns: " + ", ".join(missing))
    return frame[ordered].sort_values("warranty_claim_key", kind="mergesort").reset_index(drop=True)


def build_text_feature_manifest(result: TextBuildResult) -> dict[str, Any]:
    """Build aggregate feature inventory metadata."""

    definitions = result.definitions
    model_definitions = [item for item in definitions if item.is_model_feature]
    documents = [item for item in model_definitions if item.feature_type == "text"]
    lexical = [item for item in model_definitions if item.feature_type in {"numeric", "boolean"}]
    by_window: dict[str, dict[str, int]] = {}
    for window in ("6m", "12m", "24m", "all"):
        by_window[window] = {
            "document_features": sum(item.window == window for item in documents),
            "lexical_features": sum(
                item.window == window and item.feature_name != "text__has_prior_failure_description"
                for item in lexical
            ),
        }
    return {
        "total_text_document_features": len(documents),
        "lexical_feature_count": len(lexical),
        "boolean_feature_count": sum(item.feature_type == "boolean" for item in lexical),
        "numeric_feature_count": sum(item.feature_type == "numeric" for item in lexical),
        "feature_counts_by_window": by_window,
        "approved_source_count": 1,
        "deferred_source_count": len(result.source_coverage.get("deferred", [])),
        "prohibited_source_count": int(
            result.source_coverage.get("prohibited_model_feature_count", 0)
        ),
        "feature_names": [item.feature_name for item in model_definitions],
    }


def build_text_run_manifest(
    *,
    root: Path,
    inputs: Phase8Inputs,
    result: TextBuildResult,
    settings: TextFeatureSettings,
    contract_checksum: str,
    artifact_metadata: dict[str, Any],
    warnings: list[str],
    validation_status: str,
) -> dict[str, Any]:
    """Build the aggregate-only Phase 8 run manifest."""

    assignments = inputs.assignments
    counts = assignments["split"].value_counts().to_dict()
    manifest = {
        "phase8_version": "1.0.0",
        "package_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit_sha(root),
        "phase8_contract_checksum": contract_checksum,
        "input_phase5_run": inputs.mart_dir.name,
        "input_phase5_contract_checksum": inputs.phase5_contract_checksum,
        "input_phase5_manifest_checksum": inputs.phase5_manifest_checksum,
        "input_phase5_claim_snapshot_content_sha256": inputs.phase5_manifest.get(
            "artifact_content_fingerprints", {}
        ).get("claim_snapshot", ""),
        "input_phase6_run": inputs.split_dir.name,
        "input_phase6_contract_checksum": inputs.phase6_contract_checksum,
        "split_assignment_sha256": inputs.phase6_manifest.get("split_assignment_sha256"),
        "train_claim_key_sha256": inputs.phase6_manifest.get("train_claim_key_sha256"),
        "validation_claim_key_sha256": inputs.phase6_manifest.get("validation_claim_key_sha256"),
        "test_claim_key_sha256": inputs.phase6_manifest.get("test_claim_key_sha256"),
        "test_lock_hashes": {
            key: inputs.test_lock.get(key)
            for key in (
                "ordered_test_claim_keys_sha256",
                "unordered_test_claim_keys_sha256",
                "test_assignment_content_sha256",
            )
        },
        "input_phase7_run": inputs.structured_dir.name,
        "phase7_contract_checksum": inputs.phase7_contract_checksum,
        "structured_feature_content_sha256": inputs.phase7_content_sha256,
        "row_count": int(len(result.frame)),
        "train_count": int(counts.get("TRAIN", 0)),
        "validation_count": int(counts.get("VALIDATION", 0)),
        "test_count": int(counts.get("TEST", 0)),
        "text_feature_count": int(sum(item.is_model_feature for item in result.definitions)),
        "text_document_feature_count": int(
            sum(
                item.is_model_feature and item.feature_type == "text" for item in result.definitions
            )
        ),
        "lexical_feature_count": int(
            sum(
                item.is_model_feature and item.feature_type in {"numeric", "boolean"}
                for item in result.definitions
            )
        ),
        "artifact_paths": {
            "text_features": "text_features.parquet",
            "text_feature_manifest": "text_feature_manifest.json",
            "text_feature_lineage": "text_feature_lineage.json",
            "text_quality": "text_quality.json",
            "manifest": "manifest.json",
            "validation": "validation.json",
        },
        "artifact_file_sha256": {"text_features": artifact_metadata["file_sha256"]},
        "artifact_content_sha256": {"text_features": artifact_metadata["content_sha256"]},
        "settings": settings_payload(settings),
        "source_coverage": result.source_coverage,
        "target_column_present": False,
        "fitted_transformations": [],
        "text_model_training": False,
        "test_based_tuning": False,
        "warnings": list(dict.fromkeys(warnings)),
        "validation_status": validation_status,
    }
    return manifest


def write_text_artifact(frame: pd.DataFrame, path: Path, *, compression: str) -> dict[str, Any]:
    """Write the canonical text frame and return binary/content hashes."""

    ordered = ordered_text_frame(frame)
    return {
        str(key): value
        for key, value in write_parquet(ordered, path, compression=compression).items()
    }


def write_text_json(path: Path, payload: Any) -> None:
    """Write stable JSON using the shared aggregate-safe helper."""

    write_json(path, payload)


__all__ = [
    "CONTROL_COLUMNS",
    "DOCUMENT_COLUMNS",
    "build_text_feature_manifest",
    "build_text_run_manifest",
    "content_sha256",
    "ordered_text_frame",
    "sha256_file",
    "write_text_artifact",
    "write_text_json",
]
