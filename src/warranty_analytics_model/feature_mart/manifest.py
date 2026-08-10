"""Deterministic Parquet, manifest, lineage, and content-fingerprint helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .. import __version__
from .lineage import canonical_value
from .mart_contract import iter_contract_mappings, mappings_by_artifact
from .models import FeatureMartError, MartContract


def sha256_file(path: Path) -> str:
    """Hash one generated file without exposing its contents."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha256(frame: pd.DataFrame) -> str:
    """Hash stable column names and canonical row values, not Parquet bytes."""

    payload: list[str] = ["\x1e".join(str(column) for column in frame.columns)]
    for row in frame.itertuples(index=False, name=None):
        payload.append("\x1e".join(canonical_value(value) for value in row))
    return hashlib.sha256("\x1f".join(payload).encode("utf-8")).hexdigest()


def write_parquet(frame: pd.DataFrame, path: Path, *, compression: str) -> dict[str, str | int]:
    """Write a Parquet artifact without persisting a DataFrame index."""

    if compression == "none":
        frame.to_parquet(path, index=False)
    else:
        frame.to_parquet(path, index=False, compression=compression)
    return {
        "file_sha256": sha256_file(path),
        "content_sha256": content_sha256(frame),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
    }


def write_json(path: Path, payload: Any) -> None:
    """Write stable UTF-8 JSON with sorted keys."""

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def git_commit_sha(root: Path) -> str:
    """Return the current commit without failing local source-only builds."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _mapping_payload(mapping: Any, artifact_path: str) -> dict[str, Any]:
    return {
        "artifact": artifact_path,
        "artifact_name": mapping.artifact,
        "output_column": mapping.output_column,
        "source_table": mapping.source_table,
        "source_column": mapping.source_column,
        "policy": mapping.policy,
        "is_target": bool(mapping.is_target),
        "is_model_feature": bool(mapping.is_model_feature),
        "is_lineage": bool(mapping.is_lineage),
        "is_control": bool(mapping.is_control),
        "join_path": list(mapping.join_path),
        "as_of_rule": mapping.as_of_rule,
        "transform_type": mapping.transform_type,
    }


def build_column_manifest(
    contract: MartContract,
    frames: dict[str, pd.DataFrame],
    artifact_paths: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the authoritative per-column manifest and field-lineage map."""

    grouped = mappings_by_artifact(contract)
    entries: list[dict[str, Any]] = []
    lineage: dict[str, Any] = {}
    for artifact, frame in frames.items():
        mappings = grouped.get(artifact, [])
        by_output = {mapping.output_column: mapping for mapping in mappings}
        missing = sorted(set(frame.columns) - set(by_output))
        if missing:
            raise FeatureMartError(
                f"Artifact {artifact} has columns missing from the mart contract: {', '.join(missing)}"
            )
        absent = sorted(set(by_output) - set(frame.columns))
        if absent:
            raise FeatureMartError(
                f"Artifact {artifact} is missing declared columns: {', '.join(absent)}"
            )
        artifact_path = artifact_paths[artifact]
        for column in frame.columns:
            mapping = by_output[column]
            payload = _mapping_payload(mapping, artifact_path)
            entries.append(payload)
            key = column if artifact == "claim_snapshot" else f"{artifact}::{column}"
            lineage[key] = {
                "output_column": column,
                "artifact": artifact_path,
                "source": f"{mapping.source_table}.{mapping.source_column}",
                "join_path": list(mapping.join_path),
                "policy": mapping.policy,
                "is_model_feature": bool(mapping.is_model_feature),
                "is_target": bool(mapping.is_target),
                "is_lineage": bool(mapping.is_lineage),
                "is_control": bool(mapping.is_control),
                "transform_type": mapping.transform_type,
                "as_of_rule": mapping.as_of_rule,
            }
    entries.sort(key=lambda entry: (entry["artifact"], entry["output_column"]))
    return entries, lineage


def build_manifest(
    *,
    root: Path,
    contract: MartContract,
    mart_checksum: str,
    environment: str,
    source_database: str,
    source_row_counts: dict[str, int],
    eligibility: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    artifact_paths: dict[str, str],
    artifact_metadata: dict[str, dict[str, str | int]],
    bridge_row_counts: dict[str, int],
    history_coverage: dict[str, dict[str, int | float]],
    join_validation: dict[str, Any],
    validation_status: str = "INCOMPLETE",
) -> dict[str, Any]:
    """Create a secret-free aggregate manifest for a temporary or final build."""

    direct_count = sum(mapping.is_model_feature for mapping in contract.direct_feature_mappings)
    historical_count = sum(
        mapping.is_model_feature
        for bridge in contract.historical_bridge_definitions
        for mapping in bridge.field_mappings
    )
    snapshot = frames["claim_snapshot"]
    target_column = str(contract.target["output_column"])
    target = pd.to_numeric(snapshot[target_column], errors="coerce")
    metadata = {artifact: dict(item) for artifact, item in artifact_metadata.items()}
    total_source_claims = int(eligibility.get("total_claims", len(snapshot)))
    eligible_claims = int(eligibility.get("eligible_claims", len(snapshot)))
    positive_claims = int((target == 1).sum())
    negative_claims = int((target == 0).sum())
    synthetic_baseline = {"total_claims": 8500, "positive_claims": 259, "negative_claims": 8241}
    source_drift = {
        "total_claims": {
            "expected": 8500,
            "actual": total_source_claims,
            "delta": total_source_claims - 8500,
        },
        "positive_claims": {
            "expected": 259,
            "actual": positive_claims,
            "delta": positive_claims - 259,
        },
        "negative_claims": {
            "expected": 8241,
            "actual": negative_claims,
            "delta": negative_claims - 8241,
        },
    }
    return {
        "mart_version": contract.version,
        "mart_contract_checksum": mart_checksum,
        "schema_contract_version": contract.schema_contract_version,
        "schema_contract_checksum": contract.schema_contract_checksum,
        "target_contract_version": contract.target_contract_version,
        "target_contract_checksum": contract.target_contract_checksum,
        "feature_policy_version": contract.feature_policy_version,
        "feature_policy_checksum": contract.feature_policy_checksum,
        "leakage_policy_version": contract.leakage_policy_version,
        "leakage_policy_checksum": contract.leakage_policy_checksum,
        "build_timestamp": datetime.now(UTC).isoformat(),
        "environment": environment,
        "package_version": __version__,
        "git_commit": git_commit_sha(root),
        "source_database": source_database,
        "total_source_claims": total_source_claims,
        "eligible_claims": eligible_claims,
        "snapshot_rows": int(len(snapshot)),
        "positive_claims": positive_claims,
        "negative_claims": negative_claims,
        "synthetic_baseline": synthetic_baseline,
        "source_drift": source_drift,
        "direct_feature_count": direct_count,
        "historical_source_field_count": historical_count,
        "lineage_column_count": sum(
            mapping.is_lineage
            for mapping in iter_contract_mappings(contract)
            if mapping.artifact == "claim_snapshot" and mapping.is_lineage
        ),
        "bridge_row_counts": bridge_row_counts,
        "history_coverage": history_coverage,
        "direct_join_validation": join_validation,
        "source_row_counts": source_row_counts,
        "artifact_paths": artifact_paths,
        "artifact_file_sha256": {
            artifact: item["file_sha256"] for artifact, item in metadata.items()
        },
        "artifact_content_fingerprints": {
            artifact: item["content_sha256"] for artifact, item in metadata.items()
        },
        "artifact_metadata": metadata,
        "deferred_fields": contract.deferred_fields,
        "validation_status": validation_status,
    }
