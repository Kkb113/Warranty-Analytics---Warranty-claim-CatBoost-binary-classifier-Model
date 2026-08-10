"""Deterministic Phase 6 artifact fingerprints, manifests, and test lock."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .. import __version__
from ..feature_mart.lineage import canonical_value
from ..feature_mart.manifest import write_parquet


def phase6_run_id() -> str:
    """Create a readable UTC run identifier."""

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _hash_text(values: list[str]) -> str:
    return hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()


def _ordered_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    return assignments.sort_values(
        ["claim_date", "warranty_claim_key"], kind="mergesort", na_position="last"
    ).reset_index(drop=True)


def assignment_content_sha256(assignments: pd.DataFrame) -> str:
    """Hash canonical assignment fields in date/key order."""

    ordered = _ordered_assignments(assignments)
    rows = ["warranty_claim_key\x1eclaim_date\x1esplit"]
    for claim_key, claim_date, split in ordered[
        ["warranty_claim_key", "claim_date", "split"]
    ].itertuples(index=False, name=None):
        date_value = pd.Timestamp(claim_date).date().isoformat()
        rows.append(
            "\x1e".join((canonical_value(claim_key), date_value, canonical_value(str(split))))
        )
    return _hash_text(rows)


def claim_key_sha256(assignments: pd.DataFrame) -> str:
    """Hash claim keys in canonical date/key order."""

    ordered = _ordered_assignments(assignments)
    return _hash_text([canonical_value(value) for value in ordered["warranty_claim_key"]])


def unordered_claim_key_sha256(assignments: pd.DataFrame) -> str:
    """Hash claim keys in canonical key order, independent of row order."""

    values = sorted(canonical_value(value) for value in assignments["warranty_claim_key"])
    return _hash_text(values)


def mart_input_fingerprint(
    *,
    mart_contract_checksum: str,
    claim_snapshot_content_sha256: str,
    group_membership_content_sha256: str,
) -> str:
    """Fingerprint the exact Phase 5 mart inputs used by a split run."""

    return _hash_text(
        [mart_contract_checksum, claim_snapshot_content_sha256, group_membership_content_sha256]
    )


def artifact_metadata(
    frame: pd.DataFrame, path: Path, *, compression: str = "snappy"
) -> dict[str, Any]:
    """Write one Parquet artifact and return file/content fingerprints."""

    metadata = write_parquet(frame, path, compression=compression)
    return {str(key): value for key, value in metadata.items()}


def build_test_lock(
    *,
    split_contract_version: str,
    split_contract_checksum: str,
    input_mart_checksum: str,
    input_mart_fingerprint: str,
    claim_snapshot_content_sha256: str,
    test_assignments: pd.DataFrame,
    test_assignment_content_sha256: str,
    test_start_date: str,
    test_end_date: str,
) -> dict[str, Any]:
    """Build the hash-only immutable test lock."""

    return {
        "split_contract_version": split_contract_version,
        "split_contract_checksum": split_contract_checksum,
        "input_mart_checksum": input_mart_checksum,
        "input_mart_fingerprint": input_mart_fingerprint,
        "claim_snapshot_content_sha256": claim_snapshot_content_sha256,
        "test_row_count": int(len(test_assignments)),
        "test_start_date": test_start_date,
        "test_end_date": test_end_date,
        "ordered_test_claim_keys_sha256": claim_key_sha256(test_assignments),
        "unordered_test_claim_keys_sha256": unordered_claim_key_sha256(test_assignments),
        "test_assignment_content_sha256": test_assignment_content_sha256,
        "created_at": datetime.now(UTC).isoformat(),
        "locked": True,
        "allowed_first_target_evaluation_phase": 15,
        "package_version": __version__,
    }


def build_split_manifest(
    *,
    split_contract_version: str,
    split_contract_checksum: str,
    input_mart_run: str,
    input_mart_contract_version: str,
    input_mart_contract_checksum: str,
    input_mart_manifest_checksum: str,
    input_mart_relative_path: str,
    input_schema_contract_checksum: str,
    input_target_contract_checksum: str,
    input_feature_policy_checksum: str,
    input_leakage_policy_checksum: str,
    claim_snapshot_file_sha256: str,
    claim_snapshot_content_sha256: str,
    group_membership_file_sha256: str,
    group_membership_content_sha256: str,
    total_claims: int,
    requested_split_fractions: dict[str, float],
    actual_split_fractions: dict[str, float],
    train_end_date: str,
    validation_end_date: str,
    counts: dict[str, dict[str, int]],
    assignments: pd.DataFrame,
    group_exposure: pd.DataFrame,
    evaluation_cohorts: pd.DataFrame,
    artifact_metadata_by_name: dict[str, dict[str, Any]],
    warnings: list[str],
    validation_status: str,
    input_mart_source_drift: dict[str, Any],
) -> dict[str, Any]:
    """Build the aggregate-only split manifest."""

    train = assignments.loc[assignments["split"] == "TRAIN"]
    validation = assignments.loc[assignments["split"] == "VALIDATION"]
    test = assignments.loc[assignments["split"] == "TEST"]
    assignment_hash = assignment_content_sha256(assignments)
    artifacts = {name: dict(metadata) for name, metadata in artifact_metadata_by_name.items()}
    return {
        "phase6_version": "1.0.0",
        "package_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "split_contract_version": split_contract_version,
        "split_contract_checksum": split_contract_checksum,
        "input_mart_run": input_mart_run,
        "input_mart_relative_path": input_mart_relative_path,
        "input_mart_contract_version": input_mart_contract_version,
        "input_mart_contract_checksum": input_mart_contract_checksum,
        "input_mart_manifest_checksum": input_mart_manifest_checksum,
        "input_schema_contract_checksum": input_schema_contract_checksum,
        "input_target_contract_checksum": input_target_contract_checksum,
        "input_feature_policy_checksum": input_feature_policy_checksum,
        "input_leakage_policy_checksum": input_leakage_policy_checksum,
        "claim_snapshot_file_sha256": claim_snapshot_file_sha256,
        "claim_snapshot_content_sha256": claim_snapshot_content_sha256,
        "group_membership_file_sha256": group_membership_file_sha256,
        "group_membership_content_sha256": group_membership_content_sha256,
        "input_mart_source_drift": input_mart_source_drift,
        "total_claims": total_claims,
        "requested_split_fractions": requested_split_fractions,
        "actual_split_fractions": actual_split_fractions,
        "train_end_date": train_end_date,
        "validation_end_date": validation_end_date,
        "train_count": int(len(train)),
        "validation_count": int(len(validation)),
        "test_count": int(len(test)),
        "train_positive": int(counts["TRAIN"]["positive_count"]),
        "train_negative": int(counts["TRAIN"]["negative_count"]),
        "validation_positive": int(counts["VALIDATION"]["positive_count"]),
        "validation_negative": int(counts["VALIDATION"]["negative_count"]),
        "test_positive": int(counts["TEST"]["positive_count"]),
        "test_negative": int(counts["TEST"]["negative_count"]),
        "split_assignment_sha256": assignment_hash,
        "train_claim_key_sha256": claim_key_sha256(train),
        "validation_claim_key_sha256": claim_key_sha256(validation),
        "test_claim_key_sha256": claim_key_sha256(test),
        "group_exposure_sha256": artifacts["group_exposure"]["content_sha256"],
        "evaluation_cohort_sha256": artifacts["evaluation_cohorts"]["content_sha256"],
        "artifact_checksums": artifacts,
        "group_exposure_rows": int(len(group_exposure)),
        "evaluation_cohort_rows": int(len(evaluation_cohorts)),
        "warnings": list(dict.fromkeys(warnings)),
        "validation_status": validation_status,
    }
