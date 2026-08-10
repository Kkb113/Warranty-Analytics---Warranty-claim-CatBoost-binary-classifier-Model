"""Trusted, read-only loading of a completed Phase 5 mart."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..database.schema_contract import load_schema_contract
from ..feature_mart.manifest import content_sha256, sha256_file
from ..feature_mart.mart_contract import load_mart_contract, validate_mart_contract
from ..feature_mart.models import FeatureMartError
from ..feature_mart.validation import validate_artifact_integrity, validate_mart_directory
from ..paths import discover_repository_root
from ..policy.loader import load_phase4_contracts
from .models import SplitError


@dataclass(frozen=True, slots=True)
class Phase5MartInput:
    """All Phase 5 inputs and checksums used by a Phase 6 run."""

    root: Path
    mart_dir: Path
    manifest: dict[str, Any]
    frames: dict[str, pd.DataFrame]
    phase5_validation: dict[str, Any]
    mart_contract: Any
    mart_contract_checksum: str
    phase4_bundle: Any
    schema_contract_checksum: str
    mart_manifest_checksum: str
    claim_snapshot_file_sha256: str
    claim_snapshot_content_sha256: str
    group_membership_file_sha256: str
    group_membership_content_sha256: str


def _required_artifact_hash(
    manifest: dict[str, Any],
    artifact_name: str,
    *,
    hash_name: str,
) -> str:
    hashes = manifest.get(hash_name)
    if not isinstance(hashes, dict) or not isinstance(hashes.get(artifact_name), str):
        raise SplitError(f"Phase 5 manifest is missing {hash_name} for {artifact_name}.")
    return str(hashes[artifact_name])


def _verify_artifact_hashes(
    mart_dir: Path,
    manifest: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    artifact_name: str,
) -> tuple[str, str]:
    paths = manifest.get("artifact_paths")
    if not isinstance(paths, dict) or not isinstance(paths.get(artifact_name), str):
        raise SplitError(f"Phase 5 manifest is missing the {artifact_name} path.")
    path = mart_dir / str(paths[artifact_name])
    if not path.is_file():
        raise SplitError(f"Required Phase 5 artifact is missing: {path}")
    file_hash = sha256_file(path)
    content_hash = content_sha256(frames[artifact_name])
    if file_hash != _required_artifact_hash(
        manifest, artifact_name, hash_name="artifact_file_sha256"
    ):
        raise SplitError(f"Phase 5 {artifact_name} file checksum does not match its manifest.")
    if content_hash != _required_artifact_hash(
        manifest, artifact_name, hash_name="artifact_content_fingerprints"
    ):
        raise SplitError(f"Phase 5 {artifact_name} content checksum does not match its manifest.")
    return file_hash, content_hash


def load_phase5_mart(
    mart_dir: Path,
    *,
    project_root: Path | None = None,
) -> Phase5MartInput:
    """Validate and load only the completed Phase 5 artifacts required by Phase 6."""

    root = discover_repository_root(project_root)
    resolved_mart_dir = mart_dir.expanduser().resolve()
    if not resolved_mart_dir.is_dir():
        raise SplitError(f"Phase 5 mart directory is missing: {resolved_mart_dir}")
    manifest_path = resolved_mart_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SplitError("Phase 5 manifest.json is required before Phase 6 can start.")
    try:
        loaded, frames = validate_artifact_integrity(resolved_mart_dir)
        manifest = loaded["manifest"]
    except (FeatureMartError, OSError, ValueError) as exc:
        raise SplitError(f"Phase 5 artifact integrity validation failed: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SplitError("Phase 5 manifest.json must contain a JSON object.")
    status = str(manifest.get("validation_status", ""))
    if status not in {"PASS", "PASS WITH WARNINGS"}:
        raise SplitError(f"Phase 5 validation status {status or 'MISSING'} blocks Phase 6.")
    for required in ("claim_snapshot", "claim_group_membership"):
        if required not in frames:
            raise SplitError(f"Phase 5 artifact {required} is required before Phase 6.")
    try:
        phase5_validation = validate_mart_directory(resolved_mart_dir, project_root=root)
    except (FeatureMartError, OSError, ValueError) as exc:
        raise SplitError(f"Phase 5 validation could not be completed: {exc}") from exc
    if phase5_validation.get("errors"):
        errors = "; ".join(str(item) for item in phase5_validation["errors"])
        raise SplitError(f"Phase 5 validation blocks Phase 6: {errors}")

    schema_contract, schema_checksum = load_schema_contract(root)
    phase4_bundle = load_phase4_contracts(root)
    mart_contract, mart_checksum = load_mart_contract(root)
    mart_plan = validate_mart_contract(
        schema_contract,
        phase4_bundle,
        schema_contract_checksum=schema_checksum,
        contract=mart_contract,
        contract_checksum=mart_checksum,
    )
    if not mart_plan.valid:
        raise SplitError("Phase 5 mart contract blocks Phase 6: " + "; ".join(mart_plan.errors))
    if manifest.get("mart_contract_checksum") != mart_checksum:
        raise SplitError(
            "Phase 5 mart contract checksum does not match the completed mart manifest."
        )
    expected_contracts = {
        "schema_contract_checksum": schema_checksum,
        "target_contract_checksum": phase4_bundle.target_checksum,
        "feature_policy_checksum": phase4_bundle.feature_policy_checksum,
        "leakage_policy_checksum": phase4_bundle.leakage_checksum,
    }
    for field, expected in expected_contracts.items():
        if manifest.get(field) != expected:
            raise SplitError(f"Phase 5 manifest {field} does not match the current contract.")

    snapshot = frames["claim_snapshot"]
    group_membership = frames["claim_group_membership"]
    required_snapshot = {
        "warranty_claim_key",
        "claim__claim_date",
        "target__high_cost_claim_flag",
    }
    missing_snapshot = sorted(required_snapshot - set(snapshot.columns))
    if missing_snapshot:
        raise SplitError(
            "Phase 5 claim snapshot is missing required columns: " + ", ".join(missing_snapshot)
        )
    if len(snapshot) != int(manifest.get("eligible_claims", -1)):
        raise SplitError("Phase 5 snapshot rows do not match eligible_claims in the manifest.")
    if snapshot["warranty_claim_key"].isna().any():
        raise SplitError("Phase 5 snapshot contains null claim keys.")
    if snapshot["warranty_claim_key"].duplicated().any():
        raise SplitError("Phase 5 snapshot claim keys are not unique.")
    dates = pd.to_datetime(snapshot["claim__claim_date"], errors="coerce")
    if dates.isna().any():
        raise SplitError("Phase 5 snapshot contains null or invalid claim dates.")
    target = pd.to_numeric(snapshot["target__high_cost_claim_flag"], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise SplitError("Phase 5 stored target is not valid binary {0, 1}.")
    if group_membership["warranty_claim_key"].isna().any():
        raise SplitError("Phase 5 group membership contains null claim keys.")
    snapshot_file_hash, snapshot_content_hash = _verify_artifact_hashes(
        resolved_mart_dir, manifest, frames, "claim_snapshot"
    )
    group_file_hash, group_content_hash = _verify_artifact_hashes(
        resolved_mart_dir, manifest, frames, "claim_group_membership"
    )
    return Phase5MartInput(
        root=root,
        mart_dir=resolved_mart_dir,
        manifest=manifest,
        frames=frames,
        phase5_validation=phase5_validation,
        mart_contract=mart_contract,
        mart_contract_checksum=mart_checksum,
        phase4_bundle=phase4_bundle,
        schema_contract_checksum=schema_checksum,
        mart_manifest_checksum=sha256_file(manifest_path),
        claim_snapshot_file_sha256=snapshot_file_hash,
        claim_snapshot_content_sha256=snapshot_content_hash,
        group_membership_file_sha256=group_file_hash,
        group_membership_content_sha256=group_content_hash,
    )
