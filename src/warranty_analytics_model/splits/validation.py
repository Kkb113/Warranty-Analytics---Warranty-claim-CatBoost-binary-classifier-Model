"""Fail-closed validation for Phase 6 split artifacts and the frozen test lock."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..database.schema_contract import load_schema_contract
from ..feature_mart.manifest import content_sha256, sha256_file
from ..paths import discover_repository_root
from .assignments import (
    ASSIGNMENT_COLUMNS,
    assignment_date_order_errors,
    validate_assignment_frame,
)
from .cohorts import build_evaluation_cohorts
from .config import load_split_settings
from .group_exposure import build_group_exposure
from .input import Phase5MartInput, load_phase5_mart
from .manifest import (
    assignment_content_sha256,
    claim_key_sha256,
    mart_input_fingerprint,
    unordered_claim_key_sha256,
)
from .models import SplitError
from .split_contract import load_split_contract, validate_split_contract


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise SplitError(f"Required Phase 6 {label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitError(f"Phase 6 {label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SplitError(f"Phase 6 {label} must contain a JSON object: {path}")
    return payload


def _date_string(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return str(pd.Timestamp(parsed).date().isoformat())


def _split_counts(assignments: pd.DataFrame, snapshot: pd.DataFrame) -> dict[str, dict[str, int]]:
    targets = snapshot[["warranty_claim_key", "target__high_cost_claim_flag"]].copy()
    targets["target__high_cost_claim_flag"] = pd.to_numeric(
        targets["target__high_cost_claim_flag"], errors="coerce"
    )
    joined = assignments.merge(targets, on="warranty_claim_key", how="left", validate="one_to_one")
    result: dict[str, dict[str, int]] = {}
    for split in ("TRAIN", "VALIDATION", "TEST"):
        subset = joined.loc[joined["split"] == split]
        positives = int((subset["target__high_cost_claim_flag"] == 1).sum())
        negatives = int((subset["target__high_cost_claim_flag"] == 0).sum())
        result[split] = {
            "row_count": int(len(subset)),
            "positive_count": positives,
            "negative_count": negatives,
        }
    return result


def _artifact_check(
    split_dir: Path,
    manifest: dict[str, Any],
    artifact_name: str,
    frame: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []
    relative = {
        "split_assignments": "split_assignments.parquet",
        "group_exposure": "group_exposure.parquet",
        "evaluation_cohorts": "evaluation_cohorts.parquet",
    }[artifact_name]
    path = split_dir / relative
    if not path.is_file():
        return [f"Required Phase 6 artifact is missing: {relative}"]
    metadata = manifest.get("artifact_checksums", {}).get(artifact_name)
    if not isinstance(metadata, dict):
        return [f"Split manifest is missing artifact checksums for {artifact_name}."]
    expected_file = metadata.get("file_sha256")
    expected_content = metadata.get("content_sha256")
    if not isinstance(expected_file, str) or sha256_file(path) != expected_file:
        errors.append(f"Phase 6 {artifact_name} file checksum mismatch.")
    if not isinstance(expected_content, str) or content_sha256(frame) != expected_content:
        errors.append(f"Phase 6 {artifact_name} content checksum mismatch.")
    return errors


def _load_input_from_manifest(split_manifest: dict[str, Any], root: Path) -> Phase5MartInput:
    relative = split_manifest.get("input_mart_relative_path")
    run = split_manifest.get("input_mart_run")
    if isinstance(relative, str) and relative:
        mart_path = Path(relative)
        if not mart_path.is_absolute():
            mart_path = root / mart_path
    elif isinstance(run, str) and run:
        mart_path = root / "artifacts" / "feature_mart" / run
    else:
        raise SplitError("Split manifest does not identify its Phase 5 mart input.")
    return load_phase5_mart(mart_path, project_root=root)


def _contract_compatibility_errors(
    split_manifest: dict[str, Any],
    input_mart: Phase5MartInput,
    root: Path,
) -> list[str]:
    contract, contract_checksum = load_split_contract(root)
    settings = load_split_settings(root)
    schema_contract, schema_checksum = load_schema_contract(root)
    result = validate_split_contract(
        contract,
        mart_contract=input_mart.mart_contract,
        mart_contract_checksum=input_mart.mart_contract_checksum,
        schema_contract_checksum=schema_checksum,
        phase4_bundle=input_mart.phase4_bundle,
        settings=settings,
        split_contract_checksum_value=contract_checksum,
    )
    errors = list(result.errors)
    if split_manifest.get("split_contract_checksum") != contract_checksum:
        errors.append(
            "Split manifest split-contract checksum does not match the versioned contract."
        )
    if split_manifest.get("input_mart_contract_checksum") != input_mart.mart_contract_checksum:
        errors.append("Split manifest Phase 5 mart checksum does not match the input mart.")
    if split_manifest.get("input_mart_manifest_checksum") != input_mart.mart_manifest_checksum:
        errors.append("Split manifest Phase 5 manifest checksum does not match the input mart.")
    expected = {
        "input_schema_contract_checksum": input_mart.schema_contract_checksum,
        "input_target_contract_checksum": input_mart.phase4_bundle.target_checksum,
        "input_feature_policy_checksum": input_mart.phase4_bundle.feature_policy_checksum,
        "input_leakage_policy_checksum": input_mart.phase4_bundle.leakage_checksum,
        "claim_snapshot_file_sha256": input_mart.claim_snapshot_file_sha256,
        "claim_snapshot_content_sha256": input_mart.claim_snapshot_content_sha256,
        "group_membership_file_sha256": input_mart.group_membership_file_sha256,
        "group_membership_content_sha256": input_mart.group_membership_content_sha256,
    }
    for field, value in expected.items():
        if split_manifest.get(field) != value:
            errors.append(f"Split manifest {field} does not match the frozen Phase 5 input.")
    return list(dict.fromkeys(errors))


def validate_split_artifacts(
    split_dir: Path,
    *,
    project_root: Path | None = None,
    input_mart: Phase5MartInput | None = None,
    expected_exposure: pd.DataFrame | None = None,
    expected_cohorts: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Validate a completed Phase 6 run entirely from local artifacts."""

    root = discover_repository_root(project_root)
    resolved_split_dir = split_dir.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        split_manifest = _read_json(
            resolved_split_dir / "split_manifest.json", "split_manifest.json"
        )
        test_lock = _read_json(resolved_split_dir / "test_lock.json", "test_lock.json")
    except SplitError as exc:
        return {
            "status": "BLOCKED",
            "errors": [str(exc)],
            "warnings": [],
            "checks": {},
        }
    if input_mart is None:
        try:
            input_mart = _load_input_from_manifest(split_manifest, root)
        except (SplitError, OSError, ValueError) as exc:
            return {
                "status": "BLOCKED",
                "errors": [str(exc)],
                "warnings": [],
                "checks": {"input_mart_valid": False},
            }
    try:
        errors.extend(_contract_compatibility_errors(split_manifest, input_mart, root))
    except (SplitError, OSError, ValueError) as exc:
        errors.append(str(exc))
    try:
        assignments = pd.read_parquet(resolved_split_dir / "split_assignments.parquet")
        exposure = pd.read_parquet(resolved_split_dir / "group_exposure.parquet")
        cohorts = pd.read_parquet(resolved_split_dir / "evaluation_cohorts.parquet")
    except (OSError, ValueError, ImportError) as exc:
        errors.append(f"Could not read Phase 6 Parquet artifacts: {exc}")
        return {
            "status": "BLOCKED",
            "errors": list(dict.fromkeys(errors)),
            "warnings": [],
            "checks": {"input_mart_valid": True},
        }
    try:
        validate_assignment_frame(
            assignments, expected_claim_count=len(input_mart.frames["claim_snapshot"])
        )
    except SplitError as exc:
        errors.append(str(exc))
    missing_assignment_columns = sorted(set(ASSIGNMENT_COLUMNS) - set(assignments.columns))
    if missing_assignment_columns:
        errors.append(
            "Split assignments are missing required columns: "
            + ", ".join(missing_assignment_columns)
        )
        return {
            "status": "BLOCKED",
            "errors": list(dict.fromkeys(errors)),
            "warnings": [],
            "checks": {"input_mart_valid": True, "assignment_valid": False},
        }
    if set(assignments.columns) != set(ASSIGNMENT_COLUMNS):
        errors.append(
            "Split assignments must contain only claim key, claim date, and split metadata."
        )

    snapshot = input_mart.frames["claim_snapshot"]
    snapshot_keys = set(snapshot["warranty_claim_key"].tolist())
    assignment_keys = set(assignments["warranty_claim_key"].tolist())
    if assignment_keys != snapshot_keys:
        errors.append("Split assignments do not cover exactly the Phase 5 eligible claim keys.")
    if len(assignment_keys) != len(snapshot_keys):
        errors.append("Split assignment claim coverage is not unique.")
    try:
        errors.extend(assignment_date_order_errors(assignments))
    except SplitError as exc:
        errors.append(str(exc))
    snapshot_dates = snapshot[["warranty_claim_key", "claim__claim_date"]].copy()
    snapshot_dates["claim_date"] = pd.to_datetime(
        snapshot_dates.pop("claim__claim_date"), errors="coerce"
    ).dt.normalize()
    date_join = assignments.merge(
        snapshot_dates,
        on="warranty_claim_key",
        how="left",
        validate="one_to_one",
        suffixes=("", "_snapshot"),
    )
    if not date_join["claim_date_snapshot"].equals(date_join["claim_date"]):
        errors.append("Split assignment claim dates do not match the Phase 5 snapshot.")
    counts = _split_counts(assignments, snapshot)
    total_positive = int(pd.to_numeric(snapshot["target__high_cost_claim_flag"]).eq(1).sum())
    total_negative = int(pd.to_numeric(snapshot["target__high_cost_claim_flag"]).eq(0).sum())
    if sum(item["positive_count"] for item in counts.values()) != total_positive:
        errors.append("Split positive counts do not reconcile to the Phase 5 snapshot.")
    if sum(item["negative_count"] for item in counts.values()) != total_negative:
        errors.append("Split negative counts do not reconcile to the Phase 5 snapshot.")

    errors.extend(
        _artifact_check(resolved_split_dir, split_manifest, "split_assignments", assignments)
    )
    errors.extend(_artifact_check(resolved_split_dir, split_manifest, "group_exposure", exposure))
    errors.extend(
        _artifact_check(resolved_split_dir, split_manifest, "evaluation_cohorts", cohorts)
    )
    if expected_exposure is None:
        try:
            expected_exposure = build_group_exposure(
                assignments, input_mart.frames["claim_group_membership"]
            )
        except SplitError as exc:
            errors.append(str(exc))
    if expected_exposure is not None and content_sha256(exposure) != content_sha256(
        expected_exposure
    ):
        errors.append(
            "Group exposure content does not match Phase 5 group lineage and assignments."
        )
    if expected_cohorts is None:
        try:
            expected_cohorts = build_evaluation_cohorts(
                assignments, input_mart.frames["claim_group_membership"]
            )
        except SplitError as exc:
            errors.append(str(exc))
    if expected_cohorts is not None and content_sha256(cohorts) != content_sha256(expected_cohorts):
        errors.append("Evaluation cohort content does not match group exposure definitions.")
    if (
        "target__high_cost_claim_flag" in exposure.columns
        or "target__high_cost_claim_flag" in cohorts.columns
    ):
        errors.append("Phase 6 group/cohort artifacts must not contain the target.")
    if "is_model_feature" in exposure and exposure["is_model_feature"].eq(True).any():
        errors.append("Group exposure metadata cannot be marked as model features.")
    if "is_model_feature" in cohorts and cohorts["is_model_feature"].eq(True).any():
        errors.append("Evaluation cohort metadata cannot be marked as model features.")

    assignment_hash = assignment_content_sha256(assignments)
    if split_manifest.get("split_assignment_sha256") != assignment_hash:
        errors.append("Split assignment content hash does not match split_manifest.json.")
    for split in ("TRAIN", "VALIDATION", "TEST"):
        subset = assignments.loc[assignments["split"] == split]
        field = f"{split.casefold()}_claim_key_sha256"
        if split_manifest.get(field) != claim_key_sha256(subset):
            errors.append(f"{split} claim membership hash does not match split_manifest.json.")
    test_assignments = assignments.loc[assignments["split"] == "TEST"]
    expected_lock = {
        "split_contract_version": split_manifest.get("split_contract_version"),
        "split_contract_checksum": split_manifest.get("split_contract_checksum"),
        "input_mart_checksum": split_manifest.get("input_mart_contract_checksum"),
        "input_mart_fingerprint": mart_input_fingerprint(
            mart_contract_checksum=input_mart.mart_contract_checksum,
            claim_snapshot_content_sha256=input_mart.claim_snapshot_content_sha256,
            group_membership_content_sha256=input_mart.group_membership_content_sha256,
        ),
        "claim_snapshot_content_sha256": input_mart.claim_snapshot_content_sha256,
        "test_row_count": int(len(test_assignments)),
        "test_start_date": _date_string(test_assignments["claim_date"].min()),
        "test_end_date": _date_string(test_assignments["claim_date"].max()),
        "ordered_test_claim_keys_sha256": claim_key_sha256(test_assignments),
        "unordered_test_claim_keys_sha256": unordered_claim_key_sha256(test_assignments),
        "test_assignment_content_sha256": assignment_content_sha256(test_assignments),
        "locked": True,
        "allowed_first_target_evaluation_phase": 15,
    }
    for field, expected in expected_lock.items():
        if test_lock.get(field) != expected:
            errors.append(f"Test lock field {field} does not match the frozen split.")

    warnings.extend(str(item) for item in split_manifest.get("warnings", []) if item)
    warnings.extend(str(item) for item in input_mart.phase5_validation.get("warnings", []) if item)
    status = "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS")
    return {
        "status": status,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "checks": {
            "input_mart_valid": not errors,
            "claim_coverage_valid": assignment_keys == snapshot_keys
            and len(assignment_keys) == len(snapshot_keys),
            "date_order_valid": not assignment_date_order_errors(assignments),
            "same_date_integrity_valid": not any(
                "claim date appears" in item.casefold() for item in errors
            ),
            "target_counts_reconcile": sum(item["positive_count"] for item in counts.values())
            == total_positive
            and sum(item["negative_count"] for item in counts.values()) == total_negative,
            "group_exposure_valid": expected_exposure is not None
            and content_sha256(exposure) == content_sha256(expected_exposure),
            "cohorts_valid": expected_cohorts is not None
            and content_sha256(cohorts) == content_sha256(expected_cohorts),
            "test_lock_valid": not any("Test lock field" in item for item in errors),
            "reproducibility_metadata_valid": bool(split_manifest.get("split_contract_checksum")),
        },
        "split_counts": counts,
    }
