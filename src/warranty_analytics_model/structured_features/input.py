"""Load and verify the exact offline Phase 5 and Phase 6 inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import discover_repository_root
from ..splits.input import Phase5MartInput, load_phase5_mart
from ..splits.manifest import (
    assignment_content_sha256,
    claim_key_sha256,
    unordered_claim_key_sha256,
)
from ..splits.validation import validate_split_artifacts
from .contract import validate_structured_feature_contract
from .models import Phase7Inputs, StructuredFeatureError


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise StructuredFeatureError(f"Required Phase 7 {label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StructuredFeatureError(f"Phase 7 {label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise StructuredFeatureError(f"Phase 7 {label} must contain a JSON object.")
    return payload


def _load_split_json(split_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        _read_json(split_dir / "split_manifest.json", "split_manifest.json"),
        _read_json(split_dir / "test_lock.json", "test_lock.json"),
    )


def load_phase7_inputs(
    mart_dir: Path,
    split_dir: Path,
    *,
    project_root: Path | None = None,
) -> Phase7Inputs:
    """Load all required artifacts and fail closed on any Phase 5/6 error."""

    root = discover_repository_root(project_root)
    try:
        phase5: Phase5MartInput = load_phase5_mart(mart_dir, project_root=root)
    except Exception as exc:
        raise StructuredFeatureError(f"Phase 5 validation blocks Phase 7: {exc}") from exc
    resolved_split = split_dir.expanduser().resolve()
    split_manifest, test_lock = _load_split_json(resolved_split)
    try:
        phase6_validation = validate_split_artifacts(
            resolved_split, project_root=root, input_mart=phase5
        )
    except Exception as exc:
        raise StructuredFeatureError(f"Phase 6 validation blocks Phase 7: {exc}") from exc
    if phase6_validation.get("errors"):
        raise StructuredFeatureError(
            "Phase 6 validation blocks Phase 7: "
            + "; ".join(str(item) for item in phase6_validation["errors"])
        )
    if not phase6_validation.get("checks", {}).get("test_lock_valid", False):
        raise StructuredFeatureError("Phase 6 TEST lock validation blocks Phase 7.")
    if split_manifest.get("input_mart_run") != resolved_mart_name(phase5.mart_dir):
        raise StructuredFeatureError("Phase 6 split does not consume the requested Phase 5 mart.")
    required = {
        "claim_snapshot": "claim_snapshot.parquet",
        "telemetry_history": "history/telemetry_history.parquet",
        "maintenance_history": "history/maintenance_history.parquet",
        "service_history": "history/service_history.parquet",
        "component_installation_history": "history/component_installation_history.parquet",
        "prior_claim_history": "history/prior_claim_history.parquet",
        "repair_history_index": "history/repair_history_index.parquet",
    }
    for artifact, relative in required.items():
        if artifact not in phase5.frames or not (phase5.mart_dir / relative).is_file():
            raise StructuredFeatureError(f"Required Phase 5 artifact is missing: {relative}")
    if not (resolved_split / "split_assignments.parquet").is_file():
        raise StructuredFeatureError("Phase 6 split_assignments.parquet is missing.")
    return Phase7Inputs(
        root=root,
        mart_dir=phase5.mart_dir,
        split_dir=resolved_split,
        mart_manifest=phase5.manifest,
        split_manifest=split_manifest,
        test_lock=test_lock,
        frames=phase5.frames,
        phase5_validation=phase5.phase5_validation,
        phase6_validation=phase6_validation,
        phase5_contract_checksum=phase5.mart_contract_checksum,
        phase6_contract_checksum=str(split_manifest.get("split_contract_checksum", "")),
        phase5_manifest_checksum=phase5.mart_manifest_checksum,
    )


def resolved_mart_name(path: Path) -> str:
    """Return a stable Phase 5 run identifier."""

    return path.resolve().name


def verify_frozen_membership(inputs: Phase7Inputs) -> dict[str, Any]:
    """Recalculate all frozen membership and TEST-lock hashes."""

    assignments = pd.read_parquet(inputs.split_dir / "split_assignments.parquet")
    errors: list[str] = []
    expected_assignment = inputs.split_manifest.get("split_assignment_sha256")
    actual_assignment = assignment_content_sha256(assignments)
    if actual_assignment != expected_assignment:
        errors.append("split_assignment_sha256 does not match the frozen Phase 6 manifest.")
    membership: dict[str, str] = {}
    for split in ("TRAIN", "VALIDATION", "TEST"):
        subset = assignments.loc[assignments["split"] == split]
        actual = claim_key_sha256(subset)
        membership[split] = actual
        if actual != inputs.split_manifest.get(f"{split.casefold()}_claim_key_sha256"):
            errors.append(f"{split} claim membership hash changed.")
    test = assignments.loc[assignments["split"] == "TEST"]
    test_hashes = {
        "ordered_test_claim_keys_sha256": claim_key_sha256(test),
        "unordered_test_claim_keys_sha256": unordered_claim_key_sha256(test),
        "test_assignment_content_sha256": assignment_content_sha256(test),
    }
    for key, actual in test_hashes.items():
        if actual != inputs.test_lock.get(key):
            errors.append(f"TEST lock field {key} changed.")
    return {
        "valid": not errors,
        "errors": errors,
        "split_assignment_sha256": actual_assignment,
        "membership": membership,
        "test_lock": test_hashes,
        "counts": assignments["split"].value_counts().to_dict(),
    }


def phase7_plan_check(
    mart_dir: Path,
    split_dir: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Run Phase 5, Phase 6, contract, source, and TEST-lock plan gates."""

    root = discover_repository_root(project_root)
    contract_result = validate_structured_feature_contract(root)
    errors = list(contract_result.get("errors", []))
    warnings = list(contract_result.get("warnings", []))
    inputs: Phase7Inputs | None = None
    try:
        inputs = load_phase7_inputs(mart_dir, split_dir, project_root=root)
        frozen = verify_frozen_membership(inputs)
        errors.extend(frozen["errors"])
        for artifact, columns in required_source_columns().items():
            frame = inputs.frames.get(artifact)
            if frame is None:
                errors.append(f"Required source artifact is missing: {artifact}")
                continue
            missing = sorted(set(columns) - set(frame.columns))
            if missing:
                errors.append(f"{artifact} is missing source columns: {', '.join(missing)}")
        warnings.extend(str(item) for item in inputs.phase5_validation.get("warnings", []))
        warnings.extend(str(item) for item in inputs.phase6_validation.get("warnings", []))
    except StructuredFeatureError as exc:
        errors.append(str(exc))
    status = "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS")
    return {
        "status": status,
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "contract": contract_result,
        "inputs": inputs,
    }


def required_source_columns() -> dict[str, tuple[str, ...]]:
    """Declare the source columns used by the deterministic builder."""

    return {
        "claim_snapshot": (
            "warranty_claim_key",
            "claim__claim_date",
            "claim__odometer_miles_at_failure",
            "claim__engine_hours_at_failure",
            "claim__months_in_service",
        ),
        "telemetry_history": ("current_warranty_claim_key", "telemetry__month_start_date"),
        "maintenance_history": ("current_warranty_claim_key", "maintenance__maintenance_date"),
        "service_history": ("current_warranty_claim_key", "service__service_date"),
        "component_installation_history": (
            "current_warranty_claim_key",
            "component_installation__installed_date",
        ),
        "prior_claim_history": ("current_warranty_claim_key", "prior_claim__claim_date"),
        "repair_history_index": ("current_warranty_claim_key",),
    }
