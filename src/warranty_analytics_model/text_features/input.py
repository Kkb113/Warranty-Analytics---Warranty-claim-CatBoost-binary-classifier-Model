"""Validated offline Phase 5, Phase 6, and hardened Phase 7 inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import discover_repository_root
from ..structured_features.input import load_phase7_inputs, verify_frozen_membership
from ..structured_features.models import Phase7Inputs
from ..structured_features.validation import validate_feature_directory
from .contract import validate_text_feature_contract
from .models import Phase8Inputs, TextFeatureError
from .source_policy import ALLOWED_PHASE8_TEXT_VALUE_SOURCES


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise TextFeatureError(f"Required Phase 8 {label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TextFeatureError(f"Phase 8 {label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TextFeatureError(f"Phase 8 {label} must contain a JSON object.")
    return payload


def _load_phase7_artifact(
    structured_dir: Path,
    *,
    project_root: Path,
    phase7_inputs: Phase7Inputs,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    required = (
        "structured_features.parquet",
        "feature_manifest.json",
        "feature_lineage.json",
        "manifest.json",
        "validation.json",
    )
    for name in required:
        if not (structured_dir / name).is_file():
            raise TextFeatureError(f"Required Phase 7 artifact is missing: {name}")
    validation = validate_feature_directory(
        structured_dir, project_root=project_root, inputs=phase7_inputs
    )
    if validation.get("errors"):
        raise TextFeatureError(
            "Phase 7 validation blocks Phase 8: " + "; ".join(map(str, validation["errors"]))
        )
    manifest = _read_json(structured_dir / "manifest.json", "Phase 7 manifest.json")
    lineage = _read_json(structured_dir / "feature_lineage.json", "Phase 7 feature_lineage.json")
    return manifest, lineage, validation


def load_phase8_inputs(
    mart_dir: Path,
    split_dir: Path,
    structured_dir: Path,
    *,
    project_root: Path | None = None,
) -> Phase8Inputs:
    """Load and validate the exact Phase 5/6/7 artifact chain offline."""

    root = discover_repository_root(project_root)
    try:
        phase7_inputs = load_phase7_inputs(mart_dir, split_dir, project_root=root)
    except Exception as exc:
        raise TextFeatureError(f"Phase 5/6 validation blocks Phase 8: {exc}") from exc
    resolved_structured = structured_dir.expanduser().resolve()
    phase7_manifest, phase7_lineage, phase7_validation = _load_phase7_artifact(
        resolved_structured, project_root=root, phase7_inputs=phase7_inputs
    )
    assignments_path = phase7_inputs.split_dir / "split_assignments.parquet"
    assignments = pd.read_parquet(assignments_path)
    frozen = verify_frozen_membership(phase7_inputs)
    if frozen["errors"]:
        raise TextFeatureError("Frozen split/TEST lock validation blocks Phase 8.")
    prior = phase7_inputs.frames.get("prior_claim_history")
    if prior is None:
        raise TextFeatureError("Phase 5 prior_claim_history is required for Phase 8.")
    missing = sorted(
        {
            "current_warranty_claim_key",
            "prior_warranty_claim_key",
            "prior_claim__claim_date",
            "prior_failure__failure_description",
        }
        - set(prior.columns)
    )
    if missing:
        raise TextFeatureError("prior_claim_history is missing: " + ", ".join(missing))
    if phase7_manifest.get("input_phase5_mart", {}).get("run") != phase7_inputs.mart_dir.name:
        raise TextFeatureError("Phase 7 does not consume the requested Phase 5 mart.")
    if phase7_manifest.get("input_phase6_split", {}).get("run") != phase7_inputs.split_dir.name:
        raise TextFeatureError("Phase 7 does not consume the requested Phase 6 split.")
    if phase7_manifest.get("target_column_present") is not False:
        raise TextFeatureError("Phase 7 target presence blocks Phase 8.")
    if set(ALLOWED_PHASE8_TEXT_VALUE_SOURCES) != {"prior_failure__failure_description"}:
        raise TextFeatureError("Phase 8 allowlist is not the single approved historical source.")
    return Phase8Inputs(
        root=root,
        mart_dir=phase7_inputs.mart_dir,
        split_dir=phase7_inputs.split_dir,
        structured_dir=resolved_structured,
        prior_claim_history=prior.copy(),
        assignments=assignments,
        phase5_manifest=phase7_inputs.mart_manifest,
        phase6_manifest=phase7_inputs.split_manifest,
        test_lock=phase7_inputs.test_lock,
        phase7_manifest=phase7_manifest,
        phase7_lineage=phase7_lineage,
        phase5_validation=phase7_inputs.phase5_validation,
        phase6_validation=phase7_inputs.phase6_validation,
        phase7_validation=phase7_validation,
        phase5_contract_checksum=phase7_inputs.phase5_contract_checksum,
        phase6_contract_checksum=phase7_inputs.phase6_contract_checksum,
        phase5_manifest_checksum=phase7_inputs.phase5_manifest_checksum,
        phase7_contract_checksum=str(phase7_manifest.get("phase7_contract_checksum", "")),
        phase7_content_sha256=str(
            phase7_manifest.get("artifact_content_sha256", {}).get("structured_features", "")
        ),
    )


def phase8_plan_check(
    mart_dir: Path,
    split_dir: Path,
    structured_dir: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Run all Phase 8 pre-build gates without creating a text artifact."""

    root = discover_repository_root(project_root)
    contract = validate_text_feature_contract(root)
    errors = list(contract.get("errors", []))
    warnings = list(contract.get("warnings", []))
    inputs: Phase8Inputs | None = None
    try:
        inputs = load_phase8_inputs(mart_dir, split_dir, structured_dir, project_root=root)
        warnings.extend(str(item) for item in inputs.phase5_validation.get("warnings", []))
        warnings.extend(str(item) for item in inputs.phase6_validation.get("warnings", []))
        warnings.extend(str(item) for item in inputs.phase7_validation.get("warnings", []))
        assignments = inputs.assignments
        if len(assignments) != 8500:
            errors.append("Phase 8 requires exactly 8,500 split assignments.")
        if assignments["warranty_claim_key"].duplicated().any():
            errors.append("Phase 6 assignments contain duplicate claim keys.")
    except TextFeatureError as exc:
        errors.append(str(exc))
    return {
        "status": "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS"),
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "contract": contract,
        "inputs": inputs,
    }
