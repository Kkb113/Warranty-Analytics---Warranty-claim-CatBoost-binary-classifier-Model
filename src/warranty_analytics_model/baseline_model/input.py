"""Independent validation of the locked Phase 5–8 input chain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import discover_repository_root
from ..splits.manifest import (
    assignment_content_sha256,
    claim_key_sha256,
    unordered_claim_key_sha256,
)
from ..text_features.contract import load_text_feature_contract
from ..text_features.input import load_phase8_inputs
from ..text_features.validation import validate_text_directory
from .contract import validate_baseline_contract
from .feature_sets import audit_phase8_sources, resolve_feature_sets
from .models import BaselineModelError, Phase9Inputs

EXPECTED_PHASE7_CONTENT_SHA256 = "d487536f750b2cbd84d9d6d6221366410b8434d905f4cdc0dec917f8535f7f93"
EXPECTED_PHASE8_CONTENT_SHA256 = "a6a64d5820a0d1fcd97058a412685ad2e26f30605dd3a558ec427d6158636b71"
EXPECTED_PHASE8_CONTRACT_SHA256 = "392d95b81a498c84136a8b6b1927cdfd177dca3d3b028ac0d80353f73c3acc75"
ALLOWED_PHASE7_POLICIES = {"ALLOW_BASELINE_POC", "ALLOW_HISTORICAL_POC"}
PROHIBITED_SOURCE_SUFFIXES = {
    "high_cost_claim_flag",
    "total_claim_cost",
    "labor_cost",
    "parts_cost",
    "diagnostic_cost",
    "towing_cost",
    "other_cost",
    "approved_amount",
    "rejected_amount",
    "customer_paid_amount",
    "repair_end_date",
    "days_to_repair",
    "claim_status",
    "root_cause_category",
    "repeat_claim_flag",
    "potential_recall_flag",
}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise BaselineModelError(f"Required Phase 9 {label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineModelError(f"Phase 9 {label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise BaselineModelError(f"Phase 9 {label} must contain a JSON object.")
    return payload


def _verify_frozen_membership(
    assignments: pd.DataFrame,
    split_manifest: dict[str, Any],
    test_lock: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    actual_assignment = assignment_content_sha256(assignments)
    if actual_assignment != split_manifest.get("split_assignment_sha256"):
        errors.append("Phase 9 split assignment hash differs from Phase 6.")
    membership: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in ("TRAIN", "VALIDATION", "TEST"):
        subset = assignments.loc[assignments["split"] == split]
        counts[split] = len(subset)
        membership[split] = claim_key_sha256(subset)
        if membership[split] != split_manifest.get(f"{split.casefold()}_claim_key_sha256"):
            errors.append(f"Phase 9 {split} membership hash differs from Phase 6.")
    test = assignments.loc[assignments["split"] == "TEST"]
    test_hashes = {
        "ordered_test_claim_keys_sha256": claim_key_sha256(test),
        "unordered_test_claim_keys_sha256": unordered_claim_key_sha256(test),
        "test_assignment_content_sha256": assignment_content_sha256(test),
    }
    for key, actual in test_hashes.items():
        if actual != test_lock.get(key):
            errors.append(f"Phase 9 TEST lock field changed: {key}")
    if sum(counts.values()) != len(assignments):
        errors.append("Phase 9 assignments contain an unsupported split label.")
    return {
        "valid": not errors,
        "errors": errors,
        "split_assignment_sha256": actual_assignment,
        "membership": membership,
        "test_lock": test_hashes,
        "counts": counts,
        "total_count": len(assignments),
    }


def _validate_feature_membership(
    assignments: pd.DataFrame,
    frame: pd.DataFrame,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if "warranty_claim_key" not in frame or frame["warranty_claim_key"].duplicated().any():
        return [f"{label} claim keys are missing or duplicated."]
    expected = set(assignments["warranty_claim_key"].astype(int))
    actual = set(frame["warranty_claim_key"].astype(int))
    if expected != actual:
        errors.append(f"{label} membership differs from Phase 6 assignments.")
    if "split" not in frame:
        errors.append(f"{label} split control is missing.")
    else:
        joined = assignments[["warranty_claim_key", "split"]].merge(
            frame[["warranty_claim_key", "split"]],
            on="warranty_claim_key",
            how="outer",
            suffixes=("_expected", "_actual"),
            validate="one_to_one",
        )
        if (joined["split_expected"] != joined["split_actual"]).any():
            errors.append(f"{label} split controls differ from Phase 6 assignments.")
    return errors


def _audit_phase7_sources(lineage: dict[str, dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    model_features = 0
    prohibited_sources = 0
    target_dependent = 0
    raw_identifier_features = 0
    for name, item in lineage.items():
        if item.get("is_model_feature") is not True:
            continue
        model_features += 1
        if item.get("is_control") is True:
            raw_identifier_features += 1
            errors.append(f"Phase 7 control is marked as a model feature: {name}")
        if item.get("target_dependent") is not False:
            target_dependent += 1
            errors.append(f"Phase 7 model feature is target-dependent: {name}")
        if item.get("phase4_source_policy") not in ALLOWED_PHASE7_POLICIES:
            errors.append(f"Phase 7 feature has an unapproved source policy: {name}")
        for source in item.get("value_sources", []):
            suffix = str(source).split("__")[-1]
            if suffix in PROHIBITED_SOURCE_SUFFIXES:
                prohibited_sources += 1
                errors.append(f"Phase 7 feature uses prohibited current outcome source: {name}")
    return {
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "model_feature_count": model_features,
        "prohibited_source_count": prohibited_sources,
        "target_dependent_feature_count": target_dependent,
        "raw_identifier_feature_count": raw_identifier_features,
    }


def load_phase9_inputs(
    mart_dir: Path,
    split_dir: Path,
    structured_dir: Path,
    text_dir: Path,
    *,
    project_root: Path | None = None,
) -> Phase9Inputs:
    """Load and independently validate the exact Phase 5–8 chain offline."""

    root = discover_repository_root(project_root)
    try:
        phase8_inputs = load_phase8_inputs(mart_dir, split_dir, structured_dir, project_root=root)
    except Exception as exc:
        raise BaselineModelError(f"Phase 5–7 validation blocks Phase 9: {exc}") from exc
    resolved_text = text_dir.expanduser().resolve()
    phase8_validation = validate_text_directory(resolved_text, project_root=root)
    if phase8_validation.get("errors"):
        raise BaselineModelError(
            "Phase 8 validation blocks Phase 9: " + "; ".join(map(str, phase8_validation["errors"]))
        )
    phase7_manifest = phase8_inputs.phase7_manifest
    phase8_manifest = _read_json(resolved_text / "manifest.json", "Phase 8 manifest")
    phase7_lineage = {
        str(key): value
        for key, value in phase8_inputs.phase7_lineage.items()
        if isinstance(value, dict)
    }
    phase8_lineage_raw = _read_json(resolved_text / "text_feature_lineage.json", "Phase 8 lineage")
    phase8_lineage = {
        str(key): value for key, value in phase8_lineage_raw.items() if isinstance(value, dict)
    }
    structured_features = pd.read_parquet(
        phase8_inputs.structured_dir / "structured_features.parquet"
    )
    text_features = pd.read_parquet(resolved_text / "text_features.parquet")
    assignments = phase8_inputs.assignments.copy()
    errors: list[str] = []
    if (
        phase7_manifest.get("artifact_content_sha256", {}).get("structured_features")
        != EXPECTED_PHASE7_CONTENT_SHA256
    ):
        errors.append("Locked Phase 7 content SHA-256 differs from the approved Phase 9 input.")
    if (
        phase8_manifest.get("artifact_content_sha256", {}).get("text_features")
        != EXPECTED_PHASE8_CONTENT_SHA256
    ):
        errors.append("Locked Phase 8 content SHA-256 differs from the approved Phase 9 input.")
    if phase8_manifest.get("input_phase5_run") != phase8_inputs.mart_dir.name:
        errors.append("Phase 8 does not reference the requested Phase 5 run.")
    if phase8_manifest.get("input_phase6_run") != phase8_inputs.split_dir.name:
        errors.append("Phase 8 does not reference the requested Phase 6 run.")
    if phase8_manifest.get("input_phase7_run") != phase8_inputs.structured_dir.name:
        errors.append("Phase 8 does not reference the requested Phase 7 run.")
    phase8_contract, phase8_contract_checksum = load_text_feature_contract(root)
    if phase8_contract_checksum != EXPECTED_PHASE8_CONTRACT_SHA256:
        errors.append("Phase 8 contract SHA-256 differs from the approved Phase 9 input.")
    source_audit = audit_phase8_sources(phase8_lineage, phase8_contract)
    errors.extend(source_audit["errors"])
    phase7_source_audit = _audit_phase7_sources(phase7_lineage)
    errors.extend(phase7_source_audit["errors"])
    frozen = _verify_frozen_membership(
        assignments, phase8_inputs.phase6_manifest, phase8_inputs.test_lock
    )
    errors.extend(frozen["errors"])
    errors.extend(_validate_feature_membership(assignments, structured_features, "Phase 7"))
    errors.extend(_validate_feature_membership(assignments, text_features, "Phase 8"))
    if errors:
        raise BaselineModelError(
            "Phase 9 input validation failed: " + "; ".join(dict.fromkeys(errors))
        )
    source_audit = {**source_audit, "phase7": phase7_source_audit}
    return Phase9Inputs(
        root=root,
        mart_dir=phase8_inputs.mart_dir,
        split_dir=phase8_inputs.split_dir,
        structured_dir=phase8_inputs.structured_dir,
        text_dir=resolved_text,
        assignments=assignments,
        structured_features=structured_features,
        text_features=text_features,
        phase7_lineage=phase7_lineage,
        phase8_lineage=phase8_lineage,
        phase5_manifest=phase8_inputs.phase5_manifest,
        phase6_manifest=phase8_inputs.phase6_manifest,
        phase7_manifest=phase7_manifest,
        phase8_manifest=phase8_manifest,
        test_lock=phase8_inputs.test_lock,
        upstream_validations={
            "phase5": phase8_inputs.phase5_validation,
            "phase6": phase8_inputs.phase6_validation,
            "phase7": phase8_inputs.phase7_validation,
            "phase8": phase8_validation,
        },
        frozen_membership=frozen,
        source_audit=source_audit,
    )


def phase9_plan_check(
    mart_dir: Path,
    split_dir: Path,
    structured_dir: Path,
    text_dir: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Run all input, TEST-lock, source, and feature-set gates without training."""

    root = discover_repository_root(project_root)
    contract = validate_baseline_contract(root)
    errors = list(contract.get("errors", []))
    warnings = list(contract.get("warnings", []))
    inputs: Phase9Inputs | None = None
    feature_sets: dict[str, Any] = {}
    try:
        inputs = load_phase9_inputs(
            mart_dir,
            split_dir,
            structured_dir,
            text_dir,
            project_root=root,
        )
        feature_sets = resolve_feature_sets(inputs.phase7_lineage, inputs.phase8_lineage)
        for validation in inputs.upstream_validations.values():
            warnings.extend(str(item) for item in validation.get("warnings", []))
    except BaselineModelError as exc:
        errors.append(str(exc))
    return {
        "status": "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS"),
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "contract": contract,
        "inputs": inputs,
        "feature_sets": feature_sets,
    }
