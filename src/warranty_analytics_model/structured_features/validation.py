"""Fail-closed validation for Phase 7 structured feature artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..feature_mart.manifest import content_sha256, sha256_file
from ..paths import discover_repository_root
from .contract import validate_structured_feature_contract
from .input import load_phase7_inputs, verify_frozen_membership
from .manifest import ordered_feature_frame
from .models import Phase7Inputs, StructuredFeatureError


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise StructuredFeatureError(f"Missing Phase 7 {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StructuredFeatureError(f"Phase 7 {label} must be an object.")
    return payload


def validate_feature_directory(
    feature_dir: Path,
    *,
    project_root: Path | None = None,
    inputs: Phase7Inputs | None = None,
) -> dict[str, Any]:
    """Validate hashes, membership, lineage, leakage, and numeric safety offline."""

    root = discover_repository_root(project_root)
    errors: list[str] = []
    warnings: list[str] = []
    directory = feature_dir.expanduser().resolve()
    try:
        manifest = _read_json(directory / "manifest.json", "manifest.json")
        lineage = _read_json(directory / "feature_lineage.json", "feature_lineage.json")
        inventory = _read_json(directory / "feature_manifest.json", "feature_manifest.json")
        frame = pd.read_parquet(directory / "structured_features.parquet")
    except (OSError, ValueError, ImportError, json.JSONDecodeError, StructuredFeatureError) as exc:
        return {"status": "BLOCKED", "errors": [str(exc)], "warnings": [], "checks": {}}
    contract_result = validate_structured_feature_contract(root)
    errors.extend(str(item) for item in contract_result.get("errors", []))
    required_columns = {"warranty_claim_key", "split", "claim__claim_date"}
    errors.extend(
        f"Missing required feature artifact column: {column}"
        for column in sorted(required_columns - set(frame.columns))
    )
    if "warranty_claim_key" in frame:
        if frame["warranty_claim_key"].isna().any():
            errors.append("Feature artifact contains null claim keys.")
        if frame["warranty_claim_key"].duplicated().any():
            errors.append("Feature artifact contains duplicate claim keys.")
    if "target__high_cost_claim_flag" in frame:
        errors.append("Target column is present in the Phase 7 feature artifact.")
    forbidden_columns = (
        "production_batch_id",
        "component_lot_no",
        "supplier_key",
        "service_center_key",
        "truck_key",
        "vin",
        "serial",
        "technician",
        "inspector",
        "eval__",
        "fingerprint",
        "cohort",
        "repair",
        "complaint_description",
        "diagnostic_summary",
        "failure_description",
        "repair_notes",
    )
    model_names = [name for name, item in lineage.items() if item.get("is_model_feature") is True]
    for name in model_names:
        if any(token in name.casefold() for token in forbidden_columns):
            errors.append(f"Prohibited/restricted source appears as model feature: {name}")
        if str(lineage[name].get("tier")) not in {"CORE", "EXTENDED"}:
            errors.append(f"Model feature has invalid tier: {name}")
        if lineage[name].get("target_dependent") is not False:
            errors.append(f"Model feature is target-dependent: {name}")
        if lineage[name].get("fitted_transformation") is not None:
            errors.append(f"Model feature has a fitted transformation: {name}")
        if str(lineage[name].get("feature_type")) == "date_control":
            errors.append(f"Raw date was incorrectly marked as a model feature: {name}")
    if set(lineage) != set(frame.columns):
        errors.append("Feature lineage does not cover exactly the artifact columns.")
    manifest_hash = manifest.get("artifact_content_sha256", {}).get("structured_features")
    actual_content = content_sha256(ordered_feature_frame(frame))
    if manifest_hash != actual_content:
        errors.append("Structured feature content SHA-256 does not match manifest.")
    file_hash = manifest.get("artifact_file_sha256", {}).get("structured_features")
    if file_hash != sha256_file(directory / "structured_features.parquet"):
        errors.append("Structured feature Parquet file SHA-256 does not match manifest.")
    try:
        if inputs is None:
            inputs = load_phase7_inputs(
                root / "artifacts" / "feature_mart" / str(manifest["input_phase5_mart"]["run"]),
                root / "artifacts" / "splits" / str(manifest["input_phase6_split"]["run"]),
                project_root=root,
            )
        frozen = verify_frozen_membership(inputs)
        errors.extend(frozen["errors"])
        assignments = pd.read_parquet(inputs.split_dir / "split_assignments.parquet")
        expected_keys = set(assignments["warranty_claim_key"])
        if set(frame["warranty_claim_key"]) != expected_keys:
            errors.append("Feature artifact claim membership differs from Phase 6 assignments.")
        joined = frame[["warranty_claim_key", "split"]].merge(
            assignments, on="warranty_claim_key", how="outer", indicator=True
        )
        if (joined["_merge"] != "both").any() or (joined["split_x"] != joined["split_y"]).any():
            errors.append("Feature artifact split membership differs from Phase 6 assignments.")
        expected_counts = assignments["split"].value_counts().to_dict()
        actual_counts = frame["split"].value_counts().to_dict()
        if expected_counts != actual_counts:
            errors.append("Feature artifact split counts do not match Phase 6.")
        if manifest.get("split_assignment_sha256") != inputs.split_manifest.get(
            "split_assignment_sha256"
        ):
            errors.append("Feature manifest split assignment hash is not frozen Phase 6 hash.")
        if manifest.get("test_lock_hashes", {}) != {
            key: inputs.test_lock.get(key)
            for key in (
                "ordered_test_claim_keys_sha256",
                "unordered_test_claim_keys_sha256",
                "test_assignment_content_sha256",
            )
        }:
            errors.append("Feature manifest TEST lock hashes changed.")
    except Exception as exc:
        errors.append(f"Phase 7 input compatibility validation failed: {exc}")
        inputs = None
    numeric_columns = [
        name for name in model_names if name in frame and pd.api.types.is_numeric_dtype(frame[name])
    ]
    positive_infinity = int(
        sum(
            np.isposinf(frame[name].to_numpy(dtype=float, na_value=np.nan)).sum()
            for name in numeric_columns
        )
    )
    negative_infinity = int(
        sum(
            np.isneginf(frame[name].to_numpy(dtype=float, na_value=np.nan)).sum()
            for name in numeric_columns
        )
    )
    if positive_infinity or negative_infinity:
        errors.append("Infinite numeric values are present in the feature artifact.")
    for item in (
        "all_null_train_features",
        "constant_train_features",
        "high_cardinality_categorical_warnings",
    ):
        values = _read_optional_quality(directory / "feature_quality.json", item)
        if values:
            warnings.append(f"{item}: {len(values)} warning(s).")
    status = "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS")
    return {
        "status": status,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "checks": {
            "manifest_valid": not errors,
            "lineage_valid": set(lineage) == set(frame.columns),
            "target_absent": TARGET not in frame.columns,
            "test_lock_valid": not any("TEST lock" in error for error in errors),
            "membership_valid": not any(
                "membership" in error or "split counts" in error for error in errors
            ),
            "content_hash_valid": manifest_hash == actual_content,
            "file_hash_valid": file_hash == sha256_file(directory / "structured_features.parquet"),
            "positive_infinity_count": positive_infinity,
            "negative_infinity_count": negative_infinity,
        },
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "model_feature_count": len(model_names),
        "feature_inventory": inventory,
        "input_phase5_validation": inputs.phase5_validation if inputs else {},
        "input_phase6_validation": inputs.phase6_validation if inputs else {},
    }


def _read_optional_quality(path: Path, key: str) -> list[Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get(key, [])
        return value if isinstance(value, list) else []
    except (OSError, json.JSONDecodeError):
        return []


TARGET = "target__high_cost_claim_flag"
