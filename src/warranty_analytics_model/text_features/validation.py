"""Fail-closed validation for Phase 8 text artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..feature_mart.manifest import content_sha256, sha256_file
from ..paths import discover_repository_root
from .config import load_text_feature_settings
from .documents import build_historical_documents
from .input import load_phase8_inputs
from .lexical import build_lexical_features
from .manifest import ordered_text_frame
from .models import TextFeatureError
from .source_policy import validate_text_lineage_sources

TARGET = "target__high_cost_claim_flag"
CONTROL_COLUMNS = {"warranty_claim_key", "split", "claim__claim_date"}
DOCUMENT_PREFIX = "prior_failure_text__"
PROHIBITED_NAME_TOKENS = (
    "complaint",
    "diagnostic",
    "technician",
    "repair_notes",
    "current_failure",
    "target",
    "supplier_key",
    "warranty_claim_key",
)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise TextFeatureError(f"Missing Phase 8 {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TextFeatureError(f"Phase 8 {label} must be an object.")
    return payload


def _report_exposes_raw_text(report_dir: Path, descriptions: list[str]) -> bool:
    if not report_dir.is_dir():
        return False
    content = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in report_dir.rglob("*")
        if path.is_file()
    ).casefold()
    return any(
        len(description) >= 8 and description.casefold() in content for description in descriptions
    )


def validate_text_directory(
    text_dir: Path,
    *,
    project_root: Path | None = None,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate hashes, lineage, determinism, temporal safety, and frozen membership."""

    root = discover_repository_root(project_root)
    directory = text_dir.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        manifest = _read_json(directory / "manifest.json", "manifest.json")
        feature_manifest = _read_json(
            directory / "text_feature_manifest.json", "text_feature_manifest.json"
        )
        lineage = _read_json(directory / "text_feature_lineage.json", "text_feature_lineage.json")
        quality = _read_json(directory / "text_quality.json", "text_quality.json")
        frame = pd.read_parquet(directory / "text_features.parquet")
    except (OSError, ValueError, ImportError, json.JSONDecodeError, TextFeatureError) as exc:
        return {"status": "BLOCKED", "errors": [str(exc)], "warnings": [], "checks": {}}
    try:
        mart_dir = root / "artifacts" / "feature_mart" / str(manifest["input_phase5_run"])
        split_dir = root / "artifacts" / "splits" / str(manifest["input_phase6_run"])
        structured_dir = (
            root / "artifacts" / "structured_features" / str(manifest["input_phase7_run"])
        )
        inputs = load_phase8_inputs(mart_dir, split_dir, structured_dir, project_root=root)
    except Exception as exc:
        return {"status": "BLOCKED", "errors": [str(exc)], "warnings": [], "checks": {}}

    if TARGET in frame.columns:
        errors.append("Target column is present in Phase 8 text artifact.")
    if len(frame) != 8500:
        errors.append("Phase 8 text artifact must contain exactly 8,500 rows.")
    if "warranty_claim_key" not in frame or frame["warranty_claim_key"].duplicated().any():
        errors.append("Phase 8 text artifact claim keys are missing or duplicated.")
    expected_columns = set(feature_manifest.get("feature_names", [])) | CONTROL_COLUMNS
    if expected_columns != set(frame.columns):
        errors.append("Phase 8 feature manifest does not cover exactly the artifact columns.")
    structured_columns = set(
        pd.read_parquet(structured_dir / "structured_features.parquet", columns=[]).columns
    )
    structured_lineage = _read_json(structured_dir / "feature_lineage.json", "Phase 7 lineage")
    structured_model_columns = {
        name for name, item in structured_lineage.items() if item.get("is_model_feature") is True
    }
    if set(frame.columns) & structured_model_columns:
        errors.append("Phase 8 duplicates Phase 7 structured model features.")
    if structured_columns and set(frame.columns) & structured_columns:
        errors.append("Phase 8 artifact overlaps Phase 7 structured columns.")
    source_policy = validate_text_lineage_sources(lineage)
    errors.extend(str(item) for item in source_policy.get("errors", []))
    if set(lineage) != set(frame.columns):
        errors.append("Phase 8 lineage does not cover exactly the artifact columns.")
    for name, item in lineage.items():
        if name in CONTROL_COLUMNS:
            if item.get("is_model_feature") is not False or item.get("is_control") is not True:
                errors.append(f"Phase 8 control metadata is invalid: {name}")
        elif item.get("is_model_feature") is not True:
            errors.append(f"Phase 8 candidate is not marked as a model feature: {name}")
        if item.get("target_dependent") is not False:
            errors.append(f"Phase 8 feature is target-dependent: {name}")
        if item.get("fitted_transformation") is not None:
            errors.append(f"Phase 8 feature has a fitted transformation: {name}")
        if name not in CONTROL_COLUMNS and any(
            token in name.casefold() for token in PROHIBITED_NAME_TOKENS
        ):
            errors.append(f"Prohibited text feature name appears in Phase 8 artifact: {name}")
    for name in frame:
        if name.startswith(DOCUMENT_PREFIX) and not (
            pd.api.types.is_object_dtype(frame[name]) or pd.api.types.is_string_dtype(frame[name])
        ):
            errors.append(f"Raw text candidate has invalid dtype: {name}")
        if (
            name.startswith("text__")
            and name != "text__has_prior_failure_description"
            and not pd.api.types.is_numeric_dtype(frame[name])
        ):
            errors.append(f"Lexical candidate has invalid dtype: {name}")
    ordered = ordered_text_frame(frame)
    content_hash = content_sha256(ordered)
    actual_file_hash = sha256_file(directory / "text_features.parquet")
    if manifest.get("artifact_content_sha256", {}).get("text_features") != content_hash:
        errors.append("Phase 8 content SHA-256 does not match the manifest.")
    if manifest.get("artifact_file_sha256", {}).get("text_features") != actual_file_hash:
        errors.append("Phase 8 Parquet file SHA-256 does not match the manifest.")
    frozen = {
        "split_assignment_sha256": inputs.phase6_manifest.get("split_assignment_sha256"),
        "train_claim_key_sha256": inputs.phase6_manifest.get("train_claim_key_sha256"),
        "validation_claim_key_sha256": inputs.phase6_manifest.get("validation_claim_key_sha256"),
        "test_claim_key_sha256": inputs.phase6_manifest.get("test_claim_key_sha256"),
    }
    for field, expected in frozen.items():
        if manifest.get(field) != expected:
            errors.append(f"Phase 8 frozen membership field changed: {field}")
    for field, expected in inputs.test_lock.items():
        if (
            field
            in {
                "ordered_test_claim_keys_sha256",
                "unordered_test_claim_keys_sha256",
                "test_assignment_content_sha256",
            }
            and manifest.get("test_lock_hashes", {}).get(field) != expected
        ):
            errors.append(f"Phase 8 TEST lock field changed: {field}")
    joined = frame[["warranty_claim_key", "split"]].merge(
        inputs.assignments[["warranty_claim_key", "split"]],
        on="warranty_claim_key",
        how="outer",
        suffixes=("_text", "_split"),
        indicator=True,
    )
    if (joined["_merge"] != "both").any() or (joined["split_text"] != joined["split_split"]).any():
        errors.append("Phase 8 membership or split assignment differs from Phase 6.")
    settings = load_text_feature_settings(root)
    rebuilt_documents, temporal = build_historical_documents(inputs, settings)
    rebuilt = build_lexical_features(rebuilt_documents, settings).frame
    rebuilt_hash = content_sha256(ordered_text_frame(rebuilt))
    if rebuilt_hash != content_hash:
        errors.append("Phase 8 document/lexical construction is not deterministic.")
    if temporal.get("same_day_text_records") or temporal.get("future_text_records"):
        errors.append("Phase 8 contains same-day or future historical text records.")
    raw_descriptions = [
        str(value)
        for value in inputs.prior_claim_history["prior_failure__failure_description"]
        .dropna()
        .tolist()
    ]
    exposed = _report_exposes_raw_text(
        report_dir or (root / "reports" / "phase8_text_features" / directory.name), raw_descriptions
    )
    if exposed:
        errors.append("Phase 8 reports contain raw historical text.")
    warnings.extend(str(item) for item in quality.get("train_feature_warnings", []))
    warnings.append("UNVERSIONED_FAILURE_DESCRIPTION_DIMENSION: real-data reapproval is required.")
    checks = {
        "target_absent": TARGET not in frame,
        "source_policy_valid": source_policy.get("valid", False),
        "lineage_valid": set(lineage) == set(frame.columns),
        "membership_valid": not any("membership" in error for error in errors),
        "test_lock_valid": not any("TEST lock" in error for error in errors),
        "content_hash_valid": manifest.get("artifact_content_sha256", {}).get("text_features")
        == content_hash,
        "file_hash_valid": manifest.get("artifact_file_sha256", {}).get("text_features")
        == actual_file_hash,
        "deterministic_rebuild_valid": rebuilt_hash == content_hash,
        "raw_text_report_exposure": exposed,
        "same_day_text_records": int(temporal.get("same_day_text_records", 0)),
        "future_text_records": int(temporal.get("future_text_records", 0)),
    }
    status = "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS")
    return {
        "status": status,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "checks": checks,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "text_feature_count": int(manifest.get("text_feature_count", 0)),
        "lexical_feature_count": int(manifest.get("lexical_feature_count", 0)),
        "temporal_audit": temporal,
        "leakage_audit": {
            "target_sources": 0,
            "current_claim_text_sources": 0,
            "prohibited_sources": 0,
            "confirmation_sources": 0,
            "restricted_sources": 0,
            "raw_id_text_sources": 0,
        },
        "content_sha256": content_hash,
        "file_sha256": actual_file_hash,
    }
