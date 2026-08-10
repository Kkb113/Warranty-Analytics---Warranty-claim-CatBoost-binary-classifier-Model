"""Deterministic Phase 7 manifests, lineage, and aggregate diagnostics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .. import __version__
from ..feature_mart.manifest import (
    content_sha256,
    git_commit_sha,
    sha256_file,
    write_json,
)
from ..feature_mart.models import MartContract
from ..splits.manifest import claim_key_sha256
from .models import FeatureDefinition, Phase7Inputs, StructuredFeatureSettings


def ordered_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize row order by claim key without changing column order."""

    return frame.sort_values("warranty_claim_key", kind="mergesort").reset_index(drop=True)


def definition_payload(definitions: list[FeatureDefinition]) -> dict[str, dict[str, Any]]:
    """Build a stable per-column lineage mapping."""

    return {
        item.feature_name: item.as_dict()
        for item in sorted(definitions, key=lambda value: value.feature_name)
    }


def feature_manifest(definitions: list[FeatureDefinition]) -> dict[str, Any]:
    """Summarize model features by tier, type, and family."""

    model = [item for item in definitions if item.is_model_feature]
    by_family = Counter(item.family for item in model)
    by_type = Counter(item.feature_type for item in model)
    return {
        "total_feature_count": len(model),
        "core_feature_count": sum(item.tier == "CORE" for item in model),
        "extended_feature_count": sum(item.tier == "EXTENDED" for item in model),
        "numeric_count": by_type.get("numeric", 0),
        "categorical_count": by_type.get("categorical", 0),
        "boolean_count": by_type.get("boolean", 0),
        "date_control_count": sum(item.feature_type == "date_control" for item in definitions),
        "feature_count_by_family": {
            family: by_family.get(family, 0)
            for family in (
                "direct",
                "lifecycle",
                "usage",
                "warranty",
                "telemetry",
                "maintenance",
                "service",
                "component",
                "prior_claim",
                "history_coverage",
            )
        },
        "control_count": sum(item.is_control for item in definitions),
        "lineage_count": sum(item.is_lineage for item in definitions),
    }


def source_coverage(
    definitions: list[FeatureDefinition], mart_contract: MartContract
) -> dict[str, Any]:
    """Account for every Phase 5 direct and safe historical source field."""

    used = {
        source
        for item in definitions
        for source in (*item.source_columns, *item.value_sources, *item.control_sources)
    }
    direct: list[dict[str, Any]] = []
    for mapping in mart_contract.direct_feature_mappings:
        status = (
            "USED"
            if mapping.output_column in used
            or mapping.output_column in {item.feature_name for item in definitions}
            else "DEFERRED_WITH_REASON"
        )
        reason = (
            "raw date retained as control"
            if mapping.output_column
            in {
                "claim__claim_date",
                "truck__build_date",
                "truck__delivery_date",
                "truck__in_service_date",
            }
            else "not carried into the structured feature artifact"
        )
        direct.append(
            {
                "field": mapping.output_column,
                "policy": mapping.policy,
                "status": status,
                "reason": reason,
            }
        )
    historical: list[dict[str, Any]] = []
    for bridge in mart_contract.historical_bridge_definitions:
        for mapping in bridge.field_mappings:
            field = mapping.output_column
            if field == "prior_failure__failure_description":
                status = "DEFERRED_WITH_REASON"
                reason = "text-like failure description is deferred to Phase 8"
            elif field in used:
                status = "USED"
                reason = "materialized into one or more deterministic structured features"
            else:
                status = "DEFERRED_WITH_REASON"
                reason = "safe value requires an explicit Phase 7 representation"
            historical.append(
                {"artifact": bridge.artifact, "field": field, "status": status, "reason": reason}
            )
    deferred = [
        {
            "source": "prior_failure__failure_description",
            "reason": "deferred to Phase 8 text modeling",
        },
        {"source": "repair_history_index", "reason": "control/eligibility only; no model features"},
        {
            "source": "restricted identifiers",
            "reason": "retained only in Phase 6 evaluation metadata",
        },
    ]
    return {"direct": direct, "historical": historical, "deferred": deferred}


def quality_diagnostics(
    frame: pd.DataFrame, definitions: list[FeatureDefinition]
) -> dict[str, Any]:
    """Generate aggregate TRAIN-based feature quality diagnostics."""

    train = frame.loc[frame["split"] == "TRAIN"]
    diagnostics: dict[str, Any] = {}
    all_null: list[str] = []
    constant: list[str] = []
    high_cardinality: list[str] = []
    for item in definitions:
        if not item.is_model_feature:
            continue
        values = train[item.feature_name]
        entry: dict[str, Any] = {
            "feature_type": item.feature_type,
            "null_count": int(values.isna().sum()),
            "null_percentage": round(float(values.isna().mean() * 100), 6),
            "unique_count": int(values.nunique(dropna=True)),
        }
        if item.feature_type == "numeric":
            numeric_values = pd.to_numeric(values, errors="coerce")
            entry.update(
                {
                    "min": numeric_values.min(),
                    "max": numeric_values.max(),
                    "mean": numeric_values.mean(),
                    "std": numeric_values.std(ddof=0),
                }
            )
        else:
            entry["cardinality"] = int(values.nunique(dropna=True))
            if item.feature_type == "categorical" and values.nunique(dropna=True) > max(
                100, len(train) * 0.5
            ):
                high_cardinality.append(item.feature_name)
        if values.dropna().empty:
            all_null.append(item.feature_name)
        elif values.dropna().nunique() <= 1:
            constant.append(item.feature_name)
        diagnostics[item.feature_name] = entry
    return {
        "train_row_count": int(len(train)),
        "features": diagnostics,
        "all_null_train_features": all_null,
        "constant_train_features": constant,
        "high_cardinality_categorical_warnings": high_cardinality,
    }


def build_run_manifest(
    *,
    root: Path,
    inputs: Phase7Inputs,
    contract_checksum: str,
    settings: StructuredFeatureSettings,
    frame: pd.DataFrame,
    definitions: list[FeatureDefinition],
    parquet_metadata: dict[str, Any],
    validation_status: str,
    warnings: list[str],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    """Create the aggregate manifest required by the Phase 7 contract."""

    assignments = pd.read_parquet(inputs.split_dir / "split_assignments.parquet")
    feature_summary = feature_manifest(definitions)
    split_counts = frame["split"].value_counts().to_dict()
    split_manifest = inputs.split_manifest
    return {
        "phase7_version": "1.0.0",
        "package_version": __version__,
        "git_commit": git_commit_sha(root),
        "phase7_contract_checksum": contract_checksum,
        "input_phase5_mart": {
            "run": inputs.mart_dir.name,
            "manifest_checksum": inputs.phase5_manifest_checksum,
            "contract_checksum": inputs.phase5_contract_checksum,
            "artifact_file_sha256": inputs.mart_manifest.get("artifact_file_sha256", {}),
            "artifact_content_sha256": inputs.mart_manifest.get(
                "artifact_content_fingerprints", {}
            ),
        },
        "input_phase5_checksums": {
            "claim_snapshot_content_sha256": inputs.mart_manifest.get(
                "artifact_content_fingerprints", {}
            ).get("claim_snapshot"),
            "history_content_sha256": {
                key: value
                for key, value in inputs.mart_manifest.get(
                    "artifact_content_fingerprints", {}
                ).items()
                if key.endswith("_history") or key == "repair_history_index"
            },
        },
        "input_phase6_split": {
            "run": inputs.split_dir.name,
            "contract_checksum": inputs.phase6_contract_checksum,
        },
        "input_split_checksum": inputs.phase6_contract_checksum,
        "split_assignment_sha256": split_manifest.get("split_assignment_sha256"),
        "train_claim_key_sha256": split_manifest.get("train_claim_key_sha256"),
        "validation_claim_key_sha256": split_manifest.get("validation_claim_key_sha256"),
        "test_claim_key_sha256": split_manifest.get("test_claim_key_sha256"),
        "test_lock_hashes": {
            key: inputs.test_lock.get(key)
            for key in (
                "ordered_test_claim_keys_sha256",
                "unordered_test_claim_keys_sha256",
                "test_assignment_content_sha256",
            )
        },
        "claim_snapshot_content_sha256": inputs.mart_manifest.get(
            "artifact_content_fingerprints", {}
        ).get("claim_snapshot"),
        "row_count": int(len(frame)),
        "train_count": int(split_counts.get("TRAIN", 0)),
        "validation_count": int(split_counts.get("VALIDATION", 0)),
        "test_count": int(split_counts.get("TEST", 0)),
        "feature_count": feature_summary["total_feature_count"],
        "core_feature_count": feature_summary["core_feature_count"],
        "extended_feature_count": feature_summary["extended_feature_count"],
        "feature_manifest": feature_summary,
        "settings": {
            "windows_months": list(settings.windows_months),
            "include_all_history": settings.include_all_history,
            "std_min_observations": settings.std_min_observations,
            "slope_min_observations": settings.slope_min_observations,
            "compression": settings.compression,
        },
        "artifact_paths": {
            "structured_features": "structured_features.parquet",
            "feature_manifest": "feature_manifest.json",
            "feature_lineage": "feature_lineage.json",
            "feature_quality": "feature_quality.json",
            "validation": "validation.json",
        },
        "artifact_file_sha256": {"structured_features": parquet_metadata.get("file_sha256")},
        "artifact_content_sha256": {"structured_features": parquet_metadata.get("content_sha256")},
        "source_coverage": coverage,
        "warnings": list(dict.fromkeys(warnings)),
        "validation_status": validation_status,
        "target_column_present": False,
        "target_reserved_for_phase": 15,
        "test_based_tuning": False,
        "model_training": False,
        "feature_selection": False,
        "text_features": False,
        "global_fitted_transformations": [],
        "frozen_assignment_counts": {key: int(value) for key, value in split_counts.items()},
        "frozen_assignment_content_sha256": {
            split: claim_key_sha256(assignments.loc[assignments["split"] == split])
            for split in ("TRAIN", "VALIDATION", "TEST")
        },
    }


def write_feature_artifacts(
    directory: Path,
    *,
    frame: pd.DataFrame,
    definitions: list[FeatureDefinition],
    settings: StructuredFeatureSettings,
) -> dict[str, Any]:
    """Write the canonical matrix, lineage, manifest, and TRAIN diagnostics."""

    directory.mkdir(parents=True, exist_ok=True)
    canonical = ordered_feature_frame(frame)
    metadata = {
        "file_sha256": "",
        "content_sha256": content_sha256(canonical),
        "row_count": int(len(canonical)),
        "column_count": int(len(canonical.columns)),
    }
    path = directory / "structured_features.parquet"
    canonical.to_parquet(
        path,
        index=False,
        compression=None if settings.compression == "none" else settings.compression,
    )
    metadata["file_sha256"] = sha256_file(path)
    write_json(directory / "feature_lineage.json", definition_payload(definitions))
    write_json(directory / "feature_manifest.json", feature_manifest(definitions))
    write_json(directory / "feature_quality.json", quality_diagnostics(canonical, definitions))
    return metadata
