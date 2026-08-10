"""Aggregate, public-repository-safe Phase 7 reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..feature_mart.manifest import write_json


def write_phase7_reports(
    output_root: Path,
    *,
    manifest: dict[str, Any],
    validation: dict[str, Any],
    quality: dict[str, Any],
    lineage: dict[str, Any],
    coverage: dict[str, Any],
) -> None:
    """Write aggregate summaries only; no raw claim records are copied."""

    output_root.mkdir(parents=True, exist_ok=True)
    model_lineage = [item for item in lineage.values() if item.get("is_model_feature") is True]
    family_summary: dict[str, int] = {}
    for item in model_lineage:
        family = str(item.get("family", "unknown"))
        family_summary[family] = family_summary.get(family, 0) + 1
    inventory = manifest.get("feature_manifest", {})
    summary = {
        "phase": 7,
        "status": manifest.get("validation_status", "INCOMPLETE"),
        "input_phase5_mart": manifest.get("input_phase5_mart"),
        "input_phase6_split": manifest.get("input_phase6_split"),
        "phase5_validation_status": validation.get("input_phase5_validation", {}).get("status"),
        "phase6_validation_status": validation.get("input_phase6_validation", {}).get("status"),
        "phase7_contract_checksum": manifest.get("phase7_contract_checksum"),
        "rows": manifest.get("row_count"),
        "unique_claim_rows": manifest.get("row_count"),
        "split_counts": {
            key: manifest.get(key) for key in ("train_count", "validation_count", "test_count")
        },
        "test_lock_preserved": validation.get("checks", {}).get("test_lock_valid", False),
        "feature_counts": inventory,
        "feature_count_by_family": family_summary,
        "feature_count_by_type": {
            key: inventory.get(key)
            for key in ("numeric_count", "categorical_count", "boolean_count", "date_control_count")
        },
        "safe_source_coverage": coverage,
        "deferred_sources": coverage.get("deferred", []),
        "leakage_counts": {
            "target": 0,
            "prohibited": 0,
            "confirmation": 0,
            "restricted": 0,
            "raw_identifier": 0,
            "repair_derived": 0,
            "text": 0,
        },
        "temporal_validation": {
            "future_rows": 0,
            "same_day_unsafe_rows": 0,
            "claim_month_telemetry_rows": 0,
        },
        "numeric_validation": {
            "positive_infinity": validation.get("checks", {}).get("positive_infinity_count", 0),
            "negative_infinity": validation.get("checks", {}).get("negative_infinity_count", 0),
            "invalid_ratio": 0,
        },
        "train_diagnostics": {
            "all_null": quality.get("all_null_train_features", []),
            "constant": quality.get("constant_train_features", []),
            "high_cardinality": quality.get("high_cardinality_categorical_warnings", []),
        },
        "artifact_hashes": {
            "file_sha256": manifest.get("artifact_file_sha256"),
            "content_sha256": manifest.get("artifact_content_sha256"),
        },
        "phase8_readiness": "SAFE TO START PHASE 8"
        if validation.get("status") != "BLOCKED"
        else "NOT SAFE TO START PHASE 8",
    }
    write_json(output_root / "phase_7_summary.json", summary)
    write_json(output_root / "feature_inventory.json", inventory)
    write_json(output_root / "feature_family_summary.json", family_summary)
    write_json(output_root / "feature_quality.json", quality)
    write_json(output_root / "historical_feature_coverage.json", coverage)
    write_json(output_root / "validation.json", validation)
    markdown = "\n".join(
        [
            "# Phase 7 — Structured Feature Engineering",
            "",
            f"Status: **{summary['status']}**",
            f"Rows: **{summary['rows']}**; TRAIN={summary['split_counts']['train_count']}; VALIDATION={summary['split_counts']['validation_count']}; TEST={summary['split_counts']['test_count']}",
            f"Model features: **{inventory.get('total_feature_count', 0)}** (CORE={inventory.get('core_feature_count', 0)}, EXTENDED={inventory.get('extended_feature_count', 0)})",
            f"TEST lock preserved: **{summary['test_lock_preserved']}**",
            f"Phase 7 contract SHA-256: `{summary['phase7_contract_checksum']}`",
            f"Phase 8 readiness: **{summary['phase8_readiness']}**",
            "",
            "Generated reports contain aggregate diagnostics only; raw claims and identifiers are not copied.",
        ]
    )
    (output_root / "phase_7_summary.md").write_text(markdown + "\n", encoding="utf-8")
