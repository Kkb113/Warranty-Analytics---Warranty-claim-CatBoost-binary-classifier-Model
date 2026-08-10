"""Aggregate-only Phase 6 reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..feature_mart.manifest import write_json
from .assignments import split_date_ranges
from .models import SplitError


def build_split_distribution(
    assignments: pd.DataFrame,
    snapshot: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate target diagnostics only after split labels exist."""

    target_column = "target__high_cost_claim_flag"
    if target_column not in snapshot:
        raise SplitError("Phase 5 snapshot target is required for post-assignment diagnostics.")
    target = snapshot[["warranty_claim_key", target_column]].copy()
    target[target_column] = pd.to_numeric(target[target_column], errors="coerce")
    joined = assignments.merge(target, on="warranty_claim_key", how="left", validate="one_to_one")
    total = int(len(joined))
    total_positive = int(joined[target_column].eq(1).sum())
    total_negative = int(joined[target_column].eq(0).sum())
    result: dict[str, Any] = {
        "total_claims": total,
        "overall_positive_count": total_positive,
        "overall_negative_count": total_negative,
        "overall_positive_percentage": _percentage(total_positive, total),
        "by_split": {},
    }
    ranges = split_date_ranges(assignments)
    for split in ("TRAIN", "VALIDATION", "TEST"):
        subset = joined.loc[joined["split"] == split]
        positives = int(subset[target_column].eq(1).sum())
        negatives = int(subset[target_column].eq(0).sum())
        result["by_split"][split] = {
            "row_count": int(len(subset)),
            "row_percentage": _percentage(len(subset), total),
            "positive_count": positives,
            "negative_count": negatives,
            "positive_percentage": _percentage(positives, len(subset)),
            "earliest_claim_date": ranges[split]["earliest_claim_date"],
            "latest_claim_date": ranges[split]["latest_claim_date"],
        }
    return result


def _percentage(numerator: int, denominator: int) -> float:
    return round((numerator / denominator) * 100.0, 6) if denominator else 0.0


def build_phase6_summary(
    *,
    manifest: dict[str, Any],
    validation: dict[str, Any],
    split_distribution: dict[str, Any],
    group_overlap: dict[str, Any],
    cohort_summary: dict[str, Any],
    fingerprint_overlap: dict[str, Any],
    phase5_validation: dict[str, Any],
    test_lock_valid: bool,
) -> dict[str, Any]:
    """Create the aggregate summary used by JSON and Markdown reports."""

    checks = validation.get("checks", {})
    safe_for_phase7 = not validation.get("errors") and test_lock_valid
    return {
        "phase6_status": manifest.get("validation_status"),
        "phase5_mart_run": manifest.get("input_mart_run"),
        "phase5_validation_status": phase5_validation.get("status"),
        "total_claims": split_distribution.get("total_claims", 0),
        "split_distribution": split_distribution.get("by_split", {}),
        "overall_positive_prevalence": split_distribution.get("overall_positive_percentage", 0.0),
        "train_end_date": manifest.get("train_end_date"),
        "validation_end_date": manifest.get("validation_end_date"),
        "target_values_used_to_choose_boundaries": False,
        "dates_split_between_partitions": not bool(checks.get("same_date_integrity_valid", False)),
        "duplicate_claim_keys_across_partitions": not bool(
            checks.get("claim_coverage_valid", False)
        ),
        "all_claims_assigned": bool(checks.get("claim_coverage_valid", False)),
        "fingerprint_overlap": fingerprint_overlap,
        "evaluation_cohorts": cohort_summary,
        "group_overlap": group_overlap,
        "historical_supplier_exposure": group_overlap.get("group_types", {}).get(
            "historical_supplier", {}
        ),
        "component_lot_exposure": group_overlap.get("group_types", {}).get(
            "historical_component_lot", {}
        ),
        "component_batch_exposure": group_overlap.get("group_types", {}).get(
            "historical_component_batch", {}
        ),
        "test_lock_valid": test_lock_valid,
        "safe_for_phase7": safe_for_phase7,
        "validation": validation,
        "warnings": manifest.get("warnings", []),
    }


def _markdown_summary(summary: dict[str, Any]) -> str:
    distribution = summary.get("split_distribution", {})
    lines = [
        "# Phase 6 — Train / Validation / Test Split Design",
        "",
        f"- Phase 5 mart used: `{summary.get('phase5_mart_run', '-')}`",
        f"- Phase 5 validation: **{summary.get('phase5_validation_status', 'UNKNOWN')}**",
        f"- Phase 6 status: **{summary.get('phase6_status', 'UNKNOWN')}**",
        f"- Total eligible claims: **{summary.get('total_claims', 0)}**",
        "",
        "## Chronological partition",
        "",
        "Boundaries were selected from claim dates, date-level row counts, and configured fractions only. Target values were not used to choose boundaries.",
        "",
        "| Split | Rows | Actual % | Positive | Negative | Positive % | Date range |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for split in ("TRAIN", "VALIDATION", "TEST"):
        item = distribution.get(split, {})
        lines.append(
            f"| {split} | {item.get('row_count', 0)} | {item.get('row_percentage', 0.0):.6f}% | "
            f"{item.get('positive_count', 0)} | {item.get('negative_count', 0)} | "
            f"{item.get('positive_percentage', 0.0):.6f}% | "
            f"{item.get('earliest_claim_date', '')} to {item.get('latest_claim_date', '')} |"
        )
    lines.extend(
        [
            "",
            f"- Train boundary date: `{summary.get('train_end_date', '-')}`",
            f"- Validation boundary date: `{summary.get('validation_end_date', '-')}`",
            f"- Overall positive prevalence: **{summary.get('overall_positive_prevalence', 0.0):.6f}%**",
            f"- Same-date claims split across partitions: **{summary.get('dates_split_between_partitions')}**",
            f"- Duplicate claim keys across partitions: **{summary.get('duplicate_claim_keys_across_partitions')}**",
            f"- All eligible claims assigned exactly once: **{summary.get('all_claims_assigned')}**",
            "",
            "## Exposure and evaluation cohorts",
            "",
            f"- Fingerprint overlap is reported as a **{summary.get('fingerprint_overlap', {}).get('overlap_severity', 'WARNING')}**, and overlapping claims remain in the primary chronological partitions.",
            f"- Fingerprint-clean validation claims: **{summary.get('fingerprint_overlap', {}).get('validation_fingerprint_clean_claims', 0)}**",
            f"- Fingerprint-clean test claims: **{summary.get('fingerprint_overlap', {}).get('test_fingerprint_clean_claims', 0)}**",
            f"- Unseen validation trucks: **{summary.get('evaluation_cohorts', {}).get('by_split', {}).get('VALIDATION', {}).get('unseen_truck_claims', 0)} claims**",
            f"- Unseen test trucks: **{summary.get('evaluation_cohorts', {}).get('by_split', {}).get('TEST', {}).get('unseen_truck_claims', 0)} claims**",
            f"- Unseen validation production batches: **{summary.get('evaluation_cohorts', {}).get('by_split', {}).get('VALIDATION', {}).get('unseen_production_batch_claims', 0)} claims**",
            f"- Unseen test production batches: **{summary.get('evaluation_cohorts', {}).get('by_split', {}).get('TEST', {}).get('unseen_production_batch_claims', 0)} claims**",
            f"- Unseen validation service centers: **{summary.get('evaluation_cohorts', {}).get('by_split', {}).get('VALIDATION', {}).get('unseen_service_center_claims', 0)} claims**",
            f"- Unseen test service centers: **{summary.get('evaluation_cohorts', {}).get('by_split', {}).get('TEST', {}).get('unseen_service_center_claims', 0)} claims**",
            "",
            "Historical supplier and component lot/batch exposure is metadata-only and is not converted into target-rate features.",
            "",
            "## Test lock and Phase 7 readiness",
            "",
            f"- Test lock valid: **{summary.get('test_lock_valid')}**",
            "- Test target-based evaluation is reserved for Phase 15; Phases 9–14 must not use test performance for development decisions.",
            f"- Safe to proceed to Phase 7 target-independent feature construction: **{summary.get('safe_for_phase7')}**",
            "",
            "No features were engineered, no labels were transformed, no resampling was performed, no model was trained, and no model-performance metric was calculated in Phase 6.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase6_reports(
    *,
    output_root: Path,
    summary: dict[str, Any],
    split_distribution: dict[str, Any],
    group_overlap: dict[str, Any],
    cohort_summary: dict[str, Any],
    validation: dict[str, Any],
) -> list[Path]:
    """Write aggregate-only Phase 6 reports."""

    output_root.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, Any] = {
        "phase_6_summary.json": summary,
        "split_distribution.json": split_distribution,
        "group_overlap.json": group_overlap,
        "evaluation_cohorts.json": cohort_summary,
        "split_validation.json": validation,
    }
    paths: list[Path] = []
    for filename, payload in payloads.items():
        path = output_root / filename
        write_json(path, payload)
        paths.append(path)
    markdown = output_root / "phase_6_summary.md"
    markdown.write_text(_markdown_summary(summary), encoding="utf-8")
    paths.append(markdown)
    return paths
