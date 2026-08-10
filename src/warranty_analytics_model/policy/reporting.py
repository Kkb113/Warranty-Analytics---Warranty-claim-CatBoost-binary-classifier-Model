"""Secret-safe Phase 4 validation report generation."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from .models import FeaturePolicyContract, LeakagePolicyContract, Phase4ValidationResult


def _json_payload(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n"


def _summary_markdown(
    result: Phase4ValidationResult,
    feature_policy: FeaturePolicyContract,
) -> str:
    target = result.target_validation
    coverage = result.contract_validation
    checksums = result.checksums
    source = result.source_policy_validation
    lines = [
        "# Phase 4 validation summary",
        "",
        f"Status: {result.status}",
        "",
        "## Target and eligibility",
        "",
        f"- Target: {target.get('target_name', 'dbo.fact_warranty_claim.high_cost_claim_flag')}",
        f"- Stored target valid: {target.get('target_valid', False)}",
        f"- Total claims: {target.get('total_claims', 0)}",
        f"- Eligible claims: {target.get('eligible_claims', 0)}",
        f"- Excluded claims: {target.get('excluded_claims', 0)}",
        f"- Positive claims: {target.get('positive_claims', 0)}",
        f"- Negative claims: {target.get('negative_claims', 0)}",
        f"- Positive prevalence among eligible claims: {target.get('positive_percentage', 0.0)}%",
        "- Business definition confirmed: false",
        "- Synthetic-development-only: true",
        "- Prediction reference: claim_date (provisional date-level reference)",
        "",
        "### Eligibility categories",
        "",
        "| Category | Count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {category} | {count} |"
        for category, count in sorted(target.get("category_counts", {}).items())
    )
    lines.extend(
        [
            "",
            "## Field-policy coverage",
            "",
            f"- Schema columns: {coverage.schema_columns}",
            f"- Classified columns: {coverage.classified_columns}",
            f"- Unclassified columns: {coverage.unclassified_columns}",
            "",
            "| Policy | Count |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        f"| {policy} | {count} |" for policy, count in sorted(coverage.policy_counts.items())
    )
    lines.extend(
        [
            "",
            "## Historical timing rules",
            "",
            "- Event histories use event_date < claim_date; same-day records are excluded.",
            "- Monthly telemetry uses end_of_month(month_start_date) < claim_date.",
            "- Current-claim service events and repair lines are excluded from history.",
            "- Historical repairs require a prior claim completion date strictly before the current claim.",
            "- Current causal-component and failure-code fields remain unresolved and are not Tier A inputs.",
            "",
            "## Enforcement",
            "",
            f"- Known leakage fields prohibited: {result.leakage_policy_validation.get('valid', False)}",
            f"- Identifier overlap with Tier A: {result.leakage_policy_validation.get('identifier_safe_baseline_overlap', [])}",
            f"- Restricted fields isolated from Tier A: {not set(coverage.restricted_experimental_list) & set(coverage.safe_baseline_allowlist)}",
            f"- Historical source rules valid: {source.get('valid', False)}",
            "- No feature mart, split, or model is created by Phase 4.",
            "",
            "## Contract checksums",
            "",
        ]
    )
    lines.extend(f"- {name}: {checksum}" for name, checksum in sorted(checksums.items()))
    if result.errors:
        lines.extend(["", "## Blocking errors", "", *[f"- {error}" for error in result.errors]])
    if result.warnings:
        lines.extend(["", "## Warnings", "", *[f"- {warning}" for warning in result.warnings]])
    lines.extend(
        [
            "",
            "## Feature tiers",
            "",
            f"- Tier A safe baseline entries: {len(coverage.safe_baseline_allowlist)}",
            f"- Tier A historical entries: {len(coverage.historical_allowlist)}",
            f"- Tier B restricted experimental entries: {len(coverage.restricted_experimental_list)}",
            f"- Requires-confirmation entries: {len(coverage.requires_confirmation_list)}",
            f"- Lineage/split-control entries: {len(coverage.lineage_fields)} including explicit control identifiers and derived audit metadata.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase4_reports(
    result: Phase4ValidationResult,
    feature_policy: FeaturePolicyContract,
    leakage_policy: LeakagePolicyContract,
    output_root: Path,
    formats: tuple[str, ...] = ("json", "markdown"),
) -> list[Path]:
    """Write the required Phase 4 reports beneath a timestamped directory."""

    del leakage_policy
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_dir = output_root / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)
    result.report_directory = str(report_dir)
    payload = result.model_dump(mode="json")
    payload["report_directory"] = str(report_dir)
    written: list[Path] = []
    if "json" in formats:
        json_reports = {
            "phase_4_summary.json": payload,
            "target_validation.json": result.target_validation,
            "feature_policy_validation.json": result.contract_validation.model_dump(mode="json"),
            "leakage_policy_validation.json": result.leakage_policy_validation,
            "field_policy_coverage.json": {
                "schema_columns": result.contract_validation.schema_columns,
                "classified_columns": result.contract_validation.classified_columns,
                "unclassified_columns": result.contract_validation.unclassified_columns,
                "policy_counts": result.contract_validation.policy_counts,
                "safe_baseline_allowlist": result.contract_validation.safe_baseline_allowlist,
                "historical_allowlist": result.contract_validation.historical_allowlist,
                "restricted_experimental_list": result.contract_validation.restricted_experimental_list,
                "requires_confirmation_list": result.contract_validation.requires_confirmation_list,
                "lineage_fields": result.contract_validation.lineage_fields,
            },
        }
        for name, item in json_reports.items():
            path = report_dir / name
            path.write_text(_json_payload(item), encoding="utf-8")
            written.append(path)
        inventory_path = report_dir / "field_policy_inventory.csv"
        with inventory_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "table",
                    "column",
                    "policy",
                    "role",
                    "as_of_rule",
                    "synthetic_poc_allowed",
                    "production_approved",
                    "is_model_feature",
                    "is_lineage",
                ],
            )
            writer.writeheader()
            for entry in feature_policy.field_policies:
                writer.writerow(
                    {
                        "table": entry.table,
                        "column": entry.column,
                        "policy": entry.policy,
                        "role": entry.role,
                        "as_of_rule": entry.as_of_rule,
                        "synthetic_poc_allowed": entry.synthetic_poc_allowed,
                        "production_approved": entry.production_approved,
                        "is_model_feature": entry.is_model_feature,
                        "is_lineage": entry.is_lineage,
                    }
                )
        written.append(inventory_path)
    if "markdown" in formats:
        markdown_path = report_dir / "phase_4_summary.md"
        markdown_path.write_text(_summary_markdown(result, feature_policy), encoding="utf-8")
        written.append(markdown_path)
    return written
