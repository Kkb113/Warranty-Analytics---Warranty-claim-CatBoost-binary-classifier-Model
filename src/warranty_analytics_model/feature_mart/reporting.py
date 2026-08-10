"""Aggregate-only Phase 5 reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .manifest import write_json


def _summary_payload(manifest: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    snapshot = validation.get("snapshot", {})
    leakage = validation.get("leakage", {})
    temporal = validation.get("temporal", {})
    bridges = validation.get("bridge_row_counts", {})
    coverage = validation.get("history_coverage", {})
    direct_count = int(manifest.get("direct_feature_count", 0))
    historical_count = int(manifest.get("historical_source_field_count", 0))
    return {
        "phase": "Phase 5 — Claim-Level Feature Mart Construction",
        "status": validation.get("status", "INCOMPLETE"),
        "source_claims": int(manifest.get("total_source_claims", 0)),
        "source_drift": manifest.get("source_drift", {}),
        "eligible_claims": int(manifest.get("eligible_claims", 0)),
        "claim_snapshot_rows": int(snapshot.get("rows", manifest.get("snapshot_rows", 0))),
        "warranty_claim_key_unique": bool(
            snapshot.get("unique_claims", 0) == snapshot.get("rows", 0)
        ),
        "positive_claims": int(snapshot.get("positive_claims", manifest.get("positive_claims", 0))),
        "negative_claims": int(snapshot.get("negative_claims", manifest.get("negative_claims", 0))),
        "direct_tier_a_fields": {
            "expected": direct_count,
            "materialized": direct_count
            - len(
                [
                    item
                    for item in manifest.get("deferred_fields", [])
                    if item.get("tier") == "direct"
                ]
            ),
            "deferred": len(
                [
                    item
                    for item in manifest.get("deferred_fields", [])
                    if item.get("tier") == "direct"
                ]
            ),
        },
        "historical_tier_a_fields": {
            "expected": historical_count,
            "mapped": historical_count
            - len(
                [
                    item
                    for item in manifest.get("deferred_fields", [])
                    if item.get("tier") == "historical"
                ]
            ),
            "deferred": len(
                [
                    item
                    for item in manifest.get("deferred_fields", [])
                    if item.get("tier") == "historical"
                ]
            ),
        },
        "direct_join_multiplication_count": sum(
            int(item.get("multiplication_count", 0))
            for item in manifest.get("direct_join_validation", {}).values()
        ),
        "bridge_row_counts": bridges,
        "history_coverage": coverage,
        "leakage": {
            "prohibited_model_fields": leakage.get("prohibited_model_fields", []),
            "confirmation_model_fields": leakage.get("confirmation_model_fields", []),
            "restricted_tier_a_fields": leakage.get("restricted_tier_a_fields", []),
            "identifier_model_fields": leakage.get("identifier_model_fields", []),
            "target_feature_leakage": leakage.get("target_feature_leakage", []),
            "wildcard_leakage_violations": leakage.get("wildcard_leakage_violations", []),
        },
        "temporal": temporal,
        "safe_to_start_phase6": validation.get("status") in {"PASS", "PASS WITH WARNINGS"}
        and not validation.get("errors"),
        "warnings": validation.get("warnings", []),
        "errors": validation.get("errors", []),
    }


def _summary_markdown(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    snapshot = summary["claim_snapshot_rows"]
    safe = (
        "SAFE TO START PHASE 6" if summary["safe_to_start_phase6"] else "NOT SAFE TO START PHASE 6"
    )
    lines = [
        "# Phase 5 — Claim-Level Feature Mart Construction",
        "",
        f"Status: **{summary['status']}**",
        "",
        "The feature mart is a local Parquet bundle. Train/validation/test assignments "
        "and predictive-model training are deferred to later phases.",
        "",
        "## Aggregate result",
        "",
        f"- Source claims: {summary['source_claims']}",
        f"- Eligible claims: {summary['eligible_claims']}",
        f"- Claim snapshot rows: {snapshot}",
        f"- Unique claims: {summary['warranty_claim_key_unique']}",
        f"- Positive/negative target counts: {summary['positive_claims']} / {summary['negative_claims']}",
        f"- Direct Tier A fields: {summary['direct_tier_a_fields']['materialized']} / {summary['direct_tier_a_fields']['expected']}",
        f"- Historical Tier A fields: {summary['historical_tier_a_fields']['mapped']} / {summary['historical_tier_a_fields']['expected']}",
        f"- Direct join multiplication count: {summary['direct_join_multiplication_count']}",
        f"- Phase 6 recommendation: **{safe}**",
        "",
        "## Historical bridges",
        "",
    ]
    for name, count in sorted(summary["bridge_row_counts"].items()):
        lines.append(f"- {name}: {count} rows")
    lines.extend(
        [
            "",
            "## Leakage and temporal gates",
            "",
            f"- Prohibited model fields: {len(summary['leakage']['prohibited_model_fields'])}",
            f"- Confirmation model fields: {len(summary['leakage']['confirmation_model_fields'])}",
            f"- Restricted Tier A fields: {len(summary['leakage']['restricted_tier_a_fields'])}",
            f"- Identifier model fields: {len(summary['leakage']['identifier_model_fields'])}",
            f"- Wildcard leakage violations: {len(summary['leakage']['wildcard_leakage_violations'])}",
            f"- Same-day/future violations: {summary['temporal'].get('same_day_violations', 0)} / {summary['temporal'].get('future_history_violations', 0)}",
            f"- Claim-month telemetry violations: {summary['temporal'].get('claim_month_telemetry_violations', 0)}",
            f"- Current service-event violations: {summary['temporal'].get('current_service_event_violations', 0)}",
            f"- Current repair-line violations: {summary['temporal'].get('current_repair_line_violations', 0)}",
            "",
            "## Policy lineage",
            "",
            f"- Mart contract checksum: `{manifest.get('mart_contract_checksum', '-')}`",
            f"- Schema contract checksum: `{manifest.get('schema_contract_checksum', '-')}`",
            f"- Target policy checksum: `{manifest.get('target_contract_checksum', '-')}`",
            f"- Feature policy checksum: `{manifest.get('feature_policy_checksum', '-')}`",
            f"- Leakage policy checksum: `{manifest.get('leakage_policy_checksum', '-')}`",
            "",
        ]
    )
    for warning in summary.get("warnings", []):
        lines.append(f"> Warning: {warning}")
    return "\n".join(lines) + "\n"


def write_phase5_reports(
    manifest: dict[str, Any],
    validation: dict[str, Any],
    output_root: Path,
    run_id: str,
) -> list[Path]:
    """Write the required aggregate-only JSON and Markdown reports."""

    report_dir = output_root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary_payload(manifest, validation)
    direct = {
        "expected_fields": int(manifest.get("direct_feature_count", 0)),
        "materialized_fields": int(manifest.get("direct_feature_count", 0)),
        "deferred_fields": [
            item for item in manifest.get("deferred_fields", []) if item.get("tier") == "direct"
        ],
        "coverage_percentage": 100.0 if manifest.get("direct_feature_count", 0) else 0.0,
    }
    historical = {
        "expected_fields": int(manifest.get("historical_source_field_count", 0)),
        "mapped_fields": int(manifest.get("historical_source_field_count", 0)),
        "deferred_fields": [
            item for item in manifest.get("deferred_fields", []) if item.get("tier") == "historical"
        ],
        "coverage_percentage": 100.0 if manifest.get("historical_source_field_count", 0) else 0.0,
    }
    payloads = {
        "phase_5_summary.json": summary,
        "mart_validation.json": validation,
        "history_coverage.json": validation.get("history_coverage", {}),
        "direct_feature_coverage.json": direct,
        "historical_field_coverage.json": historical,
    }
    paths: list[Path] = []
    for name, payload in payloads.items():
        path = report_dir / name
        write_json(path, payload)
        paths.append(path)
    markdown = report_dir / "phase_5_summary.md"
    markdown.write_text(_summary_markdown(summary, manifest), encoding="utf-8")
    paths.append(markdown)
    return paths
