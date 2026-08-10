"""Aggregate-only Phase 8 reports with no claim identifiers or raw text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..feature_mart.manifest import write_json


def _summary_payload(
    manifest: dict[str, Any],
    quality: dict[str, Any],
    source_coverage: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": "Phase 8 — Text Feature Development",
        "status": validation.get("status", manifest.get("validation_status", "INCOMPLETE")),
        "input_phase5_run": manifest.get("input_phase5_run"),
        "input_phase6_run": manifest.get("input_phase6_run"),
        "input_phase7_run": manifest.get("input_phase7_run"),
        "rows": manifest.get("row_count", 0),
        "split_counts": {
            "TRAIN": manifest.get("train_count", 0),
            "VALIDATION": manifest.get("validation_count", 0),
            "TEST": manifest.get("test_count", 0),
        },
        "text_document_feature_count": manifest.get("text_document_feature_count", 0),
        "lexical_feature_count": manifest.get("lexical_feature_count", 0),
        "quality_by_window": {
            window: {key: value for key, value in item.items() if key != "warnings"}
            for window, item in quality.items()
            if isinstance(item, dict) and window in {"6m", "12m", "24m", "all"}
        },
        "source_coverage": source_coverage,
        "temporal_audit": validation.get("temporal_audit", {}),
        "leakage_audit": validation.get("leakage_audit", {}),
        "errors": validation.get("errors", []),
        "warnings": validation.get("warnings", []),
        "raw_text_report_exposure": False,
    }


def _summary_markdown(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    safe = "SAFE TO START PHASE 9" if not summary.get("errors") else "NOT SAFE TO START PHASE 9"
    lines = [
        "# Phase 8 — Text Feature Development",
        "",
        f"Status: **{summary['status']}**",
        "",
        "Phase 8 creates deterministic historical text candidates as a companion "
        "to the locked Phase 7 structured artifact. It does not fit a vectorizer, "
        "create embeddings, train a model, or calculate target-based metrics.",
        "",
        "## Inputs and population",
        "",
        f"- Phase 5 run: `{summary['input_phase5_run']}`",
        f"- Phase 6 run: `{summary['input_phase6_run']}`",
        f"- Phase 7 run: `{summary['input_phase7_run']}`",
        f"- Rows: {summary['rows']}",
        f"- TRAIN / VALIDATION / TEST: {summary['split_counts']['TRAIN']} / {summary['split_counts']['VALIDATION']} / {summary['split_counts']['TEST']}",
        "",
        "## Candidate inventory",
        "",
        f"- Raw text documents: {summary['text_document_feature_count']}",
        f"- Lexical/boolean candidates: {summary['lexical_feature_count']}",
        "- Approved text value source: `prior_failure__failure_description` only",
        "",
        "## Quality and safety",
        "",
        f"- Same-day/future text records: {summary['temporal_audit'].get('same_day_text_records', 0)} / {summary['temporal_audit'].get('future_text_records', 0)}",
        f"- Target/current/prohibited/raw-ID text sources: {sum(summary['leakage_audit'].values()) if summary['leakage_audit'] else 0}",
        f"- Raw text emitted to reports: {summary['raw_text_report_exposure']}",
        "",
        f"Final recommendation: **{safe}**",
        "",
        "Warnings remain visible for synthetic-only development and the "
        "UNVERSIONED_FAILURE_DESCRIPTION_DIMENSION production reapproval requirement.",
        "",
    ]
    return "\n".join(lines)


def write_phase8_reports(
    *,
    output_root: Path,
    run_id: str,
    manifest: dict[str, Any],
    quality: dict[str, Any],
    source_coverage: dict[str, Any],
    validation: dict[str, Any],
) -> list[Path]:
    """Write required aggregate-only JSON and Markdown reports."""

    report_dir = output_root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary_payload(manifest, quality, source_coverage, validation)
    payloads = {
        "phase_8_summary.json": summary,
        "text_feature_inventory.json": {
            "text_document_feature_count": manifest.get("text_document_feature_count", 0),
            "lexical_feature_count": manifest.get("lexical_feature_count", 0),
            "text_feature_names": manifest.get("text_feature_names", []),
        },
        "text_quality.json": quality,
        "source_coverage.json": source_coverage,
        "validation.json": validation,
    }
    paths: list[Path] = []
    for name, payload in payloads.items():
        path = report_dir / name
        write_json(path, payload)
        paths.append(path)
    markdown = report_dir / "phase_8_summary.md"
    markdown.write_text(_summary_markdown(summary, manifest), encoding="utf-8")
    paths.append(markdown)
    return paths
