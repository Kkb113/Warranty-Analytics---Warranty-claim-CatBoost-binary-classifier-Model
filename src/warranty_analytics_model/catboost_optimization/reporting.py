"""Aggregate Phase 10 reports with no claim-level identifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _safe_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_payload(item) for item in value]
    return value


def write_phase10_reports(report_directory: Path, summary: dict[str, Any]) -> None:
    """Write JSON and Markdown aggregate summaries only."""

    report_directory.mkdir(parents=True, exist_ok=True)
    payload = _safe_payload(summary)
    (report_directory / "phase_10_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    comparison = payload.get("optimization_comparison", {})
    lines = [
        "# Phase 10 — CatBoost Optimization",
        "",
        f"Status: **{payload.get('status', 'BLOCKED')}**",
        f"Development champion: `{payload.get('phase10_development_champion', '-')}`",
        "",
        "Phase 9 remains immutable. Hyperparameter studies used TRAIN-only chronological inner folds;",
        "outer VALIDATION was loaded only after the study freeze and TEST remained sealed until Phase 15.",
        "",
        "## Track comparison",
        "",
    ]
    if isinstance(comparison, dict):
        for track, item in sorted(comparison.items()):
            if isinstance(item, dict):
                lines.append(
                    f"- {track}: AP delta `{item.get('average_precision_delta', '-')}`, "
                    f"fallback `{item.get('fallback_to_baseline', '-')}`"
                )
    warnings = payload.get("warnings", [])
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")
    (report_directory / "phase_10_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, content in {
        "optimization_comparison.json": payload.get("optimization_comparison", {}),
        "inner_cv_summary.json": payload.get("inner_cv_summary", {}),
        "best_parameters.json": payload.get("best_parameters", {}),
        "validation_metrics.json": payload.get("validation_metrics", {}),
        "validation.json": payload.get("validation", {}),
    }.items():
        (report_directory / name).write_text(
            json.dumps(_safe_payload(content), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
