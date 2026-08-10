"""Aggregate-only Phase 9 reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..feature_mart.manifest import write_json


def write_phase9_reports(report_dir: Path, summary: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "baseline_model_report.json", summary)
    lines = [
        "# Phase 9 — Baseline Model Training",
        "",
        f"Status: **{summary['status']}**",
        f"Development champion: **{summary['champion_experiment_id']}**",
        "",
        "## Validation metrics",
        "",
        "| Experiment | Status | Features | Average precision | ROC AUC | Log loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for experiment_id, item in summary["experiments"].items():
        metrics = item.get("metrics", {})
        lines.append(
            f"| {experiment_id} | {item['status']} | {item.get('feature_count', 0)} | "
            f"{metrics.get('average_precision', '-')} | {metrics.get('roc_auc', '-')} | "
            f"{metrics.get('log_loss', '-')} |"
        )
    lines.extend(
        [
            "",
            "## Safety seal",
            "",
            "Only TRAIN labels were used for fitting. Metrics use VALIDATION labels only. "
            "No TEST labels, predictions, or metrics were accessed or produced.",
            "",
            "This is a synthetic proof-of-concept baseline and is not approved for production.",
        ]
    )
    (report_dir / "baseline_model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
