"""Claim-free Phase 15 report writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..feature_mart.manifest import write_json


def write_phase15_reports(report_root: Path, run_id: str, payload: dict[str, Any]) -> Path:
    directory = report_root / str(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    mappings = {
        "phase15_summary.json": payload,
        "phase15_summary.md": None,
        "final_test_metrics.json": payload.get("test_metrics", {}),
        "validation_test_comparison.json": payload.get("validation_test_comparison", {}),
        "topk_summary.json": payload.get("test_topk_lift", {}),
        "risk_decile_summary.json": payload.get("ranking_concentration_summary", {}),
        "threshold_summary.json": payload.get("test_threshold_metrics", {}),
        "uncertainty_summary.json": payload.get("test_bootstrap_summary", {}),
        "temporal_summary.json": payload.get("temporal_summary", {}),
        "slice_summary.json": payload.get("slice_summary", {}),
        "error_summary.json": payload.get("test_error_summary", {}),
        "model_status.json": payload.get("phase15_final_model_status", {}),
        "validation.json": payload.get("validation", {}),
    }
    for name, value in mappings.items():
        if name.endswith(".md"):
            continue
        write_json(directory / name, value)
    status = payload.get("phase15_final_model_status", {}).get("final_model_status", "UNKNOWN")
    metrics = payload.get("test_metrics", {})
    (directory / "phase15_summary.md").write_text(
        "# Phase 15 Final Untouched TEST Evaluation\n\n"
        f"Run: `{run_id}`\n\n"
        "The frozen development champion was evaluated on untouched TEST data "
        "without TEST-driven model changes.\n\n"
        f"Average Precision: `{metrics.get('average_precision', 'n/a')}`; "
        f"ROC-AUC: `{metrics.get('roc_auc', 'n/a')}`; "
        f"Final model status: `{status}`.\n\n"
        "Claim-level identifiers are intentionally excluded from this report.\n",
        encoding="utf-8",
    )
    return directory


__all__ = ["write_phase15_reports"]
