"""Claim-free Phase 14 aggregate report writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..feature_mart.manifest import write_json


def write_phase14_reports(report_root: Path, run_id: str, payload: dict[str, Any]) -> Path:
    directory = report_root / str(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    for name in (
        "phase_14_summary.json",
        "robustness_summary.json",
        "temporal_summary.json",
        "slice_summary.json",
        "drift_summary.json",
        "error_analysis_summary.json",
        "phase15_readiness.json",
        "validation.json",
    ):
        write_json(directory / name, payload.get(name.removesuffix(".json"), payload))
    (directory / "phase_14_summary.md").write_text(
        "# Phase 14 Robustness, Stability & Error Analysis\n\n"
        f"Run: `{run_id}`\n\n"
        "Aggregate evidence only; claim-level identifiers remain local.\n",
        encoding="utf-8",
    )
    return directory


__all__ = ["write_phase14_reports"]
