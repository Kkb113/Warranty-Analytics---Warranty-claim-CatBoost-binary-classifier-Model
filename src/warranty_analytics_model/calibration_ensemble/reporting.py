"""Small, claim-free Phase 13 report bundle writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..feature_mart.manifest import write_json


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_phase13_reports(report_root: Path, run_id: str, payload: dict[str, Any]) -> Path:
    directory = report_root / str(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    validation = _json_safe(payload.get("validation", {}))
    validation_metrics = _json_safe(payload.get("validation_metrics", {}))
    calibration = _json_safe(payload.get("calibration_summary", []))
    ensemble = _json_safe(payload.get("ensemble_selection", {}))
    parent = _json_safe(payload.get("parent_resolution", {}))
    threshold = _json_safe(payload.get("threshold_policy", {}))
    write_json(directory / "phase_13_summary.json", _json_safe(payload))
    write_json(directory / "calibration_comparison.json", {"rows": calibration})
    write_json(directory / "reliability_analysis.json", {"source": "reliability_bins.parquet"})
    write_json(directory / "ensemble_comparison.json", ensemble)
    write_json(directory / "parent_comparison.json", parent)
    write_json(directory / "threshold_comparison.json", threshold)
    write_json(directory / "validation_metrics.json", validation_metrics)
    write_json(directory / "validation.json", validation)
    markdown = [
        "# Phase 13 Probability Calibration & Controlled Ensembling",
        "",
        f"Run: `{run_id}`",
        "",
        f"Validation status: **{validation.get('hardening_status', validation.get('status', 'UNKNOWN'))}**",
        "",
        f"Development champion: `{validation_metrics.get('phase13_development_champion', 'unknown')}`",
        "",
        "This report intentionally contains aggregate metrics and provenance only; claim-level data remains in the artifact bundle.",
    ]
    (directory / "phase_13_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return directory


__all__ = ["write_phase13_reports"]
