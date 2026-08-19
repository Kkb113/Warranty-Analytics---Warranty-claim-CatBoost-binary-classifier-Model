"""Final POC status logic; TEST cannot select a model or threshold."""

from __future__ import annotations

from typing import Any


def final_model_status(
    signal: dict[str, Any],
    comparison: dict[str, Any],
    *,
    provenance_valid: bool,
    leakage_valid: bool,
    scoring_valid: bool,
    test_use_valid: bool,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    warnings = list(warnings or [])
    if (
        signal.get("status") == "SIGNAL_COLLAPSE"
        or comparison.get("generalization_status") == "SEVERE_DEGRADATION"
        or not all((provenance_valid, leakage_valid, scoring_valid, test_use_valid))
    ):
        status = "FAILED_GENERALIZATION"
    elif comparison.get("generalization_status") == "MODERATE_DEGRADATION" or warnings:
        status = "ACCEPTED_WITH_LIMITATIONS"
    else:
        status = "ACCEPTED_FOR_POC"
    return {
        "phase": 15,
        "final_model_status": status,
        "signal_status": signal.get("status"),
        "generalization_status": comparison.get("generalization_status"),
        "provenance_valid": bool(provenance_valid),
        "leakage_valid": bool(leakage_valid),
        "scoring_valid": bool(scoring_valid),
        "test_use_valid": bool(test_use_valid),
        "warnings": sorted(set(warnings)),
        "safe_to_start_phase16": status in {"ACCEPTED_FOR_POC", "ACCEPTED_WITH_LIMITATIONS"},
    }


__all__ = ["final_model_status"]
