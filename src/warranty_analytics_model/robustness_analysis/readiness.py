"""Independent Phase 15 readiness decision rules."""

from __future__ import annotations

from typing import Any


def readiness_gate(
    overall: dict[str, Any],
    warnings: list[str],
    *,
    hard_blockers: list[str] | None = None,
    test_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blockers = list(hard_blockers or [])
    prevalence = float(overall.get("prevalence", overall.get("observed_prevalence", 0.0)))
    ap = overall.get("average_precision")
    roc = overall.get("roc_auc")
    if ap is None or float(ap) <= prevalence + 1.0e-6:
        blockers.append("ROBUSTNESS_SIGNAL_COLLAPSE: AP is not above prevalence.")
    if roc is None or float(roc) <= 0.50:
        blockers.append("ROBUSTNESS_SIGNAL_COLLAPSE: ROC-AUC is not above 0.50.")
    if test_audit and any(
        test_audit.get(key) not in expected
        for key, expected in {
            "test_target_rows_loaded": (0,),
            "test_predictions_created": (0,),
            "test_metrics_computed": (False,),
        }.items()
    ):
        blockers.append("TEST_ACCESS_VIOLATION")
    unique_warnings = sorted(set(str(item) for item in warnings))
    status = "BLOCKED" if blockers else ("READY_WITH_WARNINGS" if unique_warnings else "READY")
    return {
        "status": status,
        "safe_to_start_phase15": not blockers,
        "hard_blockers": sorted(set(blockers)),
        "warnings": unique_warnings,
        "development_decisions_frozen": True,
        "model_changes_after_phase14_analysis": "prohibited",
    }


__all__ = ["readiness_gate"]
