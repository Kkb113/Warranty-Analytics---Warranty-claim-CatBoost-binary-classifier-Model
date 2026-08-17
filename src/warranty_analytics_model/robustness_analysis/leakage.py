"""Recheck the actual frozen champion feature names against Phase 4 blacklists."""

from __future__ import annotations

from typing import Any

from ..baseline_model.input import PROHIBITED_SOURCE_SUFFIXES


def leakage_recheck(
    feature_names: list[str], lineage: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    prohibited: list[str] = []
    for name in feature_names:
        suffix = str(name).split("__")[-1]
        if suffix in PROHIBITED_SOURCE_SUFFIXES:
            prohibited.append(str(name))
        if any(
            token in str(name).lower()
            for token in ("total_claim_cost", "repair_end_date", "root_cause", "approval_status")
        ):
            prohibited.append(str(name))
        if (
            lineage
            and isinstance(lineage.get(name), dict)
            and lineage[name].get("target_dependent") is not False
        ):
            prohibited.append(str(name))
    unique = sorted(set(prohibited))
    return {
        "prohibited_features": unique,
        "prohibited_feature_count": len(unique),
        "valid": not unique,
    }


__all__ = ["leakage_recheck"]
