"""Phase 15 ranking helpers with deterministic tie handling."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..robustness_analysis.ranking import risk_decile_metrics, topk_lift


def concentration_summary(deciles: pd.DataFrame) -> dict[str, Any]:
    if deciles.empty:
        return {
            "positive_share_d10": 0.0,
            "positive_share_d10_d9": 0.0,
            "positive_share_d10_d8": 0.0,
        }
    total = int(deciles["positive_count"].sum())
    by = deciles.set_index("decile")

    def share(labels: list[str]) -> float:
        count = int(by.loc[by.index.intersection(labels), "positive_count"].sum())
        return float(count / total) if total else 0.0

    d10 = by.loc[by.index == "D10", "prevalence"]
    d1 = by.loc[by.index == "D1", "prevalence"]
    return {
        "positive_share_d10": share(["D10"]),
        "positive_share_d10_d9": share(["D10", "D9"]),
        "positive_share_d10_d8": share(["D10", "D9", "D8"]),
        "d10_lift": float(by.loc["D10", "lift"]) if "D10" in by.index else 0.0,
        "highest_risk_decile_prevalence": float(d10.iloc[0]) if len(d10) else 0.0,
        "lowest_risk_decile_prevalence": float(d1.iloc[0]) if len(d1) else 0.0,
    }


__all__ = ["concentration_summary", "risk_decile_metrics", "topk_lift"]
