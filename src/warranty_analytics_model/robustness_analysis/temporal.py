"""Temporal robustness views built from frozen date-only partitions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import Phase14Settings
from .metrics import safe_metric_dict


def temporal_metrics(
    frame: pd.DataFrame,
    targets: pd.Series,
    probabilities: pd.Series,
    threshold: float,
    settings: Phase14Settings,
    *,
    overall: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dates = pd.to_datetime(frame["claim__claim_date"], errors="coerce")
    masks: list[tuple[str, pd.Series]] = []
    masks.extend(
        (str(value), dates.dt.to_period("M").astype("string") == str(value))
        for value in sorted(dates.dt.to_period("M").dropna().astype(str).unique())
    )
    masks.extend(
        (str(value), dates.dt.to_period("Q").astype("string") == str(value))
        for value in sorted(dates.dt.to_period("Q").dropna().astype(str).unique())
    )
    ordered = (
        frame[["warranty_claim_key", "claim__claim_date"]]
        .assign(_date=dates)
        .sort_values(["_date", "warranty_claim_key"], kind="mergesort")
    )
    third = pd.Series("EARLY", index=ordered.index, dtype="string")
    n = len(ordered)
    third.iloc[n // 3 : 2 * n // 3] = "MIDDLE"
    third.iloc[2 * n // 3 :] = "LATE"
    third_by_key = pd.Series(third.to_numpy(), index=ordered["warranty_claim_key"].astype(int))
    masks.extend(
        (label, frame["warranty_claim_key"].astype(int).map(third_by_key) == label)
        for label in ("EARLY", "MIDDLE", "LATE")
    )
    y = pd.Series(targets.to_numpy(), index=frame.index)
    p = pd.Series(probabilities.to_numpy(), index=frame.index)
    for label, mask in masks:
        metrics = safe_metric_dict(y.loc[mask].to_numpy(), p.loc[mask].to_numpy(), threshold)
        row: dict[str, Any] = {
            "period": label,
            "period_type": "calendar"
            if label not in {"EARLY", "MIDDLE", "LATE"}
            else "chronological_third",
        }
        row.update(metrics)
        if row.get("status") == "SUPPORTED" and row.get("average_precision") is not None:
            row["ap_lift_over_prevalence"] = (
                float(row["average_precision"]) / float(row["prevalence"])
                if float(row["prevalence"])
                else None
            )
            row["ap_ratio_to_overall"] = (
                float(row["average_precision"]) / float(overall["average_precision"])
                if float(overall["average_precision"])
                else None
            )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["period_type", "period"], kind="mergesort")
        .reset_index(drop=True)
    )


__all__ = ["temporal_metrics"]
