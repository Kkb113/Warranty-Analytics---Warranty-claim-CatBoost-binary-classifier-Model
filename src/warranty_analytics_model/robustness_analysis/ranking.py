"""Frozen TRAIN-OOF risk deciles and deterministic top-k lift."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def risk_decile_metrics(
    train_oof: pd.DataFrame,
    validation: pd.DataFrame,
    targets: pd.Series,
    probabilities: pd.Series,
) -> pd.DataFrame:
    if "warranty_claim_key" not in train_oof or "probability" not in train_oof:
        raise ValueError("TRAIN OOF scores require warranty_claim_key and probability.")
    values = np.asarray(train_oof["probability"], dtype="float64")
    edges = np.unique(np.quantile(values, np.linspace(0, 1, 11))) if len(values) else np.array([])
    if len(edges) < 2:
        labels = pd.Series("D1", index=validation.index)
    else:
        bins = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
        labels = pd.cut(
            np.asarray(probabilities, dtype="float64"),
            bins=bins,
            labels=[f"D{i}" for i in range(1, len(bins))],
            include_lowest=True,
        ).astype("string")
    frame = pd.DataFrame(
        {
            "decile": labels,
            "target": np.asarray(targets, dtype="int8"),
            "probability": np.asarray(probabilities, dtype="float64"),
        }
    )
    rows: list[dict[str, Any]] = []
    total_positive = int(frame["target"].sum())
    cumulative = 0
    for decile, group in frame.groupby("decile", sort=False, observed=False):
        positives = int(group["target"].sum())
        cumulative += positives
        prevalence = float(group["target"].mean()) if len(group) else 0.0
        rows.append(
            {
                "decile": str(decile),
                "row_count": int(len(group)),
                "positive_count": positives,
                "prevalence": prevalence,
                "precision": float(positives / len(group)) if len(group) else 0.0,
                "recall_contribution": float(positives / total_positive) if total_positive else 0.0,
                "cumulative_recall": float(cumulative / total_positive) if total_positive else 0.0,
                "lift": float(prevalence / frame["target"].mean())
                if frame["target"].mean()
                else 0.0,
                "mean_predicted_probability": float(group["probability"].mean())
                if len(group)
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def topk_lift(
    keys: Any,
    targets: Any,
    probabilities: Any,
    fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30),
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "key": np.asarray(keys),
            "target": np.asarray(targets, dtype="int8"),
            "probability": np.asarray(probabilities, dtype="float64"),
        }
    )
    frame = frame.sort_values(
        ["probability", "key"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    prevalence = float(frame["target"].mean()) if len(frame) else 0.0
    result: dict[str, Any] = {"baseline_prevalence": prevalence, "rows": []}
    for fraction in fractions:
        count = int(np.ceil(float(fraction) * len(frame)))
        selected = frame.iloc[:count]
        captured = int(selected["target"].sum())
        result["rows"].append(
            {
                "fraction": float(fraction),
                "claims_selected": count,
                "high_cost_claims_captured": captured,
                "recall_capture_rate": float(captured / frame["target"].sum())
                if frame["target"].sum()
                else 0.0,
                "precision": float(selected["target"].mean()) if len(selected) else 0.0,
                "baseline_prevalence": prevalence,
                "lift_over_random": float(
                    (selected["target"].mean() / prevalence)
                    if prevalence and len(selected)
                    else 0.0
                ),
            }
        )
    return result


__all__ = ["risk_decile_metrics", "topk_lift"]
