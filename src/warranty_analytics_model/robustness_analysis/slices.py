"""Frozen slice membership and slice-metric evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import Phase14Settings
from .metrics import safe_metric_dict


def membership_for_definition(
    definition: dict[str, Any],
    frame: pd.DataFrame,
    *,
    scores: pd.Series | np.ndarray | None = None,
) -> pd.Series:
    kind = str(definition.get("kind"))
    index = frame.index
    if kind == "categorical":
        column = str(definition["column"])
        values = frame[column].astype("string").fillna("__MISSING__")
        categories = {str(item) for item in definition.get("categories", [])}
        return values.map(
            lambda value: (
                str(value)
                if str(value) in categories and str(value) != "__OTHER__"
                else "__OTHER__"
            )
        ).astype(str)
    if kind == "numeric_quantile":
        values = pd.to_numeric(frame[str(definition["column"])], errors="coerce")
        edges = np.asarray(definition.get("edges", []), dtype="float64")
        if len(edges) < 2:
            return pd.Series("UNAVAILABLE", index=index, dtype="string")
        bins = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
        labels = [f"Q{i}" for i in range(1, len(bins))]
        return (
            pd.cut(values, bins=bins, labels=labels, include_lowest=True)
            .astype("string")
            .fillna("__MISSING__")
        )
    if kind in {"risk_score_band", "risk_score_decile"}:
        if scores is None:
            raise ValueError("Risk-score slices require scores.")
        values = pd.Series(np.asarray(scores, dtype="float64"), index=index)
        edges = np.asarray(definition.get("edges", []), dtype="float64")
        if len(edges) < 2:
            return pd.Series("UNAVAILABLE", index=index, dtype="string")
        bins = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
        prefix = "D" if kind == "risk_score_decile" else "B"
        labels = [f"{prefix}{i}" for i in range(1, len(bins))]
        return (
            pd.cut(values, bins=bins, labels=labels, include_lowest=True)
            .astype("string")
            .fillna("__MISSING__")
        )
    if kind == "feature_missingness_band":
        names = [
            str(name) for name in definition.get("feature_names", []) if str(name) in frame.columns
        ]
        counts = frame[names].isna().sum(axis=1) if names else pd.Series(0, index=index)
        edges = np.asarray(definition.get("edges", []), dtype="float64")
        if len(edges) < 2:
            return pd.Series("UNAVAILABLE", index=index, dtype="string")
        bins = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
        labels = [f"M{i}" for i in range(1, len(bins))]
        return (
            pd.cut(counts, bins=bins, labels=labels, include_lowest=True)
            .astype("string")
            .fillna("MISSING")
        )
    if kind == "calendar_month":
        dates = pd.to_datetime(frame[str(definition["column"])], errors="coerce")
        return dates.dt.to_period("M").astype("string").fillna("__MISSING__")
    if kind == "calendar_quarter":
        dates = pd.to_datetime(frame[str(definition["column"])], errors="coerce")
        return dates.dt.to_period("Q").astype("string").fillna("__MISSING__")
    if kind == "chronological_thirds":
        mapping = {
            int(key): label
            for label, keys in dict(definition.get("membership_keys", {})).items()
            for key in keys
        }
        # ``Index.map`` returns an Index, which drops the Series index contract
        # expected by the evaluator (and therefore has no ``.index`` itself).
        # Preserve row alignment explicitly so slice membership remains
        # deterministic after any upstream sorting/resetting.
        return pd.Series(
            [mapping.get(int(key), "UNAVAILABLE") for key in frame["warranty_claim_key"]],
            index=index,
            dtype="string",
        )
    raise ValueError(f"Unsupported frozen slice definition: {kind}")


def evaluate_slices(
    definitions: list[dict[str, Any]],
    frame: pd.DataFrame,
    targets: pd.Series,
    probabilities: pd.Series,
    threshold: float,
    overall: dict[str, Any],
    settings: Phase14Settings,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_memberships: set[tuple[int, str]] = set()
    y = pd.Series(np.asarray(targets, dtype="int8"), index=frame.index)
    p = pd.Series(np.asarray(probabilities, dtype="float64"), index=frame.index)
    for definition in definitions:
        memberships = membership_for_definition(definition, frame, scores=p)
        if memberships.index.has_duplicates:
            raise ValueError(f"Duplicate membership index for {definition.get('slice_id')}.")
        for label in sorted(str(item) for item in memberships.dropna().unique()):
            mask = memberships.astype(str) == label
            keys = frame.loc[mask, "warranty_claim_key"].astype(int).tolist()
            marker = (id(definition), label)
            if marker in seen_memberships:
                raise ValueError("Duplicate slice membership detected.")
            seen_memberships.add(marker)
            metrics = safe_metric_dict(y.loc[mask].to_numpy(), p.loc[mask].to_numpy(), threshold)
            row: dict[str, Any] = {
                "slice_id": str(definition["slice_id"]),
                "slice_kind": str(definition["kind"]),
                "slice_label": label,
                "claim_key_count": len(keys),
                "positive_count": int((y.loc[mask] == 1).sum()),
                "negative_count": int((y.loc[mask] == 0).sum()),
                "prevalence": float(y.loc[mask].mean()) if mask.any() else 0.0,
                "target_independent_definition": True,
            }
            row.update(metrics)
            if row.get("status") == "SUPPORTED" and metrics.get("average_precision") is not None:
                row["ap_ratio_to_overall"] = (
                    float(metrics["average_precision"]) / float(overall["average_precision"])
                    if float(overall["average_precision"])
                    else None
                )
                row["ap_lift_over_slice_prevalence"] = (
                    float(metrics["average_precision"]) / float(row["prevalence"])
                    if float(row["prevalence"])
                    else None
                )
                row["roc_auc_difference"] = float(metrics["roc_auc"]) - float(overall["roc_auc"])
                row["log_loss_difference"] = float(metrics["log_loss"]) - float(overall["log_loss"])
                row["brier_difference"] = float(metrics["brier_score"]) - float(
                    overall["brier_score"]
                )
                row["recall_difference"] = float(metrics.get("recall", 0.0)) - float(
                    overall.get("recall", 0.0)
                )
                row["mcc_difference"] = float(metrics.get("mcc", 0.0)) - float(
                    overall.get("mcc", 0.0)
                )
                slice_ap = float(metrics["average_precision"])
                if slice_ap <= float(row["prevalence"]) or float(metrics["roc_auc"]) <= 0.5:
                    row["stability_classification"] = "SEVERE_DEGRADATION"
                elif (
                    slice_ap < 0.75 * float(overall["average_precision"])
                    or float(metrics.get("mcc", 0.0)) < float(overall.get("mcc", 0.0)) - 0.10
                ):
                    row["stability_classification"] = "MODERATE_DEGRADATION"
                else:
                    row["stability_classification"] = "STABLE"
            else:
                row["stability_classification"] = "LOW_SUPPORT"
            rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["slice_id", "slice_label"], kind="mergesort").reset_index(
            drop=True
        )
    summary = {
        "total_slices": int(len(result)),
        "supported_slices": int((result.get("status", pd.Series(dtype=str)) == "SUPPORTED").sum())
        if not result.empty
        else 0,
        "low_support_slices": int(
            (result.get("status", pd.Series(dtype=str)) == "LOW_SUPPORT").sum()
        )
        if not result.empty
        else 0,
        "severe_degradation_warnings": int(
            (
                result.get("stability_classification", pd.Series(dtype=str)) == "SEVERE_DEGRADATION"
            ).sum()
        )
        if not result.empty
        else 0,
    }
    return result, summary


__all__ = ["evaluate_slices", "membership_for_definition"]
