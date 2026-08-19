"""Aggregate and local-only error cohort diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .slices import membership_for_definition

KEY = "warranty_claim_key"


def error_cohorts(keys: Any, targets: Any, probabilities: Any, threshold: float) -> pd.DataFrame:
    y = np.asarray(targets, dtype="int8")
    p = np.asarray(probabilities, dtype="float64")
    predicted = p >= float(threshold)
    error = np.where(
        (predicted == 1) & (y == 1),
        "TRUE_POSITIVE",
        np.where(
            (predicted == 1) & (y == 0),
            "FALSE_POSITIVE",
            np.where((predicted == 0) & (y == 1), "FALSE_NEGATIVE", "TRUE_NEGATIVE"),
        ),
    )
    margin = np.where(predicted, p - float(threshold), float(threshold) - p)
    return (
        pd.DataFrame(
            {
                "warranty_claim_key": np.asarray(keys, dtype="int64"),
                "target": y,
                "probability": p,
                "frozen_threshold": float(threshold),
                "predicted_class": predicted.astype("int8"),
                "error_type": error,
                "threshold_margin": margin,
            }
        )
        .sort_values("warranty_claim_key", kind="mergesort")
        .reset_index(drop=True)
    )


def high_confidence_errors(cohorts: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    selected = cohorts.loc[cohorts["error_type"].isin(["FALSE_POSITIVE", "FALSE_NEGATIVE"])].copy()
    return (
        selected.sort_values(
            ["error_type", "threshold_margin", "warranty_claim_key"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("error_type", sort=True, group_keys=False)
        .head(int(limit))
        .reset_index(drop=True)
    )


def build_error_context(
    frame: pd.DataFrame,
    definitions: list[dict[str, Any]],
    probabilities: pd.Series | np.ndarray,
    feature_names: tuple[str, ...] | list[str] = (),
) -> pd.DataFrame:
    """Materialize the frozen domain/slice registry for error profiling.

    Membership is derived only from prediction-time features and TRAIN-frozen
    definitions.  No target or new feature is introduced.  Raw model features
    are retained as well so numeric/categorical FP/FN differences can be
    reported alongside domain, history, mileage, warranty, and risk cohorts.
    """

    context = frame[[KEY]].copy().reset_index(drop=True)
    scores = pd.Series(np.asarray(probabilities, dtype="float64"), index=frame.index)
    for definition in definitions:
        slice_id = str(definition.get("slice_id"))
        membership = membership_for_definition(definition, frame, scores=scores)
        context[f"slice:{slice_id}"] = membership.astype("string").to_numpy()
    names = [str(name) for name in feature_names if str(name) in frame.columns]
    context["feature_missingness_count"] = frame[names].isna().sum(axis=1) if names else 0
    for name in names:
        context[f"feature:{name}"] = frame[name].reset_index(drop=True)
    return context


def error_profile(cohorts: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Profile every error class across frozen cohorts and raw feature values.

    Aggregate rows preserve the original summary while long-form cohort rows
    expose where FP/FN/TP/TN concentrate.  Numeric contexts report mean shifts;
    categorical contexts report concentration versus population prevalence.
    """

    frame = cohorts.merge(context, on=KEY, how="left", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    context_columns = [column for column in context.columns if column != KEY]
    for error_type, group in frame.groupby("error_type", sort=True):
        error_name = str(error_type)
        aggregate: dict[str, Any] = {
            "error_type": error_name,
            "profile_kind": "aggregate",
            "context_name": "ALL",
            "context_value": "ALL",
            "row_count": int(len(group)),
            "population_row_count": int(len(frame)),
            "concentration_rate": 1.0,
            "population_rate": float(len(group) / len(frame)) if len(frame) else 0.0,
            "concentration_lift": 1.0,
            "mean_probability": float(group["probability"].mean()) if len(group) else 0.0,
            "mean_threshold_margin": float(group["threshold_margin"].mean()) if len(group) else 0.0,
            "context_mean": None,
            "population_mean": None,
            "mean_difference": None,
            "missing_count": 0,
        }
        rows.append(aggregate)
        for column in context_columns:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce")
                group_values = pd.to_numeric(group[column], errors="coerce")
                group_mean = float(group_values.mean()) if group_values.notna().any() else None
                population_mean = float(numeric.mean()) if numeric.notna().any() else None
                rows.append(
                    {
                        "error_type": error_name,
                        "profile_kind": "numeric_summary",
                        "context_name": str(column),
                        "context_value": "__NUMERIC__",
                        "row_count": int(len(group)),
                        "population_row_count": int(len(frame)),
                        "concentration_rate": 1.0,
                        "population_rate": float(len(group) / len(frame)) if len(frame) else 0.0,
                        "concentration_lift": 1.0,
                        "mean_probability": float(group["probability"].mean())
                        if len(group)
                        else 0.0,
                        "mean_threshold_margin": float(group["threshold_margin"].mean())
                        if len(group)
                        else 0.0,
                        "context_mean": group_mean,
                        "population_mean": population_mean,
                        "mean_difference": (
                            group_mean - population_mean
                            if group_mean is not None and population_mean is not None
                            else None
                        ),
                        "missing_count": int(group_values.isna().sum()),
                    }
                )
                continue
            normalized = series.astype("string").fillna("__MISSING__").astype(str)
            group_normalized = normalized.loc[group.index]
            population_counts = normalized.value_counts(dropna=False)
            for value in sorted(population_counts.index.astype(str).tolist()):
                count = int((group_normalized == value).sum())
                population_count = int(population_counts.get(value, 0))
                concentration = float(count / len(group)) if len(group) else 0.0
                population_rate = float(population_count / len(frame)) if len(frame) else 0.0
                rows.append(
                    {
                        "error_type": error_name,
                        "profile_kind": "categorical_cohort",
                        "context_name": str(column),
                        "context_value": value,
                        "row_count": count,
                        "population_row_count": population_count,
                        "concentration_rate": concentration,
                        "population_rate": population_rate,
                        "concentration_lift": (
                            concentration / population_rate if population_rate else None
                        ),
                        "mean_probability": float(group["probability"].mean())
                        if len(group)
                        else 0.0,
                        "mean_threshold_margin": float(group["threshold_margin"].mean())
                        if len(group)
                        else 0.0,
                        "context_mean": None,
                        "population_mean": None,
                        "mean_difference": None,
                        "missing_count": int((group_normalized == "__MISSING__").sum()),
                    }
                )
    return pd.DataFrame(rows)


__all__ = [
    "build_error_context",
    "error_cohorts",
    "error_profile",
    "high_confidence_errors",
]
