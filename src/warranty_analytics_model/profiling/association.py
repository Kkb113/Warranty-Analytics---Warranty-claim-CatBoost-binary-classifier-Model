"""Descriptive target associations without model fitting."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .column_profile import is_sensitive_column


def point_biserial(values: pd.Series, target: pd.Series) -> float | None:
    """Compute point-biserial correlation using two target groups."""

    numeric = pd.to_numeric(values, errors="coerce")
    labels = pd.to_numeric(target, errors="coerce")
    mask = numeric.notna() & labels.isin([0, 1])
    x = numeric[mask].to_numpy(dtype=float)
    y = labels[mask].to_numpy(dtype=float)
    if len(x) < 2 or len(np.unique(y)) != 2 or float(np.std(x, ddof=1)) == 0.0:
        return None
    positive = x[y == 1]
    negative = x[y == 0]
    pooled = float(np.std(x, ddof=1))
    return float(
        (positive.mean() - negative.mean())
        / pooled
        * np.sqrt(len(positive) * len(negative) / len(x) ** 2)
    )


def cramer_v(values: pd.Series, target: pd.Series) -> float | None:
    """Compute bias-uncorrected Cramer's V for a categorical field."""

    frame = pd.DataFrame({"value": values, "target": target}).dropna()
    frame = frame[frame["target"].isin([0, 1])]
    if frame.empty or frame["value"].nunique() < 2 or frame["target"].nunique() < 2:
        return None
    table = pd.crosstab(frame["value"], frame["target"]).to_numpy(dtype=float)
    expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / table.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_square = np.nansum(np.where(expected > 0, (table - expected) ** 2 / expected, 0.0))
    n = table.sum()
    denominator = n * min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt(chi_square / denominator)) if denominator > 0 else None


def _target_rate_range(values: pd.Series, target: pd.Series) -> list[float] | None:
    frame = pd.DataFrame({"value": values, "target": target}).dropna()
    if frame.empty:
        return None
    rates = frame.groupby("value", dropna=False, observed=True)["target"].mean()
    return [round(float(rates.min()) * 100, 6), round(float(rates.max()) * 100, 6)]


def association_table(
    frame: pd.DataFrame,
    target_column: str = "high_cost_claim_flag",
    *,
    columns: Iterable[str] | None = None,
    leakage_columns: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Create an aggregate association table with no row-level examples."""

    if target_column not in frame:
        return []
    target = pd.to_numeric(frame[target_column], errors="coerce")
    selected = list(columns or frame.columns)
    leakage = {column.casefold() for column in leakage_columns}
    output: list[dict[str, Any]] = []
    for column in selected:
        if column == target_column or column not in frame:
            continue
        series = frame[column]
        is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
            series
        )
        if is_sensitive_column(str(column)) and str(column).casefold() not in leakage:
            # Identifier/text fields are audited separately, not presented as model candidates.
            continue
        if is_numeric:
            value = point_biserial(series, target)
            measure = "point_biserial"
            field_type = "numeric"
        else:
            value = cramer_v(series.astype("string"), target)
            measure = "cramers_v"
            field_type = "categorical"
        if value is None:
            continue
        output.append(
            {
                "field": str(column),
                "field_type": field_type,
                "sample_size": int(pd.DataFrame({"x": series, "y": target}).dropna().shape[0]),
                "association_measure": measure,
                "association_value": round(float(value), 8),
                "target_rate_range": _target_rate_range(series, target),
                "leakage_status": "suspected_post_outcome"
                if column.casefold() in leakage
                else "diagnostic_only",
                "notes": "Association is descriptive and not causal; validate availability at claim submission.",
            }
        )
    return sorted(output, key=lambda item: abs(float(item["association_value"])), reverse=True)


def missingness_by_target(
    frame: pd.DataFrame,
    target_column: str = "high_cost_claim_flag",
    columns: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    """Compare missingness rates by target without exposing values."""

    if target_column not in frame:
        return []
    target = pd.to_numeric(frame[target_column], errors="coerce")
    selected = list(columns or frame.columns)
    rows: list[dict[str, object]] = []
    for column in selected:
        if column == target_column or column not in frame:
            continue
        rates: dict[str, float | None] = {}
        for label in (0, 1):
            group = frame.loc[target == label, column]
            rates[str(label)] = round(float(group.isna().mean() * 100), 6) if len(group) else None
        difference = None
        if rates["0"] is not None and rates["1"] is not None:
            difference = round(abs(rates["1"] - rates["0"]), 6)
        rows.append(
            {
                "field": str(column),
                "missing_rate_by_target": rates,
                "absolute_percentage_point_difference": difference,
                "suspected_leakage": bool(difference is not None and difference >= 90.0),
            }
        )
    return rows
