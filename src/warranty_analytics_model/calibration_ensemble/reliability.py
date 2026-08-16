"""Deterministic equal-frequency reliability diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def reliability_bins(
    y_true: Any,
    probabilities: Any,
    *,
    bins: int = 10,
    keys: Any | None = None,
) -> pd.DataFrame:
    """Build deterministic near-equal-frequency calibration bins.

    Sorting by probability and then by claim key makes tied probabilities stable
    without relying on pandas/qcut duplicate-bin behavior.
    """

    y = np.asarray(y_true, dtype="int8").reshape(-1)
    p = np.asarray(probabilities, dtype="float64").reshape(-1)
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("Reliability inputs must be finite and equally sized.")
    if not np.isin(y, [0, 1]).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Reliability inputs must be binary and bounded probabilities.")
    if bins < 1:
        raise ValueError("Reliability bin count must be positive.")
    if keys is None:
        tie_keys = np.arange(len(p), dtype="int64")
    else:
        tie_keys = np.asarray(keys).reshape(-1)
        if len(tie_keys) != len(p):
            raise ValueError("Reliability keys must match probability rows.")
    order = np.lexsort((tie_keys, p))
    chunks = np.array_split(order, min(int(bins), len(order)))
    rows: list[dict[str, float | int]] = []
    for index, chunk in enumerate(chunks, start=1):
        if len(chunk) == 0:
            continue
        mean_probability = float(np.mean(p[chunk]))
        observed_rate = float(np.mean(y[chunk]))
        rows.append(
            {
                "bin_id": index,
                "row_count": int(len(chunk)),
                "mean_predicted_probability": mean_probability,
                "observed_positive_rate": observed_rate,
                "absolute_calibration_error": abs(mean_probability - observed_rate),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "bin_id",
            "row_count",
            "mean_predicted_probability",
            "observed_positive_rate",
            "absolute_calibration_error",
        ],
    )


def ece_mce(
    y_true: Any, probabilities: Any, *, bins: int = 10, keys: Any | None = None
) -> tuple[float, float]:
    table = reliability_bins(y_true, probabilities, bins=bins, keys=keys)
    total = float(table["row_count"].sum())
    ece = (
        float((table["row_count"] / total * table["absolute_calibration_error"]).sum())
        if total
        else 0.0
    )
    mce = float(table["absolute_calibration_error"].max()) if not table.empty else 0.0
    return ece, mce


__all__ = ["ece_mce", "reliability_bins"]
