"""TRAIN-cross-fitted calibrated threshold selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..imbalance_threshold.metrics import threshold_metrics


def threshold_grid(
    start: float = 0.001, stop: float = 0.999, step: float = 0.001
) -> tuple[float, ...]:
    if not (0 < start < stop < 1 and step > 0):
        raise ValueError("Invalid Phase 13 threshold grid.")
    count = int(round((stop - start) / step)) + 1
    values = tuple(float(f"{start + index * step:.3f}") for index in range(count))
    if len(values) != 999 or values[0] != 0.001 or values[-1] != 0.999:
        raise ValueError("Phase 13 threshold grid must contain exactly 0.001..0.999.")
    return values


def build_threshold_curve(
    y_true: Any,
    probabilities: Any,
    *,
    candidate_id: str,
    score_space: str,
    start: float = 0.001,
    stop: float = 0.999,
    step: float = 0.001,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype="int8")
    p = np.asarray(probabilities, dtype="float64")
    rows: list[dict[str, Any]] = []
    for threshold in threshold_grid(start, stop, step):
        rows.append(
            {
                "candidate_id": candidate_id,
                "score_space": score_space,
                "threshold": threshold,
                **threshold_metrics(y, p, threshold),
            }
        )
    return pd.DataFrame(rows)


def _better(left: dict[str, Any], right: dict[str, Any], tolerance: float) -> bool:
    for metric, higher in (
        ("mcc", True),
        ("f2", True),
        ("recall", True),
        ("precision", True),
    ):
        delta = float(left[metric]) - float(right[metric])
        if abs(delta) > tolerance:
            return delta > 0 if higher else delta < 0
    return float(left["threshold"]) < float(right["threshold"])


def select_mcc_threshold(curve: pd.DataFrame, tie_tolerance: float = 1.0e-12) -> dict[str, Any]:
    if curve.empty:
        raise ValueError("Cannot select a threshold from an empty curve.")
    rows = curve.to_dict("records")
    selected = rows[0]
    for row in rows[1:]:
        if _better(row, selected, tie_tolerance):
            selected = row
    return {
        "method": "MCC_MAX",
        "threshold": float(selected["threshold"]),
        "metrics": {
            key: value
            for key, value in selected.items()
            if key not in {"candidate_id", "score_space", "threshold"}
        },
        "tie_tolerance": tie_tolerance,
        "score_space": selected.get("score_space"),
    }


__all__ = ["build_threshold_curve", "select_mcc_threshold", "threshold_grid"]
