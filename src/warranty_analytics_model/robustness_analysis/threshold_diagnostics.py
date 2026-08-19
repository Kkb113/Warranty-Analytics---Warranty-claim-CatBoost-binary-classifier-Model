"""Frozen threshold and diagnostic-only sensitivity table."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import overall_metrics


def threshold_sensitivity(
    y_true: Any,
    probabilities: Any,
    frozen_threshold: float,
    multipliers: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for multiplier in multipliers:
        threshold = float(np.clip(float(frozen_threshold) * float(multiplier), 0.001, 0.999))
        metrics = overall_metrics(y_true, probabilities, threshold)
        rows.append(
            {
                "multiplier": float(multiplier),
                "threshold": threshold,
                "DO_NOT_USE_FOR_THRESHOLD_SELECTION": True,
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("multiplier", kind="mergesort").reset_index(drop=True)


__all__ = ["threshold_sensitivity"]
