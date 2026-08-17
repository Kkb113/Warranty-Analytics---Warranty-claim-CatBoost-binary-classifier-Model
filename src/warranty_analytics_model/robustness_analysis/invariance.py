"""Serialization, row-order, and batch-size prediction invariance."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .input import KEY


def prediction_invariance(
    frame: pd.DataFrame,
    scorer: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    fresh_scorer: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    batch_sizes: tuple[int, ...] = (17, 64, 256),
    seed: int = 20260810,
    tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    baseline = scorer(frame).set_index(KEY)["probability"].sort_index()
    # ``fresh_scorer`` must be a newly constructed scorer that reloads the
    # frozen CatBoost CBM, calibrator JSON, and ensemble policy from disk.  A
    # caller that omits it retains the lightweight historical fallback for
    # isolated unit fixtures, but production Phase 14 always supplies it.
    reloaded = (fresh_scorer or scorer)(frame)
    serialized = reloaded.set_index(KEY)["probability"].sort_index()
    serialization_delta = (
        float(np.max(np.abs(baseline.to_numpy() - serialized.to_numpy()))) if len(baseline) else 0.0
    )
    shuffled = frame.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    row_order = scorer(shuffled).set_index(KEY)["probability"].sort_index()
    row_delta = (
        float(np.max(np.abs(baseline.to_numpy() - row_order.to_numpy()))) if len(baseline) else 0.0
    )
    batch_deltas: dict[str, float] = {}
    for size in (len(frame), *batch_sizes):
        parts = []
        for start in range(0, len(frame), int(size)):
            parts.append(scorer(frame.iloc[start : start + int(size)]))
        combined = pd.concat(parts, ignore_index=True).set_index(KEY)["probability"].sort_index()
        batch_deltas[str(size)] = (
            float(np.max(np.abs(baseline.to_numpy() - combined.to_numpy())))
            if len(baseline)
            else 0.0
        )
    result = {
        "seed": int(seed),
        "serialization_max_probability_delta": serialization_delta,
        "row_order_max_probability_delta": row_delta,
        "batch_max_probability_delta": max(batch_deltas.values(), default=0.0),
        "batch_size_deltas": batch_deltas,
        "serialization_reload_verified": fresh_scorer is not None,
        "tolerance": float(tolerance),
    }
    max_delta = max(serialization_delta, row_delta, max(batch_deltas.values(), default=0.0))
    result["valid"] = bool(max_delta <= float(tolerance))
    return result


__all__ = ["prediction_invariance"]
