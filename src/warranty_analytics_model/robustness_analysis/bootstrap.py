"""Deterministic stratified percentile bootstrap."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .metrics import overall_metrics


def _bootstrap_one(
    seed: int,
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
) -> tuple[int, dict[str, float] | None, str | None]:
    rng = np.random.default_rng(int(seed))
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return int(seed), None, "both_classes_required"
    indices = np.concatenate(
        [rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg), replace=True)]
    )
    try:
        metrics = overall_metrics(y[indices], p[indices], threshold)
        keys = (
            "average_precision",
            "roc_auc",
            "log_loss",
            "brier_score",
            "precision",
            "recall",
            "f1",
            "f2",
            "mcc",
            "balanced_accuracy",
        )
        return int(seed), {key: float(metrics.get(key, np.nan)) for key in keys}, None
    except (ValueError, FloatingPointError, ZeroDivisionError) as exc:
        return int(seed), None, type(exc).__name__


def stratified_bootstrap(
    y_true: Any,
    probabilities: Any,
    threshold: float,
    *,
    replicates: int,
    seed: int = 20260810,
    workers: int = 1,
    confidence_level: float = 0.95,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return deterministic point/percentile intervals and every replicate outcome."""

    y = np.asarray(y_true, dtype="int8").reshape(-1)
    p = np.asarray(probabilities, dtype="float64").reshape(-1)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("Bootstrap inputs must be non-empty and equally sized.")
    if replicates < 1:
        raise ValueError("Bootstrap replicate count must be positive.")
    sequence = np.random.SeedSequence(int(seed))
    seeds = [
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in sequence.spawn(int(replicates))
    ]

    def task(value: int) -> tuple[int, dict[str, float] | None, str | None]:
        return _bootstrap_one(value, y, p, float(threshold))

    if int(workers) > 1:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            raw = list(executor.map(task, seeds))
    else:
        raw = [task(value) for value in seeds]
    raw.sort(key=lambda item: item[0])
    rows: list[dict[str, Any]] = []
    failures = 0
    for index, (_replicate_seed, replicate_values, error) in enumerate(raw):
        row: dict[str, Any] = {"replicate": index, "seed": int(_replicate_seed), "error": error}
        if replicate_values is None:
            failures += 1
            row["valid"] = False
            for key in (
                "average_precision",
                "roc_auc",
                "log_loss",
                "brier_score",
                "precision",
                "recall",
                "f1",
                "f2",
                "mcc",
                "balanced_accuracy",
            ):
                row[key] = np.nan
        else:
            row.update(replicate_values)
            row["valid"] = True
        rows.append(row)
    point = overall_metrics(y, p, threshold)
    alpha = (1.0 - float(confidence_level)) / 2.0
    summary: dict[str, Any] = {
        "seed": int(seed),
        "replicate_count": int(replicates),
        "failed_replicate_count": int(failures),
        "confidence_level": float(confidence_level),
        "method": "STRATIFIED_PERCENTILE",
    }
    for key in (
        "average_precision",
        "roc_auc",
        "mcc",
        "recall",
        "log_loss",
        "brier_score",
        "precision",
        "f1",
        "f2",
        "balanced_accuracy",
    ):
        metric_values = np.asarray([row[key] for row in rows], dtype="float64")
        finite = metric_values[np.isfinite(metric_values)]
        summary[key] = {
            "point_estimate": float(point.get(key, np.nan)),
            "ci_lower": float(np.quantile(finite, alpha)) if len(finite) else None,
            "ci_upper": float(np.quantile(finite, 1.0 - alpha)) if len(finite) else None,
            "replicate_count": int(replicates),
            "failed_replicate_count": int(failures),
        }
    return summary, rows


__all__ = ["stratified_bootstrap"]
