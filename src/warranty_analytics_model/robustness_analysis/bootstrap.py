"""Deterministic stratified percentile bootstrap."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from .metrics import overall_metrics
from .slices import membership_for_definition


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


def material_slice_bootstrap(
    slices: pd.DataFrame,
    definitions: list[dict[str, Any]],
    frame: pd.DataFrame,
    targets: Any,
    probabilities: Any,
    threshold: float,
    *,
    replicates: int,
    seed: int,
    workers: int,
    confidence_level: float,
    min_positive: int = 8,
    min_negative: int = 20,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Bootstrap every supported/material frozen slice deterministically."""

    y = np.asarray(targets, dtype="int8").reshape(-1)
    p = np.asarray(probabilities, dtype="float64").reshape(-1)
    if len(frame) != len(y) or len(y) != len(p):
        raise ValueError("Material slice bootstrap inputs must be equally sized.")
    definition_by_id = {str(item.get("slice_id")): item for item in definitions}
    selected = slices.loc[
        (slices.get("status", pd.Series(dtype=str)) == "SUPPORTED")
        & (pd.to_numeric(slices.get("positive_count", 0), errors="coerce") >= int(min_positive))
        & (pd.to_numeric(slices.get("negative_count", 0), errors="coerce") >= int(min_negative))
    ].copy()
    if not selected.empty:
        selected = selected.sort_values(["slice_id", "slice_label"], kind="mergesort")
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    seeds = np.random.SeedSequence(int(seed)).spawn(len(selected))
    selected_records = list(selected.iterrows())

    def run_one(
        item: tuple[int, tuple[Any, pd.Series]],
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        index, (_, slice_row) = item
        slice_id = str(slice_row["slice_id"])
        slice_label = str(slice_row["slice_label"])
        definition = definition_by_id.get(slice_id)
        if definition is None:
            raise ValueError(f"Material slice definition is missing: {slice_id}")
        membership = membership_for_definition(
            definition, frame, scores=pd.Series(p, index=frame.index)
        ).astype(str)
        mask = membership.to_numpy() == slice_label
        child_seed = int(seeds[index].generate_state(1, dtype=np.uint64)[0])
        summary, replicate_rows = stratified_bootstrap(
            y[mask],
            p[mask],
            threshold,
            replicates=int(replicates),
            seed=child_seed,
            # Parallelism is applied across slices here; nested worker pools
            # would oversubscribe the bounded Phase 14 CPU budget.
            workers=1,
            confidence_level=confidence_level,
        )
        key = f"{slice_id}::{slice_label}"
        summary_payload = {
            "slice_id": slice_id,
            "slice_label": slice_label,
            "row_count": int(mask.sum()),
            "positive_count": int(y[mask].sum()),
            "negative_count": int((y[mask] == 0).sum()),
            "seed": child_seed,
            "summary": summary,
        }
        return key, summary_payload, replicate_rows

    if int(workers) > 1 and selected_records:
        with ThreadPoolExecutor(max_workers=min(int(workers), len(selected_records))) as executor:
            results = list(executor.map(run_one, enumerate(selected_records)))
    else:
        results = [run_one(item) for item in enumerate(selected_records)]
    for key, summary_payload, replicate_rows in results:
        summaries[key] = summary_payload
        rows.extend(
            {
                "slice_id": summary_payload["slice_id"],
                "slice_label": summary_payload["slice_label"],
                **row,
            }
            for row in replicate_rows
        )
    return {
        "replicate_count": int(replicates),
        "eligible_slice_count": int(len(selected)),
        "failed_replicate_count": int(
            sum(
                int(item["summary"].get("failed_replicate_count", 0)) for item in summaries.values()
            )
        ),
        "slices": summaries,
        "method": "STRATIFIED_PERCENTILE",
        "seed": int(seed),
    }, rows


__all__ = ["material_slice_bootstrap", "stratified_bootstrap"]
