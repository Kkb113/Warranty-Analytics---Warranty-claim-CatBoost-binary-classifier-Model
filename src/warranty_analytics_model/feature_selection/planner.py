"""Bounded CPU planning for independent Phase 11 experiments."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from .config import FeatureSelectionError, FeatureSelectionSettings


@dataclass(frozen=True, slots=True)
class ComputePlan:
    detected_logical_processors: int
    reserved_logical_processors: int
    effective_cpu_budget: int
    worker_count: int
    threads_per_worker: int
    single_fit_threads: int
    maximum_concurrent_threads: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def build_compute_plan(
    settings: FeatureSelectionSettings,
    *,
    logical_processors: int | None = None,
    max_workers: int | None = None,
    threads_per_fit: int | None = None,
    single_fit_threads: int | None = None,
) -> ComputePlan:
    detected = int(os.cpu_count() or 1 if logical_processors is None else logical_processors)
    if detected < 1:
        raise FeatureSelectionError("Detected logical processor count must be positive.")
    reserve = max(0, min(int(settings.reserve_logical_threads), detected - 1))
    budget = max(1, detected - reserve)
    requested_workers = int(settings.preferred_max_workers if max_workers is None else max_workers)
    requested_threads = int(
        settings.preferred_threads_per_worker if threads_per_fit is None else threads_per_fit
    )
    if requested_workers < 1 or requested_threads < 1:
        raise FeatureSelectionError("Phase 11 compute overrides must be positive.")
    workers = min(requested_workers, budget)
    threads = min(requested_threads, max(1, budget // workers))
    # A single fit may use more threads than a concurrent experiment, but never
    # more logical processors than the machine exposes.
    single = min(
        int(
            settings.preferred_single_fit_threads
            if single_fit_threads is None
            else single_fit_threads
        ),
        detected,
    )
    if single < 1:
        raise FeatureSelectionError("single_fit_threads must be positive.")
    return ComputePlan(detected, reserve, budget, workers, threads, single, workers * threads)

