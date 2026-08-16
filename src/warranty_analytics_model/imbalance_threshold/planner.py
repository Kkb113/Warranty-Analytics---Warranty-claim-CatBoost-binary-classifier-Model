"""Bounded CPU planning for independent Phase 12 fits."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .config import ImbalanceThresholdError, ImbalanceThresholdSettings


@dataclass(frozen=True, slots=True)
class ComputePlan:
    detected_logical_processors: int
    reserved_logical_processors: int
    effective_cpu_budget: int
    worker_count: int
    threads_per_fit: int
    maximum_concurrent_threads: int
    single_fit_threads: int
    cli_overrides: dict[str, int | None]

    def as_dict(self) -> dict[str, int | dict[str, int | None]]:
        return {
            "detected_logical_processors": self.detected_logical_processors,
            "reserved_logical_processors": self.reserved_logical_processors,
            "effective_cpu_budget": self.effective_cpu_budget,
            "worker_count": self.worker_count,
            "threads_per_fit": self.threads_per_fit,
            "maximum_concurrent_threads": self.maximum_concurrent_threads,
            "single_fit_threads": self.single_fit_threads,
            "cli_overrides": self.cli_overrides,
        }


def build_compute_plan(
    settings: ImbalanceThresholdSettings,
    *,
    max_workers: int | None = None,
    threads_per_fit: int | None = None,
    single_fit_threads: int | None = None,
    logical_processors: int | None = None,
) -> ComputePlan:
    detected = int(logical_processors or os.cpu_count() or 1)
    if detected < 1:
        raise ImbalanceThresholdError("At least one logical processor is required.")
    reserved = min(settings.reserve_logical_threads, max(0, detected - 1))
    budget = max(1, detected - reserved)
    workers_requested = max_workers or settings.preferred_search_workers
    threads_requested = threads_per_fit or settings.preferred_threads_per_search_fit
    one_requested = single_fit_threads or settings.preferred_single_fit_threads
    if workers_requested < 1 or threads_requested < 1 or one_requested < 1:
        raise ImbalanceThresholdError("CPU overrides must be positive integers.")
    workers = min(int(workers_requested), budget)
    threads = min(int(threads_requested), max(1, budget // workers))
    while workers * threads > budget and workers > 1:
        workers -= 1
    if workers * threads > budget:
        threads = max(1, budget // workers)
    single = min(int(one_requested), budget)
    return ComputePlan(
        detected_logical_processors=detected,
        reserved_logical_processors=reserved,
        effective_cpu_budget=budget,
        worker_count=workers,
        threads_per_fit=threads,
        maximum_concurrent_threads=workers * threads,
        single_fit_threads=single,
        cli_overrides={
            "max_workers": max_workers,
            "threads_per_fit": threads_per_fit,
            "single_fit_threads": single_fit_threads,
        },
    )


__all__ = ["ComputePlan", "build_compute_plan"]
