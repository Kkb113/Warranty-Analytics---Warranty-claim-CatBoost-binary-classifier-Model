"""Bounded Phase 13 CPU planning."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .config import CalibrationEnsembleError, CalibrationEnsembleSettings


@dataclass(frozen=True, slots=True)
class ComputePlan:
    detected_physical_cores: int | None
    detected_logical_processors: int
    reserved_logical_processors: int
    effective_cpu_budget: int
    calibration_worker_count: int
    threads_per_calibration_worker: int
    catboost_replay_threads: int
    ensemble_evaluation_mode: str
    cli_overrides: dict[str, int | None]

    def as_dict(self) -> dict[str, object]:
        return {
            "detected_physical_cores": self.detected_physical_cores,
            "detected_logical_processors": self.detected_logical_processors,
            "reserved_logical_processors": self.reserved_logical_processors,
            "effective_cpu_budget": self.effective_cpu_budget,
            "calibration_worker_count": self.calibration_worker_count,
            "threads_per_calibration_worker": self.threads_per_calibration_worker,
            "catboost_replay_threads": self.catboost_replay_threads,
            "ensemble_evaluation_mode": self.ensemble_evaluation_mode,
            "cli_overrides": self.cli_overrides,
        }


def build_compute_plan(
    settings: CalibrationEnsembleSettings,
    *,
    max_workers: int | None = None,
    catboost_replay_threads: int | None = None,
    logical_processors: int | None = None,
    physical_cores: int | None = None,
) -> ComputePlan:
    detected = int(logical_processors or os.cpu_count() or 1)
    if detected < 1:
        raise CalibrationEnsembleError("At least one logical processor is required.")
    reserved = min(settings.reserve_logical_threads, max(0, detected - 1))
    budget = max(1, detected - reserved)
    requested_workers = int(max_workers or settings.preferred_calibration_workers)
    requested_replay = int(catboost_replay_threads or settings.preferred_catboost_replay_threads)
    if requested_workers < 1 or requested_replay < 1:
        raise CalibrationEnsembleError("Phase 13 CPU overrides must be positive integers.")
    workers = min(requested_workers, budget)
    replay = min(requested_replay, budget)
    return ComputePlan(
        detected_physical_cores=physical_cores,
        detected_logical_processors=detected,
        reserved_logical_processors=reserved,
        effective_cpu_budget=budget,
        calibration_worker_count=workers,
        threads_per_calibration_worker=1,
        catboost_replay_threads=replay,
        ensemble_evaluation_mode="VECTORIZED",
        cli_overrides={
            "max_workers": max_workers,
            "catboost_replay_threads": catboost_replay_threads,
        },
    )


__all__ = ["ComputePlan", "build_compute_plan"]
