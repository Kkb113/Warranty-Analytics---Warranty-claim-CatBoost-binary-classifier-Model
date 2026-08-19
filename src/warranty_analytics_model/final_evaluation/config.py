"""Fail-closed Phase 15 scientific configuration and bounded execution plan."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from ..paths import discover_repository_root

PHASE15_VERSION = "phase15_final_test_evaluation_v1"
PHASE15_SEED = 20260810
LOCKED_CONFIGURATION: dict[str, Any] = {
    "seed": PHASE15_SEED,
    "model": {
        "policy": "REUSE_FROZEN_PHASE14_CHAMPION",
        "retraining": False,
        "train_validation_refit": False,
    },
    "test": {
        "one_frozen_scoring_policy": True,
        "model_selection_prohibited": True,
        "threshold_tuning_prohibited": True,
        "calibration_tuning_prohibited": True,
        "ensemble_tuning_prohibited": True,
        "feature_selection_prohibited": True,
    },
    "bootstrap": {
        "replicates": 2000,
        "confidence_level": 0.95,
        "method": "stratified_percentile",
    },
    "top_k": [0.05, 0.10, 0.20, 0.30],
    "invariance": {
        "probability_tolerance": 1.0e-10,
        "batch_sizes": [17, 64, 256],
    },
    "generalization": {
        "moderate_ap_ratio": 0.75,
        "moderate_roc_drop": 0.10,
        "random_roc": 0.50,
    },
    "compute": {
        "reserve_logical_threads": 2,
        "preferred_bootstrap_workers": 8,
        "preferred_catboost_inference_threads": 16,
    },
    "checkpoint": True,
    "resume_supported": True,
}


class Phase15ConfigurationError(ValueError):
    """Raised when scientific or execution configuration is unsafe."""


@dataclass(frozen=True, slots=True)
class Phase15Settings:
    seed: int
    bootstrap_replicates: int
    confidence_level: float
    bootstrap_method: str
    top_k: tuple[float, ...]
    probability_tolerance: float
    batch_sizes: tuple[int, ...]
    moderate_ap_ratio: float
    moderate_roc_drop: float
    random_roc: float
    reserve_logical_threads: int
    preferred_bootstrap_workers: int
    preferred_catboost_inference_threads: int
    checkpoint: bool
    resume_supported: bool

    def as_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(json.dumps(LOCKED_CONFIGURATION, sort_keys=True)))


def configuration_sha256() -> str:
    return hashlib.sha256(
        json.dumps(LOCKED_CONFIGURATION, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_payload(root: Path) -> dict[str, Any]:
    path = root / "configs" / "final_test_evaluation.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Phase15ConfigurationError(f"Cannot read Phase 15 configuration: {path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("phase15_final_test_evaluation") != LOCKED_CONFIGURATION
    ):
        raise Phase15ConfigurationError("Phase 15 configuration drifted from the locked payload.")
    return dict(payload["phase15_final_test_evaluation"])


def load_final_test_settings(project_root: Path | None = None) -> Phase15Settings:
    root = discover_repository_root(project_root)
    payload = _read_payload(root)
    bootstrap = payload["bootstrap"]
    invariance = payload["invariance"]
    generalization = payload["generalization"]
    compute = payload["compute"]
    return Phase15Settings(
        seed=int(payload["seed"]),
        bootstrap_replicates=int(bootstrap["replicates"]),
        confidence_level=float(bootstrap["confidence_level"]),
        bootstrap_method=str(bootstrap["method"]),
        top_k=tuple(float(value) for value in payload["top_k"]),
        probability_tolerance=float(invariance["probability_tolerance"]),
        batch_sizes=tuple(int(value) for value in invariance["batch_sizes"]),
        moderate_ap_ratio=float(generalization["moderate_ap_ratio"]),
        moderate_roc_drop=float(generalization["moderate_roc_drop"]),
        random_roc=float(generalization["random_roc"]),
        reserve_logical_threads=int(compute["reserve_logical_threads"]),
        preferred_bootstrap_workers=int(compute["preferred_bootstrap_workers"]),
        preferred_catboost_inference_threads=int(compute["preferred_catboost_inference_threads"]),
        checkpoint=bool(payload["checkpoint"]),
        resume_supported=bool(payload["resume_supported"]),
    )


def compute_plan(
    settings: Phase15Settings,
    *,
    max_workers: int | None = None,
    bootstrap_replicates: int | None = None,
    catboost_inference_threads: int | None = None,
) -> dict[str, Any]:
    """Build bounded execution settings; overrides cannot change science."""

    logical = max(1, int(os.cpu_count() or 1))
    budget = max(1, logical - settings.reserve_logical_threads)
    workers = int(
        max_workers
        if max_workers is not None
        else min(settings.preferred_bootstrap_workers, budget)
    )
    if workers < 1 or workers > budget:
        raise Phase15ConfigurationError("max-workers exceeds the reserved CPU budget.")
    repeats = (
        settings.bootstrap_replicates if bootstrap_replicates is None else int(bootstrap_replicates)
    )
    if repeats < settings.bootstrap_replicates:
        raise Phase15ConfigurationError("bootstrap-replicates may only be overridden upward.")
    inference = int(
        catboost_inference_threads
        if catboost_inference_threads is not None
        else min(settings.preferred_catboost_inference_threads, budget)
    )
    if inference < 1 or inference > budget:
        raise Phase15ConfigurationError(
            "catboost-inference-threads exceeds the reserved CPU budget."
        )
    return {
        "physical_cpus": os.cpu_count(),
        "logical_cpus": logical,
        "reserved_logical_threads": settings.reserve_logical_threads,
        "effective_cpu_budget": budget,
        "bootstrap_workers": workers,
        "native_threads_per_worker": 1,
        "test_bootstrap_replicates": repeats,
        "catboost_inference_threads": inference,
        "seed": settings.seed,
        "cli_overrides": {
            "max_workers": max_workers,
            "bootstrap_replicates": bootstrap_replicates,
            "catboost_inference_threads": catboost_inference_threads,
        },
    }


__all__ = [
    "LOCKED_CONFIGURATION",
    "PHASE15_SEED",
    "PHASE15_VERSION",
    "Phase15ConfigurationError",
    "Phase15Settings",
    "compute_plan",
    "configuration_sha256",
    "load_final_test_settings",
]
