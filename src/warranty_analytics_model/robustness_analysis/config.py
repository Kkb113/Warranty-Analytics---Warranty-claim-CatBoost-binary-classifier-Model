"""Fail-closed Phase 14 scientific and execution configuration."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from ..paths import discover_repository_root

PHASE14_VERSION = "phase14_robustness_error_analysis_v1"
PHASE14_SEED = 20260810
LOCKED_CONFIGURATION: dict[str, Any] = {
    "seed": PHASE14_SEED,
    "support": {
        "min_slice_rows": 75,
        "min_slice_positives_ranking": 5,
        "min_slice_negatives_ranking": 20,
        "min_slice_positives_bootstrap": 8,
    },
    "bootstrap": {
        "overall_replicates": 2000,
        "material_slice_replicates": 1000,
        "confidence_level": 0.95,
        "method": "stratified_percentile",
    },
    "temporal": {
        "calendar_month": True,
        "calendar_quarter": True,
        "chronological_thirds": True,
    },
    "categorical": {"top_train_categories": 10, "include_other": True, "include_missing": True},
    "numeric": {"train_quantiles": [0.0, 0.25, 0.50, 0.75, 1.0]},
    "score_deciles": {"boundaries_from": "TRAIN_OOF", "count": 10},
    "top_k": [0.05, 0.10, 0.20, 0.30],
    "threshold_sensitivity": {
        "diagnostic_only": True,
        "multipliers": [0.80, 0.90, 1.00, 1.10, 1.20],
    },
    "drift": {
        "psi_low": 0.10,
        "psi_high": 0.25,
        "missingness_material_delta": 0.05,
        "missingness_high_delta": 0.10,
    },
    "invariance": {"probability_absolute_tolerance": 1.0e-10, "batch_sizes": [17, 64, 256]},
    "signal_gate": {"minimum_ap_over_prevalence": 0.000001, "minimum_roc_auc": 0.50},
    "compute": {
        "reserve_logical_threads": 2,
        "preferred_bootstrap_workers": 8,
        "preferred_catboost_inference_threads": 16,
    },
    "checkpoint": True,
    "resume_supported": True,
}


class Phase14ConfigurationError(ValueError):
    """Raised when an operator attempts to change a scientific setting."""


@dataclass(frozen=True, slots=True)
class Phase14Settings:
    """Typed view of the locked YAML payload."""

    seed: int
    min_slice_rows: int
    min_slice_positives_ranking: int
    min_slice_negatives_ranking: int
    min_slice_positives_bootstrap: int
    overall_replicates: int
    material_slice_replicates: int
    confidence_level: float
    bootstrap_method: str
    top_train_categories: int
    top_k: tuple[float, ...]
    threshold_multipliers: tuple[float, ...]
    psi_low: float
    psi_high: float
    missingness_material_delta: float
    missingness_high_delta: float
    probability_tolerance: float
    batch_sizes: tuple[int, ...]
    minimum_ap_over_prevalence: float
    minimum_roc_auc: float
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
    path = root / "configs" / "robustness_error_analysis.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Phase14ConfigurationError(f"Cannot read Phase 14 configuration: {path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("phase14_robustness_error_analysis") != LOCKED_CONFIGURATION
    ):
        raise Phase14ConfigurationError("Phase 14 configuration drifted from the locked payload.")
    return dict(payload["phase14_robustness_error_analysis"])


def load_robustness_settings(project_root: Path | None = None) -> Phase14Settings:
    root = discover_repository_root(project_root)
    payload = _read_payload(root)
    support = payload["support"]
    bootstrap = payload["bootstrap"]
    categorical = payload["categorical"]
    threshold = payload["threshold_sensitivity"]
    drift = payload["drift"]
    invariance = payload["invariance"]
    signal = payload["signal_gate"]
    compute = payload["compute"]
    return Phase14Settings(
        seed=int(payload["seed"]),
        min_slice_rows=int(support["min_slice_rows"]),
        min_slice_positives_ranking=int(support["min_slice_positives_ranking"]),
        min_slice_negatives_ranking=int(support["min_slice_negatives_ranking"]),
        min_slice_positives_bootstrap=int(support["min_slice_positives_bootstrap"]),
        overall_replicates=int(bootstrap["overall_replicates"]),
        material_slice_replicates=int(bootstrap["material_slice_replicates"]),
        confidence_level=float(bootstrap["confidence_level"]),
        bootstrap_method=str(bootstrap["method"]),
        top_train_categories=int(categorical["top_train_categories"]),
        top_k=tuple(float(v) for v in payload["top_k"]),
        threshold_multipliers=tuple(float(v) for v in threshold["multipliers"]),
        psi_low=float(drift["psi_low"]),
        psi_high=float(drift["psi_high"]),
        missingness_material_delta=float(drift["missingness_material_delta"]),
        missingness_high_delta=float(drift["missingness_high_delta"]),
        probability_tolerance=float(invariance["probability_absolute_tolerance"]),
        batch_sizes=tuple(int(v) for v in invariance["batch_sizes"]),
        minimum_ap_over_prevalence=float(signal["minimum_ap_over_prevalence"]),
        minimum_roc_auc=float(signal["minimum_roc_auc"]),
        reserve_logical_threads=int(compute["reserve_logical_threads"]),
        preferred_bootstrap_workers=int(compute["preferred_bootstrap_workers"]),
        preferred_catboost_inference_threads=int(compute["preferred_catboost_inference_threads"]),
        checkpoint=bool(payload["checkpoint"]),
        resume_supported=bool(payload["resume_supported"]),
    )


def compute_plan(
    settings: Phase14Settings,
    *,
    max_workers: int | None = None,
    bootstrap_replicates: int | None = None,
    catboost_inference_threads: int | None = None,
) -> dict[str, Any]:
    """Build a bounded execution plan; overrides never change scientific defaults."""

    logical = max(1, int(os.cpu_count() or 1))
    workers = int(
        max_workers
        if max_workers is not None
        else min(
            settings.preferred_bootstrap_workers, max(1, logical - settings.reserve_logical_threads)
        )
    )
    if workers < 1 or workers > max(1, logical - settings.reserve_logical_threads):
        raise Phase14ConfigurationError("max-workers exceeds the reserved CPU budget.")
    repeats = (
        settings.overall_replicates if bootstrap_replicates is None else int(bootstrap_replicates)
    )
    if repeats < settings.overall_replicates:
        raise Phase14ConfigurationError("bootstrap-replicates may only be overridden upward.")
    inference = int(
        catboost_inference_threads
        if catboost_inference_threads is not None
        else min(
            settings.preferred_catboost_inference_threads,
            max(1, logical - settings.reserve_logical_threads),
        )
    )
    if inference < 1 or inference > max(1, logical - settings.reserve_logical_threads):
        raise Phase14ConfigurationError(
            "catboost-inference-threads exceeds the reserved CPU budget."
        )
    return {
        "logical_cpus": logical,
        "reserved_logical_threads": settings.reserve_logical_threads,
        "effective_cpu_budget": max(1, logical - settings.reserve_logical_threads),
        "bootstrap_workers": workers,
        "native_threads_per_worker": 1,
        "overall_bootstrap_replicates": repeats,
        "material_slice_bootstrap_replicates": settings.material_slice_replicates,
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
    "PHASE14_SEED",
    "PHASE14_VERSION",
    "Phase14ConfigurationError",
    "Phase14Settings",
    "compute_plan",
    "configuration_sha256",
    "load_robustness_settings",
]
