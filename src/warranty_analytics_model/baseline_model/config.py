"""Load and validate fixed Phase 9 baseline settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..paths import discover_repository_root
from .models import BaselineModelError, BaselineSettings

FORBIDDEN_CATBOOST_KEYS = {
    "class_weights",
    "auto_class_weights",
    "scale_pos_weight",
    "early_stopping_rounds",
    "od_type",
    "od_wait",
}


def load_baseline_settings(project_root: Path | None = None) -> BaselineSettings:
    """Load the non-secret fixed baseline configuration and fail closed."""

    root = discover_repository_root(project_root)
    path = root / "configs" / "baseline_model.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BaselineModelError(f"Could not read Phase 9 configuration: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("baseline_model"), dict):
        raise BaselineModelError("Phase 9 configuration must contain baseline_model mapping.")
    raw = payload["baseline_model"]
    catboost = raw.get("catboost")
    text_processing = raw.get("text_processing")
    if not isinstance(catboost, dict) or not isinstance(text_processing, dict):
        raise BaselineModelError("Phase 9 CatBoost and text processing settings are required.")
    forbidden = sorted(FORBIDDEN_CATBOOST_KEYS & set(catboost))
    if forbidden:
        raise BaselineModelError("Phase 9 forbids CatBoost options: " + ", ".join(forbidden))
    required = {
        "loss_function": "Logloss",
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_strength": 1.0,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.0,
        "task_type": "CPU",
        "thread_count": 1,
        "allow_writing_files": False,
        "verbose": False,
        "use_best_model": False,
    }
    for key, expected in required.items():
        if catboost.get(key) != expected:
            raise BaselineModelError(f"Phase 9 fixed CatBoost setting changed: {key}")
    seed = raw.get("random_seed")
    if not isinstance(seed, int):
        raise BaselineModelError("Phase 9 random_seed is required and must be an integer.")
    if float(raw.get("fixed_threshold", -1)) != 0.5:
        raise BaselineModelError("Phase 9 threshold must be fixed at 0.5.")
    if raw.get("model_family") != "catboost" or raw.get("primary_metric") != "average_precision":
        raise BaselineModelError("Phase 9 model family or primary metric changed.")
    effective = {str(key): value for key, value in catboost.items()}
    effective["random_seed"] = seed
    return BaselineSettings(
        model_family="catboost",
        random_seed=seed,
        catboost_parameters=effective,
        text_processing={str(key): value for key, value in text_processing.items()},
        categorical_missing_value=str(raw.get("categorical_missing_value", "__MISSING__")),
        text_missing_value=str(raw.get("text_missing_value", "__NO_HISTORY__")),
        fixed_threshold=0.5,
        primary_metric="average_precision",
        output_directory=str(raw.get("output_directory", "artifacts/baseline_models")),
        report_directory=str(raw.get("report_directory", "reports/phase9_baseline_models")),
        compression=str(raw.get("compression", "snappy")),
    )


def settings_payload(settings: BaselineSettings) -> dict[str, Any]:
    """Return stable configuration metadata for manifests."""

    return {
        "model_family": settings.model_family,
        "random_seed": settings.random_seed,
        "catboost": settings.catboost_parameters,
        "text_processing": settings.text_processing,
        "categorical_missing_value": settings.categorical_missing_value,
        "text_missing_value": settings.text_missing_value,
        "fixed_threshold": settings.fixed_threshold,
        "primary_metric": settings.primary_metric,
        "output_directory": settings.output_directory,
        "report_directory": settings.report_directory,
        "compression": settings.compression,
    }
