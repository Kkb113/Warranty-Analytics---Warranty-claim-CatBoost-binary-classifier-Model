"""Exact Optuna search-space definitions and parameter validation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .config import SEARCH_PARAMETER_NAMES
from .models import OptimizationError


def validate_trial_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Validate one fully materialized parameter set against the allowlist."""

    if set(parameters) != set(SEARCH_PARAMETER_NAMES):
        raise OptimizationError("Trial parameters contain unknown or missing search parameters.")
    iterations = parameters["iterations"]
    if iterations not in (300, 500, 700, 1000, 1400):
        raise OptimizationError("iterations is outside the Phase 10 categorical space.")
    if not 0.015 <= float(parameters["learning_rate"]) <= 0.15:
        raise OptimizationError("learning_rate is outside the Phase 10 range.")
    if not isinstance(parameters["depth"], int) or not 4 <= parameters["depth"] <= 9:
        raise OptimizationError("depth is outside the Phase 10 integer range.")
    if not 1.0 <= float(parameters["l2_leaf_reg"]) <= 30.0:
        raise OptimizationError("l2_leaf_reg is outside the Phase 10 range.")
    if not 0.0 <= float(parameters["random_strength"]) <= 3.0:
        raise OptimizationError("random_strength is outside the Phase 10 range.")
    if not 0.0 <= float(parameters["bagging_temperature"]) <= 5.0:
        raise OptimizationError("bagging_temperature is outside the Phase 10 range.")
    if parameters["border_count"] not in (64, 128, 254):
        raise OptimizationError("border_count is outside the Phase 10 categorical space.")
    if not 0.70 <= float(parameters["rsm"]) <= 1.00:
        raise OptimizationError("rsm is outside the Phase 10 range.")
    return {str(key): value for key, value in parameters.items()}


def suggest_trial_parameters(trial: Any) -> dict[str, Any]:
    """Ask Optuna for exactly the eight allowlisted parameters."""

    parameters = {
        "iterations": trial.suggest_categorical("iterations", [300, 500, 700, 1000, 1400]),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 9),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        "rsm": trial.suggest_float("rsm", 0.70, 1.00),
    }
    return validate_trial_parameters(parameters)


def parameter_sha256(parameters: dict[str, Any]) -> str:
    """Hash the sorted JSON representation of one exact parameter set."""

    canonical = validate_trial_parameters(parameters)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()
