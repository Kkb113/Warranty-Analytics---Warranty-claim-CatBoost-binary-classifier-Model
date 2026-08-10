"""Deterministic provenance and policy helpers for the Phase 9 hardening pass."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pandas as pd

from .models import BaselineModelError

HARDENING_VERSION = "phase9_corrective_hardening_v1"
EXPERIMENT_IDS = ("E0", "E1", "E2", "E3", "E4")
RUNTIME_VERSION_KEYS = (
    "python_version",
    "python_implementation",
    "catboost_version",
    "scikit_learn_version",
    "pandas_version",
    "numpy_version",
    "pyarrow_version",
    "platform",
    "machine",
    "os",
)
MODEL_CORE_PARAMETERS: dict[str, Any] = {
    "loss_function": "Logloss",
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0,
    "random_seed": 20260810,
    "use_best_model": False,
}
DISABLED_WEIGHT_VALUES = (None, False, 0, 1, "", "none", "None", "false", "disabled")


def _distribution_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unavailable"


def runtime_provenance() -> dict[str, str]:
    """Return secret-free interpreter, library, and host provenance."""

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "catboost_version": _distribution_version("catboost"),
        "scikit_learn_version": _distribution_version("scikit-learn"),
        "pandas_version": _distribution_version("pandas"),
        "numpy_version": _distribution_version("numpy"),
        "pyarrow_version": _distribution_version("pyarrow"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "os": platform.system(),
    }


def prediction_content_sha256(frame: pd.DataFrame) -> str:
    """Hash canonical validation predictions independent of Parquet encoding."""

    columns = ["warranty_claim_key", "experiment_id", "probability"]
    if list(frame.columns) != columns:
        raise BaselineModelError("Prediction content hashing requires the exact Phase 9 schema.")
    if frame.duplicated(columns).any():
        raise BaselineModelError("Prediction content hashing requires unique experiment/key rows.")
    ordered = frame.sort_values(["experiment_id", "warranty_claim_key"], kind="mergesort")
    probabilities = pd.to_numeric(ordered["probability"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise BaselineModelError("Prediction content hashing requires finite [0, 1] probabilities.")
    records = [
        [int(key), str(experiment_id), format(float(probability), ".17g")]
        for key, experiment_id, probability in ordered.itertuples(index=False, name=None)
    ]
    payload = json.dumps(
        {"columns": columns, "rows": records},
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _disabled_weight(key: str, value: Any) -> bool:
    if value in DISABLED_WEIGHT_VALUES:
        return True
    if key == "scale_pos_weight":
        try:
            return float(value) == 1.0
        except (TypeError, ValueError):
            return False
    if key == "class_weights" and isinstance(value, (list, tuple)):
        return all(float(item) == 1.0 for item in value)
    return False


def model_policy_errors(
    effective_parameters: dict[str, Any], *, context: str = "persisted model"
) -> list[str]:
    """Validate the effective persisted CatBoost policy without tuning semantics."""

    errors: list[str] = []
    if not isinstance(effective_parameters, dict):
        return [f"{context} effective_parameters are missing or not an object."]
    for key, expected in MODEL_CORE_PARAMETERS.items():
        if key not in effective_parameters:
            errors.append(f"{context} is missing locked CatBoost parameter: {key}")
            continue
        actual = effective_parameters[key]
        if isinstance(expected, float):
            matches = False
            try:
                matches = bool(np.isclose(float(actual), expected, rtol=0.0, atol=1.0e-8))
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            errors.append(f"{context} CatBoost parameter differs: {key}")
    for key in ("class_weights", "auto_class_weights", "scale_pos_weight"):
        if key in effective_parameters and not _disabled_weight(key, effective_parameters[key]):
            errors.append(f"{context} has active class weighting: {key}")
    for key in ("eval_set", "early_stopping_rounds", "od_wait"):
        if key in effective_parameters and effective_parameters[key] not in DISABLED_WEIGHT_VALUES:
            errors.append(f"{context} uses prohibited early-stopping/eval policy: {key}")
    if "od_type" in effective_parameters and str(
        effective_parameters["od_type"]
    ).casefold() not in {
        "none",
        "",
    }:
        errors.append(f"{context} uses prohibited early-stopping/eval policy: od_type")
    return errors


def runtime_provenance_errors(payload: dict[str, Any], *, required: bool = True) -> list[str]:
    """Validate the required secret-free runtime fields in a manifest."""

    if not isinstance(payload, dict):
        return ["Runtime provenance is missing or not an object."] if required else []
    errors: list[str] = []
    missing = [key for key in RUNTIME_VERSION_KEYS if not str(payload.get(key, "")).strip()]
    if missing and required:
        errors.append("Runtime provenance is missing: " + ", ".join(missing))
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True).casefold()
    for forbidden in (
        "username",
        "userprofile",
        "homedrive",
        "homepath",
        "password",
        "secret",
        "token",
    ):
        if forbidden in serialized:
            errors.append("Runtime provenance contains a prohibited identity or secret value.")
            break
    return list(dict.fromkeys(errors))
