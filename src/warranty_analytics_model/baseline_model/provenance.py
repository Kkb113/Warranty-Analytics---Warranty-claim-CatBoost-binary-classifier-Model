"""Deterministic provenance and policy helpers for the Phase 9 hardening pass."""

from __future__ import annotations

import hashlib
import json
import platform
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

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
PHASE9_RUNTIME_EXTRAS = ("profiling", "mart", "modeling")
PHASE10_RUNTIME_EXTRAS = ("optimization",)
PHASE9_RUNTIME_DISTRIBUTIONS = {
    "numpy": "numpy_version",
    "pandas": "pandas_version",
    "pyarrow": "pyarrow_version",
    "catboost": "catboost_version",
    "scikit-learn": "scikit_learn_version",
}


def _distribution_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unavailable"


def runtime_provenance(*, include_optimization: bool = False) -> dict[str, str]:
    """Return secret-free interpreter, library, and host provenance."""

    payload = {
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
    if include_optimization:
        payload["optuna_version"] = _distribution_version("optuna")
    return payload


def _phase9_declared_requirements(
    project_root: Path, *, include_optimization: bool = False
) -> tuple[str, dict[str, str]]:
    pyproject_path = project_root / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        project = tomllib.load(handle).get("project", {})
    python_requirement = str(project.get("requires-python", "")).strip()
    optional = project.get("optional-dependencies", {})
    raw_requirements = list(project.get("dependencies", []))
    extras = PHASE9_RUNTIME_EXTRAS + (PHASE10_RUNTIME_EXTRAS if include_optimization else ())
    distributions = dict(PHASE9_RUNTIME_DISTRIBUTIONS)
    if include_optimization:
        distributions["optuna"] = "optuna_version"
    for extra in extras:
        raw_requirements.extend(optional.get(extra, []))

    specifier_parts: dict[str, list[str]] = {}
    for raw in raw_requirements:
        requirement = Requirement(str(raw))
        name = canonicalize_name(requirement.name)
        if name not in distributions:
            continue
        parts = [part.strip() for part in str(requirement.specifier).split(",") if part.strip()]
        bucket = specifier_parts.setdefault(name, [])
        bucket.extend(part for part in parts if part not in bucket)
    return python_requirement, {
        name: ",".join(specifier_parts.get(name, [])) for name in distributions
    }


def validate_runtime_dependency_constraints(
    project_root: Path,
    runtime: dict[str, Any] | None = None,
    *,
    include_optimization: bool = False,
) -> dict[str, Any]:
    """Validate Phase 9 runtime versions against authoritative project metadata."""

    observed = (
        runtime_provenance(include_optimization=include_optimization)
        if runtime is None
        else runtime
    )
    errors: list[str] = []
    checked: dict[str, dict[str, Any]] = {}
    try:
        python_requirement, package_requirements = _phase9_declared_requirements(
            project_root, include_optimization=include_optimization
        )
    except (OSError, tomllib.TOMLDecodeError, InvalidRequirement) as exc:
        return {
            "status": "BLOCKED",
            "valid": False,
            "errors": [f"Phase 9 dependency requirements could not be loaded: {exc}"],
            "checked_requirements": {},
        }

    requirements = {"python": python_requirement, **package_requirements}
    version_keys = {"python": "python_version", **PHASE9_RUNTIME_DISTRIBUTIONS}
    if include_optimization:
        version_keys["optuna"] = "optuna_version"
    for name, requirement_text in requirements.items():
        actual = str(observed.get(version_keys[name], "")).strip()
        compatible = False
        if not requirement_text:
            errors.append(f"Phase 9 has no declared dependency constraint for {name}.")
        elif not actual or actual == "unavailable":
            errors.append(f"Phase 9 runtime dependency is unavailable: {name}.")
        else:
            try:
                compatible = SpecifierSet(requirement_text).contains(
                    Version(actual), prereleases=True
                )
            except (InvalidSpecifier, InvalidVersion):
                errors.append(f"Phase 9 runtime dependency version is invalid: {name}={actual!r}.")
            if not compatible and not any(
                error.startswith(f"Phase 9 runtime dependency version is invalid: {name}=")
                for error in errors
            ):
                errors.append(
                    f"Phase 9 runtime dependency {name} {actual} does not satisfy "
                    f"{requirement_text}."
                )
        checked[name] = {
            "declared_requirement": requirement_text,
            "resolved_version": actual,
            "compatible": compatible,
        }
    return {
        "status": "PASS" if not errors else "BLOCKED",
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "checked_requirements": checked,
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
