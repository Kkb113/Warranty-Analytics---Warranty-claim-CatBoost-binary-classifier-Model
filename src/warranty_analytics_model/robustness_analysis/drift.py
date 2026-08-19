"""Feature and score distribution drift diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance


def _psi(expected: np.ndarray, actual: np.ndarray, bins: np.ndarray | None = None) -> float:
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    if bins is None:
        bins = np.unique(np.quantile(np.concatenate([expected, actual]), np.linspace(0, 1, 11)))
    if len(bins) < 2:
        return 0.0
    edges = np.concatenate(([-np.inf], bins[1:-1], [np.inf]))
    left = np.histogram(expected, bins=edges)[0].astype("float64")
    right = np.histogram(actual, bins=edges)[0].astype("float64")
    left = np.clip(left / max(float(left.sum()), 1.0), 1.0e-6, None)
    right = np.clip(right / max(float(right.sum()), 1.0), 1.0e-6, None)
    return float(np.sum((right - left) * np.log(right / left)))


def _psi_from_probabilities(
    expected: np.ndarray, actual: np.ndarray, *, epsilon: float = 1.0e-6
) -> float:
    """Compute PSI from already aligned category proportions.

    Categorical values are not observations on a numeric axis.  Treating their
    counts as histogram samples makes the PSI depend on the numeric magnitude
    of the counts rather than on the category distribution.  Add the same
    small mass to every aligned category, renormalize, and apply the canonical
    PSI formula directly.
    """

    left = np.asarray(expected, dtype="float64").reshape(-1)
    right = np.asarray(actual, dtype="float64").reshape(-1)
    if len(left) == 0 or len(left) != len(right):
        return 0.0
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Categorical proportions must be finite.")
    left = np.clip(left, 0.0, None) + float(epsilon)
    right = np.clip(right, 0.0, None) + float(epsilon)
    left /= left.sum()
    right /= right.sum()
    return float(np.sum((right - left) * np.log(right / left)))


def _numeric_row(name: str, train: pd.Series, validation: pd.Series) -> dict[str, Any]:
    # Normalize booleans and numeric extension dtypes to float before passing
    # values to SciPy.  ``wasserstein_distance`` subtracts samples internally
    # and NumPy does not support subtraction on boolean arrays.
    left = pd.to_numeric(train, errors="coerce").astype("float64")
    right = pd.to_numeric(validation, errors="coerce").astype("float64")
    left_values = left[np.isfinite(left)]
    right_values = right[np.isfinite(right)]
    edges = (
        np.unique(np.quantile(left_values, np.linspace(0, 1, 11)))
        if len(left_values)
        else np.array([])
    )
    train_missing = float(left.isna().mean())
    validation_missing = float(right.isna().mean())
    psi = _psi(left_values.to_numpy(), right_values.to_numpy(), edges)
    return {
        "feature": name,
        "feature_type": "numeric",
        "train_missing_rate": train_missing,
        "validation_missing_rate": validation_missing,
        "missing_rate_delta": validation_missing - train_missing,
        "psi": psi,
        "psi_classification": "LOW_SHIFT"
        if psi < 0.10
        else ("MODERATE_SHIFT" if psi < 0.25 else "HIGH_SHIFT"),
        "train_median": float(left_values.median()) if len(left_values) else None,
        "validation_median": float(right_values.median()) if len(right_values) else None,
        "median_shift": float(right_values.median() - left_values.median())
        if len(left_values) and len(right_values)
        else None,
        "train_iqr": float(left_values.quantile(0.75) - left_values.quantile(0.25))
        if len(left_values)
        else None,
        "validation_iqr": float(right_values.quantile(0.75) - right_values.quantile(0.25))
        if len(right_values)
        else None,
        "wasserstein_distance": float(wasserstein_distance(left_values, right_values))
        if len(left_values) and len(right_values)
        else None,
        "unseen_validation_category_rate": None,
        "js_divergence": None,
    }


def _categorical_row(name: str, train: pd.Series, validation: pd.Series) -> dict[str, Any]:
    left = train.astype("string").fillna("__MISSING__")
    right = validation.astype("string").fillna("__MISSING__")
    categories = sorted(set(left.astype(str)) | set(right.astype(str)))
    left_counts = (
        left.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy(dtype="float64")
    )
    right_counts = (
        right.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy(dtype="float64")
    )
    left_probs = left_counts / max(left_counts.sum(), 1.0)
    right_probs = right_counts / max(right_counts.sum(), 1.0)
    unseen = (~right.astype(str).isin(set(left.astype(str)))).mean()
    psi = _psi_from_probabilities(left_probs, right_probs)
    return {
        "feature": name,
        "feature_type": "categorical",
        "train_missing_rate": float(left.eq("__MISSING__").mean()),
        "validation_missing_rate": float(right.eq("__MISSING__").mean()),
        "missing_rate_delta": float(right.eq("__MISSING__").mean() - left.eq("__MISSING__").mean()),
        "psi": psi,
        "psi_classification": (
            "LOW_SHIFT" if psi < 0.10 else ("MODERATE_SHIFT" if psi < 0.25 else "HIGH_SHIFT")
        ),
        "train_median": None,
        "validation_median": None,
        "median_shift": None,
        "train_iqr": None,
        "validation_iqr": None,
        "wasserstein_distance": None,
        "unseen_validation_category_rate": float(unseen),
        # SciPy returns Jensen-Shannon distance (the square root of the
        # divergence), so keep the mathematically correct name in the artifact.
        "js_distance": float(jensenshannon(left_probs, right_probs, base=2.0)),
    }


def feature_drift(
    train: pd.DataFrame, validation: pd.DataFrame, feature_names: list[str], categorical: set[str]
) -> pd.DataFrame:
    rows = []
    for name in feature_names:
        if name not in train.columns or name not in validation.columns:
            rows.append({"feature": name, "status": "UNAVAILABLE"})
        elif name in categorical:
            rows.append(_categorical_row(name, train[name], validation[name]))
        else:
            rows.append(_numeric_row(name, train[name], validation[name]))
    result = pd.DataFrame(rows)
    if not result.empty:
        if "status" not in result:
            result["status"] = "AVAILABLE"
        else:
            result["status"] = result["status"].fillna("AVAILABLE")
        result["missingness_classification"] = (
            result.get("missing_rate_delta", pd.Series(0.0, index=result.index))
            .abs()
            .map(
                lambda value: (
                    "HIGH_MISSINGNESS_SHIFT"
                    if value >= 0.10
                    else ("MATERIAL_MISSINGNESS_SHIFT" if value >= 0.05 else "LOW_SHIFT")
                )
            )
        )
    return result.sort_values("feature", kind="mergesort").reset_index(drop=True)


def score_drift(train_scores: Any, validation_scores: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    left = np.asarray(train_scores, dtype="float64")
    right = np.asarray(validation_scores, dtype="float64")
    if len(left) == 0 or len(right) == 0:
        raise ValueError("Score drift requires non-empty score arrays.")
    percentiles = {"p01": 1, "p05": 5, "p25": 25, "p50": 50, "p75": 75, "p95": 95, "p99": 99}
    summary = {
        f"train_{name}": float(np.percentile(left, percentile))
        for name, percentile in percentiles.items()
    }
    summary.update(
        {
            f"validation_{name}": float(np.percentile(right, percentile))
            for name, percentile in percentiles.items()
        }
    )
    summary.update(
        {
            "train_mean": float(left.mean()),
            "validation_mean": float(right.mean()),
            "train_median": float(np.median(left)),
            "validation_median": float(np.median(right)),
            "train_std": float(left.std()),
            "validation_std": float(right.std()),
        }
    )
    drift = {
        "score_psi": _psi(left, right),
        "ks_statistic": float(ks_2samp(left, right).statistic),
        "wasserstein_distance": float(wasserstein_distance(left, right)),
        "classification": "SCORE_DISTRIBUTION_SHIFT" if _psi(left, right) >= 0.10 else "LOW_SHIFT",
    }
    return summary, drift


__all__ = ["feature_drift", "score_drift"]
