"""Transparent sigmoid and isotonic calibrators."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from typing import Any, cast

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _validate(
    raw_probability: Any, target: Any, *, require_both_classes: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    p = cast(np.ndarray, np.asarray(raw_probability, dtype="float64").reshape(-1))
    y = cast(np.ndarray, np.asarray(target, dtype="int8").reshape(-1))
    if len(p) == 0 or len(p) != len(y):
        raise ValueError("Calibrator input and target must be non-empty and equally sized.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Calibrator probabilities must be finite and in [0, 1].")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Calibrator training targets must be binary.")
    if require_both_classes and np.unique(y).size < 2:
        raise ValueError("Calibrator training targets must contain both binary classes.")
    return p, y


def sigmoid_logit(probabilities: Any, epsilon: float = 1.0e-6) -> np.ndarray:
    p = np.asarray(probabilities, dtype="float64")
    if epsilon <= 0 or epsilon >= 0.5:
        raise ValueError("Sigmoid epsilon must be between zero and one half.")
    clipped = np.clip(p, epsilon, 1.0 - epsilon)
    return cast(np.ndarray, np.log(clipped / (1.0 - clipped)))


def isotonic_eligibility(
    target: Any,
    raw_probability: Any,
    *,
    minimum_positive: int = 20,
    minimum_negative: int = 100,
    minimum_unique: int = 50,
) -> tuple[bool, str]:
    p, y = _validate(raw_probability, target, require_both_classes=False)
    positive = int(np.count_nonzero(y == 1))
    negative = int(np.count_nonzero(y == 0))
    unique = int(np.unique(p).size)
    reasons: list[str] = []
    if positive < minimum_positive:
        reasons.append(f"positive_count={positive}<{minimum_positive}")
    if negative < minimum_negative:
        reasons.append(f"negative_count={negative}<{minimum_negative}")
    if unique < minimum_unique:
        reasons.append(f"unique_probability_count={unique}<{minimum_unique}")
    return (not reasons, "ELIGIBLE" if not reasons else ";".join(reasons))


def calibrator_sha(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "calibrator_sha"}
    return _canonical_sha(body)


def fit_calibrator(
    method: str,
    raw_probability: Any,
    target: Any,
    *,
    epsilon: float = 1.0e-6,
    isotonic_y_min: float = 0.0,
    isotonic_y_max: float = 1.0,
    isotonic_out_of_bounds: str = "clip",
    isotonic_minimum_positive: int = 20,
    isotonic_minimum_negative: int = 100,
    isotonic_minimum_unique: int = 50,
    input_sha: str | None = None,
) -> dict[str, Any]:
    method = str(method)
    p, y = _validate(raw_probability, target, require_both_classes=method != "C2_ISOTONIC")
    base: dict[str, Any] = {
        "method": method,
        "epsilon": float(epsilon),
        "fit_row_count": int(len(y)),
        "fit_positive_count": int(np.count_nonzero(y == 1)),
        "fit_negative_count": int(np.count_nonzero(y == 0)),
        "input_sha": input_sha,
        "sklearn_version": version("scikit-learn"),
        "eligible": True,
        "eligibility_reason": "ELIGIBLE",
    }
    if method == "C0_NONE":
        payload = {**base, "method": "NONE", "identity": True}
    elif method == "C1_SIGMOID":
        model = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=1000,
            class_weight=None,
        )
        model.fit(sigmoid_logit(p, epsilon).reshape(-1, 1), y)
        payload = {
            **base,
            "method": "SIGMOID",
            "identity": False,
            "coefficient": float(model.coef_[0][0]),
            "intercept": float(model.intercept_[0]),
            "solver": "lbfgs",
            "max_iter": 1000,
            "penalty": None,
            "class_weight": None,
        }
    elif method == "C2_ISOTONIC":
        eligible, reason = isotonic_eligibility(
            y,
            p,
            minimum_positive=isotonic_minimum_positive,
            minimum_negative=isotonic_minimum_negative,
            minimum_unique=isotonic_minimum_unique,
        )
        base["eligible"] = eligible
        base["eligibility_reason"] = reason
        if not eligible:
            payload = {**base, "method": "ISOTONIC", "identity": False, "breakpoints": None}
            payload["calibrator_sha"] = calibrator_sha(payload)
            return payload
        model = IsotonicRegression(
            y_min=isotonic_y_min,
            y_max=isotonic_y_max,
            out_of_bounds=isotonic_out_of_bounds,
        )
        model.fit(p, y)
        payload = {
            **base,
            "method": "ISOTONIC",
            "identity": False,
            "y_min": float(isotonic_y_min),
            "y_max": float(isotonic_y_max),
            "out_of_bounds": isotonic_out_of_bounds,
            "X_thresholds": [float(value) for value in model.X_thresholds_],
            "y_thresholds": [float(value) for value in model.y_thresholds_],
        }
    else:
        raise ValueError(f"Unsupported Phase 13 calibrator: {method}")
    payload["calibrator_sha"] = calibrator_sha(payload)
    return payload


def apply_calibrator(payload: dict[str, Any], raw_probability: Any) -> np.ndarray:
    p = np.asarray(raw_probability, dtype="float64").reshape(-1)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Calibrator probabilities must be finite and in [0, 1].")
    method = str(payload.get("method"))
    if method == "NONE":
        result = p
    elif method == "SIGMOID":
        z = sigmoid_logit(p, float(payload.get("epsilon", 1.0e-6)))
        score = float(payload["coefficient"]) * z + float(payload["intercept"])
        result = 1.0 / (1.0 + np.exp(-score))
    elif method == "ISOTONIC":
        x = np.asarray(payload.get("X_thresholds", []), dtype="float64")
        y = np.asarray(payload.get("y_thresholds", []), dtype="float64")
        if len(x) == 0 or len(x) != len(y):
            raise ValueError("Isotonic breakpoint payload is invalid.")
        result = np.interp(p, x, y, left=y[0], right=y[-1])
    else:
        raise ValueError(f"Unsupported serialized calibrator: {method}")
    result = np.asarray(result, dtype="float64")
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("Calibrated probabilities are not finite and bounded.")
    return result


__all__ = [
    "apply_calibrator",
    "calibrator_sha",
    "fit_calibrator",
    "isotonic_eligibility",
    "sigmoid_logit",
]
