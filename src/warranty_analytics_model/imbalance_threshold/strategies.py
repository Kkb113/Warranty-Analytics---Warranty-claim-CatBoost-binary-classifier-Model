"""Deterministic Phase 12 weighting strategy definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..catboost_optimization.provenance import canonical_json_sha256
from .config import STRATEGY_IDS, STRATEGY_TYPES, STRATEGY_VALUES, ImbalanceThresholdError


@dataclass(frozen=True, slots=True)
class StrategyDefinition:
    strategy_id: str
    strategy_type: str
    parameter: float | str | None
    complexity_order: int
    parameter_sha256: str
    resolved_parameter: float | str | None = None

    @property
    def weighted(self) -> bool:
        return self.strategy_type != "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_type": self.strategy_type,
            "parameter": self.parameter,
            "complexity_order": self.complexity_order,
            "parameter_sha256": self.parameter_sha256,
            "resolved_parameter": self.resolved_parameter,
        }


def strategy_parameter_payload(
    strategy: StrategyDefinition,
    *,
    resolved_value: float | str | None = None,
) -> dict[str, Any]:
    """Return the complete, hashable weighting parameter payload."""

    parameter = strategy.parameter if resolved_value is None else resolved_value
    payload = {
        "strategy_id": strategy.strategy_id,
        "strategy_type": strategy.strategy_type,
        "parameter": parameter,
        "complexity_order": strategy.complexity_order,
    }
    if strategy.resolved_parameter is not None:
        payload["resolved_parameter"] = strategy.resolved_parameter
    return payload


def build_strategy_definitions(
    positive_count: int | None = None,
    negative_count: int | None = None,
) -> tuple[StrategyDefinition, ...]:
    """Build the exact eight strategies, resolving auto values when counts exist."""

    if (positive_count is None) != (negative_count is None):
        raise ImbalanceThresholdError("Both class counts are required to resolve auto weighting.")
    if positive_count is not None and (
        positive_count < 1 or negative_count is None or negative_count < 1
    ):
        raise ImbalanceThresholdError("Auto weighting requires positive TRAIN class counts.")
    if positive_count is not None:
        assert negative_count is not None
    positive = positive_count
    negative = negative_count
    result: list[StrategyDefinition] = []
    for index, (strategy_id, strategy_type, declared) in enumerate(
        zip(STRATEGY_IDS, STRATEGY_TYPES, STRATEGY_VALUES, strict=True)
    ):
        value: float | str | None = declared
        resolved: float | str | None = None
        if strategy_id == "S6_AUTO_SQRT_BALANCED" and positive is not None and negative is not None:
            resolved = float((negative / positive) ** 0.5)
        elif strategy_id == "S7_AUTO_BALANCED" and positive is not None and negative is not None:
            resolved = float(negative / positive)
        definition = StrategyDefinition(strategy_id, strategy_type, value, index, "", resolved)
        parameter_sha256 = canonical_json_sha256(strategy_parameter_payload(definition))
        result.append(
            StrategyDefinition(strategy_id, strategy_type, value, index, parameter_sha256, resolved)
        )
    return tuple(result)


def strategy_parameters(
    base_parameters: dict[str, Any], strategy: StrategyDefinition
) -> dict[str, Any]:
    """Apply exactly one weighting mechanism to a frozen CatBoost parameter map."""

    parameters = dict(base_parameters)
    for key in ("class_weights", "auto_class_weights", "scale_pos_weight"):
        parameters.pop(key, None)
    if strategy.strategy_type == "scale_pos_weight":
        if strategy.parameter is None:
            raise ImbalanceThresholdError("Scale-positive strategy has no value.")
        parameters["scale_pos_weight"] = float(strategy.parameter)
    elif strategy.strategy_type == "auto_class_weights":
        if not isinstance(strategy.parameter, str):
            raise ImbalanceThresholdError("Auto strategy parameter must be a CatBoost policy name.")
        parameters["auto_class_weights"] = strategy.parameter
    elif strategy.strategy_type != "none":
        raise ImbalanceThresholdError(f"Unsupported weighting strategy: {strategy.strategy_type}")
    return parameters


def validate_strategy_parameters(
    parameters: dict[str, Any], strategy: StrategyDefinition, parent_parameters: dict[str, Any]
) -> None:
    """Ensure statistical parent parameters differ only by the permitted weighting key."""

    forbidden = {"class_weights", "auto_class_weights", "scale_pos_weight"}
    left = {key: value for key, value in parameters.items() if key not in forbidden}
    right = {key: value for key, value in parent_parameters.items() if key not in forbidden}
    if left != right:
        raise ImbalanceThresholdError("Phase 12 strategy changed a frozen parent parameter.")
    present = [key for key in forbidden if key in parameters]
    if strategy.strategy_type == "none" and present:
        raise ImbalanceThresholdError("S0_NONE must not contain a weighting parameter.")
    if strategy.strategy_type == "scale_pos_weight" and present != ["scale_pos_weight"]:
        raise ImbalanceThresholdError("Scale-positive strategy has simultaneous weighting methods.")
    if strategy.strategy_type == "auto_class_weights" and present != ["auto_class_weights"]:
        raise ImbalanceThresholdError("Auto strategy has simultaneous weighting methods.")


__all__ = [
    "StrategyDefinition",
    "build_strategy_definitions",
    "strategy_parameter_payload",
    "strategy_parameters",
    "validate_strategy_parameters",
]
