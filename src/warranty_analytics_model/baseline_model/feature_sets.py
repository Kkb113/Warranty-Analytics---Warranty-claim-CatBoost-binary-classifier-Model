"""Resolve predetermined Phase 9 experiments exclusively from lineage."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import BaselineModelError, FeatureSetSpec

ALLOWED_PHASE8_SOURCE = "prior_failure__failure_description"
MODEL_TYPES = {"numeric", "categorical", "boolean", "text"}


def _feature_set_hash(experiment_id: str, names: tuple[str, ...]) -> str:
    payload = json.dumps(
        {"experiment_id": experiment_id, "feature_names": list(names)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_phase8_sources(
    lineage: dict[str, dict[str, Any]],
    phase8_contract: dict[str, Any],
) -> dict[str, Any]:
    """Independently inspect every Phase 8 value source and fitted-transform policy."""

    errors: list[str] = []
    model_candidates = 0
    source_entries = 0
    unauthorized_entries = 0
    target_dependent = 0
    fitted_transformations = 0
    for name, item in lineage.items():
        if item.get("is_model_feature") is not True:
            continue
        model_candidates += 1
        values = item.get("value_sources")
        if not isinstance(values, list) or not values:
            errors.append(f"Phase 8 feature {name} has no explicit value_sources list.")
            continue
        for source in values:
            source_entries += 1
            if source != ALLOWED_PHASE8_SOURCE:
                unauthorized_entries += 1
                errors.append(f"Phase 8 feature {name} uses unauthorized value source: {source}")
        if item.get("target_dependent") is not False:
            target_dependent += 1
            errors.append(f"Phase 8 feature {name} is target-dependent.")
        if item.get("fitted_transformation") is not None:
            fitted_transformations += 1
            errors.append(f"Phase 8 feature {name} contains a fitted transformation.")
    policy = phase8_contract.get("fitted_transform_policy", {})
    required_false = (
        "tfidf",
        "count_vectorizer",
        "embeddings",
        "llm",
        "vocabulary_learning",
        "model_training",
    )
    if not isinstance(policy, dict):
        errors.append("Phase 8 fitted_transform_policy is missing.")
    else:
        for key in required_false:
            if policy.get(key) is not False:
                errors.append(f"Phase 8 {key} must be false before model training.")
    return {
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "model_candidate_count": model_candidates,
        "value_source_entry_count": source_entries,
        "unauthorized_value_source_count": unauthorized_entries,
        "target_dependent_feature_count": target_dependent,
        "fitted_transformation_count": fitted_transformations,
        "vocabulary_learning": policy.get("vocabulary_learning")
        if isinstance(policy, dict)
        else None,
    }


def resolve_feature_sets(
    phase7_lineage: dict[str, dict[str, Any]],
    phase8_lineage: dict[str, dict[str, Any]],
) -> dict[str, FeatureSetSpec]:
    """Resolve E1–E4 membership and deterministic feature order from lineage."""

    phase7 = [
        (name, item)
        for name, item in phase7_lineage.items()
        if item.get("is_model_feature") is True
    ]
    phase8 = [
        (name, item)
        for name, item in phase8_lineage.items()
        if item.get("is_model_feature") is True
    ]
    for name, item in phase7 + phase8:
        feature_type = item.get("feature_type")
        if feature_type not in MODEL_TYPES:
            raise BaselineModelError(f"Unsupported model feature type for {name}: {feature_type}")
        if item.get("is_control") is True or item.get("target_dependent") is not False:
            raise BaselineModelError(f"Unsafe model candidate metadata: {name}")
    phase7_core = [(name, item) for name, item in phase7 if item.get("tier") == "CORE"]
    phase7_extended = [(name, item) for name, item in phase7 if item.get("tier") == "EXTENDED"]
    if len(phase7_core) + len(phase7_extended) != len(phase7):
        raise BaselineModelError("Phase 7 model features must be CORE or EXTENDED.")
    phase8_lexical = [
        (name, item) for name, item in phase8 if item.get("feature_type") in {"numeric", "boolean"}
    ]
    phase8_text = [(name, item) for name, item in phase8 if item.get("feature_type") == "text"]
    if len(phase8_lexical) + len(phase8_text) != len(phase8):
        raise BaselineModelError("Phase 8 candidates must be numeric, boolean, or text.")
    definitions = {
        "E1": phase7_core,
        "E2": phase7,
        "E3": phase7 + phase8_lexical,
        "E4": phase7 + phase8_lexical + phase8_text,
    }
    result: dict[str, FeatureSetSpec] = {}
    for experiment_id, items in definitions.items():
        names = tuple(name for name, _ in items)
        if len(names) != len(set(names)):
            raise BaselineModelError(f"Duplicate model feature in {experiment_id}.")
        by_type = {
            kind: tuple(name for name, item in items if item.get("feature_type") == kind)
            for kind in MODEL_TYPES
        }
        result[experiment_id] = FeatureSetSpec(
            experiment_id=experiment_id,
            feature_names=names,
            numeric_features=by_type["numeric"],
            categorical_features=by_type["categorical"],
            boolean_features=by_type["boolean"],
            text_features=by_type["text"],
            phase7_core_count=sum(item.get("tier") == "CORE" for _, item in items),
            phase7_extended_count=sum(item.get("tier") == "EXTENDED" for _, item in items),
            phase8_lexical_count=sum(
                name in {candidate for candidate, _ in phase8_lexical} for name, _ in items
            ),
            phase8_text_count=sum(
                name in {candidate for candidate, _ in phase8_text} for name, _ in items
            ),
            feature_set_sha256=_feature_set_hash(experiment_id, names),
        )
    if not (
        set(result["E1"].feature_names) < set(result["E2"].feature_names)
        and set(result["E2"].feature_names) < set(result["E3"].feature_names)
        and set(result["E3"].feature_names) < set(result["E4"].feature_names)
    ):
        raise BaselineModelError("Phase 9 experiment feature sets are not strictly nested.")
    return result


def feature_sets_payload(feature_sets: dict[str, FeatureSetSpec]) -> dict[str, Any]:
    return {
        experiment_id: feature_sets[experiment_id].as_dict()
        for experiment_id in sorted(feature_sets)
    }
