"""Deterministic, lineage-backed feature family assignment."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from .config import FeatureSelectionError

_FAMILY_MAP = {
    "direct": "claim_safe_direct_context",
    "telemetry": "telemetry_history",
    "maintenance": "maintenance_history",
    "service": "service_history",
    "component": "component_installation_history",
    "prior_claim": "prior_claim_history",
    "warranty": "warranty_policy_context",
    "history_coverage": "history_coverage",
    "lifecycle": "lifecycle_context",
    "usage": "temporal_usage_aggregates",
}


def _phase8_family(item: dict[str, Any]) -> str:
    sources = item.get("value_sources")
    if not isinstance(sources, list) or not sources:
        raise FeatureSelectionError("Phase 8 model feature lacks value_sources lineage.")
    if any(str(value) != "prior_failure__failure_description" for value in sources):
        raise FeatureSelectionError("Phase 8 feature uses an unauthorized lineage source.")
    return "historical_lexical_features"


def build_feature_group_manifest(
    parent_features: dict[str, tuple[str, ...]],
    phase7_lineage: dict[str, dict[str, Any]],
    phase8_lineage: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Assign each parent feature exactly once using Phase 7/8 lineage."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    all_parent = {feature for names in parent_features.values() for feature in names}
    for _track, names in sorted(parent_features.items()):
        for feature in names:
            if feature in seen:
                # A feature appears in both tracks by design; it still has one
                # family assignment per track and one canonical lineage record.
                continue
            seen.add(feature)
            item = phase7_lineage.get(feature) or phase8_lineage.get(feature)
            if not isinstance(item, dict):
                raise FeatureSelectionError(
                    f"No lineage record exists for parent feature {feature}."
                )
            if item.get("is_model_feature") is not True:
                raise FeatureSelectionError(f"Non-model feature entered Phase 11: {feature}.")
            if item.get("is_control") is True or item.get("target_dependent") is not False:
                raise FeatureSelectionError(f"Unsafe/control feature entered Phase 11: {feature}.")
            if feature in phase8_lineage and feature not in phase7_lineage:
                family = _phase8_family(item)
                source_phase = "Phase8"
            else:
                raw_family = str(item.get("family", "")).strip()
                if not raw_family:
                    raise FeatureSelectionError(f"Phase 7 feature lacks family lineage: {feature}.")
                family = _FAMILY_MAP.get(raw_family, raw_family)
                source_phase = "Phase7"
            rows.append(
                {
                    "feature": feature,
                    "family": family,
                    "source_phase": source_phase,
                    "feature_type": item.get("feature_type"),
                    "tier": item.get("tier"),
                    "value_sources": json.dumps(item.get("value_sources", []), sort_keys=True),
                    "control_sources": json.dumps(item.get("control_sources", []), sort_keys=True),
                    "source_artifacts": json.dumps(
                        item.get("source_artifacts", []), sort_keys=True
                    ),
                    "source_columns": json.dumps(item.get("source_columns", []), sort_keys=True),
                    "window": item.get("window"),
                    "is_model_feature": True,
                    "is_control": False,
                    "target_dependent": False,
                }
            )
    if len(seen) != len(all_parent):
        raise FeatureSelectionError("Parent feature grouping did not cover every feature.")
    frame = pd.DataFrame(rows).sort_values("feature", kind="mergesort").reset_index(drop=True)
    if frame["feature"].duplicated().any() or len(frame) != len(all_parent):
        raise FeatureSelectionError("Feature family assignment is not one-to-one.")
    family_counts = frame.groupby("family", sort=True).size().to_dict()
    membership_sha256 = hashlib.sha256(
        frame[["feature", "family"]].to_json(orient="records", date_format="iso").encode("utf-8")
    ).hexdigest()
    manifest = {
        "phase": 11,
        "feature_count": int(len(frame)),
        "family_count": int(len(family_counts)),
        "families": {str(key): int(value) for key, value in sorted(family_counts.items())},
        "tracks": {
            track: {"feature_count": len(names), "features": list(names)}
            for track, names in sorted(parent_features.items())
        },
        "membership_sha256": membership_sha256,
    }
    return manifest, frame


def validate_group_membership(frame: pd.DataFrame, expected_features: set[str]) -> None:
    required = {"feature", "family", "is_model_feature", "is_control", "target_dependent"}
    if not required.issubset(frame.columns):
        raise FeatureSelectionError("Feature group membership schema is incomplete.")
    if (
        set(frame["feature"].astype(str)) != expected_features
        or frame["feature"].duplicated().any()
    ):
        raise FeatureSelectionError(
            "Feature group membership does not cover each parent feature once."
        )
    if frame["family"].isna().any() or (frame["family"].astype(str).str.len() == 0).any():
        raise FeatureSelectionError("Feature group membership contains an unassigned family.")
    if (
        (~frame["is_model_feature"]).any()
        or frame["is_control"].any()
        or frame["target_dependent"].any()
    ):
        raise FeatureSelectionError("Unsafe feature/control membership is present.")
