"""Deterministic feature-set construction, ranking, and selection rules."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..baseline_model.models import FeatureSetSpec
from .config import FeatureSelectionError, FeatureSelectionSettings


def feature_set_sha256(experiment_id: str, features: Iterable[str]) -> str:
    names = list(features)
    payload = json.dumps(
        {"experiment_id": experiment_id, "feature_names": names},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def feature_list_sha256(features: Iterable[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(features), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def subset_feature_set(
    parent: FeatureSetSpec, features: Iterable[str], candidate_id: str
) -> FeatureSetSpec:
    selected = tuple(features)
    if not selected or len(selected) != len(set(selected)):
        raise FeatureSelectionError("Phase 11 candidate feature list must be non-empty and unique.")
    if not set(selected).issubset(parent.feature_names):
        raise FeatureSelectionError(f"Candidate {candidate_id} adds a feature outside its parent.")
    order = {name: index for index, name in enumerate(parent.feature_names)}
    selected = tuple(sorted(selected, key=lambda name: order[name]))
    return FeatureSetSpec(
        experiment_id=candidate_id,
        feature_names=selected,
        numeric_features=tuple(name for name in parent.numeric_features if name in selected),
        categorical_features=tuple(
            name for name in parent.categorical_features if name in selected
        ),
        boolean_features=tuple(name for name in parent.boolean_features if name in selected),
        text_features=tuple(name for name in parent.text_features if name in selected),
        phase7_core_count=sum(
            name in selected for name in parent.feature_names[: parent.phase7_core_count]
        ),
        phase7_extended_count=sum(
            name in selected
            for name in parent.feature_names[
                parent.phase7_core_count : parent.phase7_core_count + parent.phase7_extended_count
            ]
        ),
        phase8_lexical_count=sum(
            name in selected
            for name in parent.numeric_features
            if name
            not in parent.feature_names[: parent.phase7_core_count + parent.phase7_extended_count]
        ),
        phase8_text_count=sum(name in selected for name in parent.text_features),
        feature_set_sha256=feature_set_sha256(candidate_id, selected),
    )


def stable_importance_ranking(stability: list[dict[str, Any]]) -> list[str]:
    return [
        str(row["feature"])
        for row in sorted(
            stability, key=lambda row: (-float(row["stable_score"]), str(row["feature"]))
        )
    ]


@dataclass(frozen=True, slots=True)
class CandidateDefinition:
    candidate_id: str
    track: str
    feature_count: int
    removed_feature_count: int
    reduction_fraction: float
    feature_set_sha256: str
    feature_list: tuple[str, ...]
    source_strategy: str
    family_inventory: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "track": self.track,
            "feature_count": self.feature_count,
            "removed_feature_count": self.removed_feature_count,
            "reduction_fraction": self.reduction_fraction,
            "feature_set_sha256": self.feature_set_sha256,
            "feature_list": list(self.feature_list),
            "source_strategy": self.source_strategy,
            "family_inventory": dict(sorted(self.family_inventory.items())),
        }


def _family_inventory(features: Iterable[str], family_by_feature: dict[str, str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for feature in features:
        family = family_by_feature[feature]
        result[family] = result.get(family, 0) + 1
    return result


def generate_candidates(
    track: str,
    parent: FeatureSetSpec,
    ranking: list[str],
    family_ablation: list[dict[str, Any]],
    stability: list[dict[str, Any]],
    family_by_feature: dict[str, str],
    settings: FeatureSelectionSettings,
) -> list[CandidateDefinition]:
    if len(ranking) != parent.feature_count or set(ranking) != set(parent.feature_names):
        raise FeatureSelectionError(f"{track} stable ranking is not an exact parent permutation.")
    definitions: list[tuple[str, str, list[str]]] = []
    parent_names = list(parent.feature_names)
    definitions.append(("FULL_PARENT", "full_parent", parent_names))
    for fraction in settings.candidate_fractions[1:]:
        count = max(settings.minimum_feature_count, int(round(parent.feature_count * fraction)))
        definitions.append(
            (f"TOP_{int(round(fraction * 100)):03d}", "stable_importance_fraction", ranking[:count])
        )
    # A family is prunable only when the leave-one-family-out experiment is at
    # least neutral (delta is parent minus ablation, so <= 0 means no loss).
    removable = [
        str(row["family"])
        for row in sorted(family_ablation, key=lambda row: str(row["family"]))
        if float(row.get("delta_ap_vs_parent", 0.0)) <= 0.0005
    ]
    family_pruned = [name for name in parent_names if family_by_feature[name] not in set(removable)]
    if len(family_pruned) < settings.minimum_feature_count:
        family_pruned = ranking[: settings.minimum_feature_count]
    definitions.append(("FAMILY_PRUNED", "neutral_family_ablation", family_pruned))
    stability_rows = {str(row["feature"]): row for row in stability}
    stable = [
        name
        for name in ranking
        if int(stability_rows[name].get("top_50_percent_fold_count", 0)) >= 2
        or int(stability_rows[name].get("top_25_percent_fold_count", 0)) >= 1
    ]
    if len(stable) < settings.minimum_feature_count:
        stable = ranking[: settings.minimum_feature_count]
    definitions.append(("STABILITY_PRUNED", "fold_stability", stable))
    seen: set[str] = set()
    candidates: list[CandidateDefinition] = []
    for suffix, strategy, names in definitions:
        unique_names = tuple(name for name in parent_names if name in set(names))
        if len(unique_names) < settings.minimum_feature_count:
            raise FeatureSelectionError(
                f"Candidate {track}_{suffix} violates minimum feature count."
            )
        digest = feature_set_sha256(f"P11_{track}_{suffix}", unique_names)
        if digest in seen:
            continue
        seen.add(digest)
        candidates.append(
            CandidateDefinition(
                candidate_id=f"P11_{track}_{suffix}",
                track=track,
                feature_count=len(unique_names),
                removed_feature_count=parent.feature_count - len(unique_names),
                reduction_fraction=(parent.feature_count - len(unique_names))
                / parent.feature_count,
                feature_set_sha256=digest,
                feature_list=unique_names,
                source_strategy=strategy,
                family_inventory=_family_inventory(unique_names, family_by_feature),
            )
        )
    if len(candidates) > settings.maximum_candidate_subsets_per_track:
        raise FeatureSelectionError(f"{track} generated more than the Phase 11 candidate cap.")
    return candidates


def select_candidate(
    candidates: list[dict[str, Any]], settings: FeatureSelectionSettings
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not candidates:
        raise FeatureSelectionError("Cannot select from an empty Phase 11 candidate set.")
    best = max(
        candidates, key=lambda row: (float(row["mean_average_precision"]), str(row["candidate_id"]))
    )
    best_ap = float(best["mean_average_precision"])
    best_min = float(best["min_average_precision"])
    best_std = float(best["std_average_precision"])
    tolerance = min(best_std / (3**0.5), settings.maximum_ap_tolerance)
    eligible: list[dict[str, Any]] = []
    for row in candidates:
        is_eligible = not settings.simpler_enabled or (
            float(row["mean_average_precision"]) >= best_ap - tolerance
            and float(row["min_average_precision"]) >= best_min - settings.maximum_min_ap_drop
            and float(row["mean_roc_auc"])
            >= float(best["mean_roc_auc"]) - settings.maximum_roc_auc_drop
            and float(row["mean_log_loss"])
            <= float(best["mean_log_loss"]) + settings.maximum_log_loss_increase
        )
        if is_eligible:
            eligible.append(row)
    selected = sorted(
        eligible,
        key=lambda row: (
            int(row["feature_count"]),
            -float(row["mean_average_precision"]),
            -float(row["min_average_precision"]),
            float(row["std_average_precision"]),
            -float(row["mean_roc_auc"]),
            float(row["mean_log_loss"]),
            str(row["candidate_id"]),
        ),
    )[0]
    trace = {
        "best_candidate_id": best["candidate_id"],
        "best_mean_average_precision": best_ap,
        "best_min_average_precision": best_min,
        "best_std_average_precision": best_std,
        "standard_error": best_std / (3**0.5),
        "effective_ap_tolerance": tolerance,
        "eligible_candidate_ids": sorted(str(row["candidate_id"]) for row in eligible),
        "selected_candidate_id": selected["candidate_id"],
        "rule": "fewest_features, higher AP, higher min AP, lower AP std, higher ROC, lower logloss, candidate ID",
    }
    return selected, trace


def replacement_decision(
    parent: dict[str, Any], selected: dict[str, Any], settings: FeatureSelectionSettings
) -> dict[str, Any]:
    ap_delta = float(selected["average_precision"]) - float(parent["average_precision"])
    reduction = (int(parent["feature_count"]) - int(selected["feature_count"])) / int(
        parent["feature_count"]
    )
    direct = ap_delta > settings.ap_improvement_tolerance
    complexity = (
        settings.complexity_allowed
        and float(selected["average_precision"])
        >= float(parent["average_precision"]) - settings.complexity_maximum_ap_drop
        and reduction >= settings.complexity_minimum_reduction_fraction
        and float(selected["roc_auc"])
        >= float(parent["roc_auc"]) - settings.complexity_maximum_roc_auc_drop
        and float(selected["log_loss"])
        <= float(parent["log_loss"]) + settings.complexity_maximum_log_loss_increase
    )
    replace = direct or complexity
    return {
        "replace_parent": replace,
        "reason": "AP_IMPROVEMENT"
        if direct
        else ("COMPLEXITY_TRADEOFF" if complexity else "FALLBACK_PARENT"),
        "average_precision_delta": ap_delta,
        "feature_reduction_fraction": reduction,
        "complexity_tradeoff_eligible": complexity,
        "ap_improvement_eligible": direct,
    }

