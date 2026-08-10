"""Claim-level evaluation cohorts derived from group exposure only."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..feature_mart.common import deterministic_sort
from .assignments import validate_assignment_frame
from .group_exposure import normalize_group_membership

COHORT_COLUMNS = [
    "warranty_claim_key",
    "split",
    "eval__fingerprint_unseen",
    "eval__fingerprint_clean",
    "eval__truck_unseen",
    "eval__production_batch_unseen",
    "eval__service_center_unseen",
    "eval__historical_supplier_count",
    "eval__historical_supplier_seen_count",
    "eval__historical_supplier_unseen_count",
    "eval__any_historical_supplier_unseen",
    "eval__all_historical_suppliers_unseen",
    "eval__component_lot_count",
    "eval__component_lot_seen_count",
    "eval__component_lot_unseen_count",
    "eval__any_component_lot_unseen",
    "eval__all_component_lots_unseen",
    "eval__component_batch_count",
    "eval__component_batch_seen_count",
    "eval__component_batch_unseen_count",
    "eval__any_component_batch_unseen",
    "eval__all_component_batches_unseen",
    "is_model_feature",
]


def _reference_splits(split: str) -> set[str]:
    if split == "VALIDATION":
        return {"TRAIN"}
    if split == "TEST":
        return {"TRAIN", "VALIDATION"}
    return set()


def _claim_group_sets(group_membership: pd.DataFrame) -> dict[Any, dict[str, set[str]]]:
    result: dict[Any, dict[str, set[str]]] = {}
    for claim_key, group_type, group_hash in group_membership[
        ["warranty_claim_key", "group_type", "group_value_hash"]
    ].itertuples(index=False, name=None):
        result.setdefault(claim_key, {}).setdefault(str(group_type), set()).add(str(group_hash))
    return result


def _global_group_sets(
    group_membership: pd.DataFrame,
    reference_splits: set[str],
) -> dict[str, set[str]]:
    if not reference_splits:
        return {}
    grouped: dict[str, set[str]] = {}
    for group_type, group_hash, split in group_membership[
        ["group_type", "group_value_hash", "split"]
    ].itertuples(index=False, name=None):
        if str(split) in reference_splits:
            grouped.setdefault(str(group_type), set()).add(str(group_hash))
    return grouped


def _unseen_flag(values: set[str], reference_values: set[str]) -> bool:
    """Return whether an available direct group is unseen in the reference."""

    return bool(values) and not bool(values & reference_values)


def _historical_counts(
    values: set[str], reference_values: set[str]
) -> tuple[int, int, int, bool, bool]:
    seen = len(values & reference_values)
    unseen = len(values - reference_values)
    return len(values), seen, unseen, unseen > 0, bool(values) and unseen == len(values)


def build_evaluation_cohorts(
    assignments: pd.DataFrame,
    group_membership: pd.DataFrame,
) -> pd.DataFrame:
    """Build one metadata row per claim using split-relative group exposure."""

    validate_assignment_frame(assignments)
    normalized = normalize_group_membership(assignments, group_membership)
    per_claim = _claim_group_sets(normalized)
    reference_sets = {
        split: _global_group_sets(normalized, _reference_splits(split))
        for split in ("TRAIN", "VALIDATION", "TEST")
    }
    rows: list[dict[str, Any]] = []
    for claim_key, split in assignments[["warranty_claim_key", "split"]].itertuples(
        index=False, name=None
    ):
        split_value = str(split)
        references = reference_sets[split_value]
        claim_groups = per_claim.get(claim_key, {})
        fingerprint = claim_groups.get("safe_scenario_fingerprint", set())
        truck = claim_groups.get("truck", set())
        batch = claim_groups.get("production_batch", set())
        service_center = claim_groups.get("service_center", set())
        fingerprint_unseen = _unseen_flag(
            fingerprint, references.get("safe_scenario_fingerprint", set())
        )
        rows.append(
            {
                "warranty_claim_key": claim_key,
                "split": split_value,
                "eval__fingerprint_unseen": fingerprint_unseen,
                "eval__fingerprint_clean": fingerprint_unseen or split_value == "TRAIN",
                "eval__truck_unseen": _unseen_flag(truck, references.get("truck", set())),
                "eval__production_batch_unseen": _unseen_flag(
                    batch, references.get("production_batch", set())
                ),
                "eval__service_center_unseen": _unseen_flag(
                    service_center, references.get("service_center", set())
                ),
                **_historical_payload(
                    "historical_supplier",
                    claim_groups,
                    references,
                    "historical_supplier",
                    "historical_suppliers",
                ),
                **_historical_payload(
                    "historical_component_lot",
                    claim_groups,
                    references,
                    "component_lot",
                    "component_lots",
                ),
                **_historical_payload(
                    "historical_component_batch",
                    claim_groups,
                    references,
                    "component_batch",
                    "component_batches",
                ),
                "is_model_feature": False,
            }
        )
    cohorts = pd.DataFrame(rows, columns=COHORT_COLUMNS)
    return deterministic_sort(cohorts, ["warranty_claim_key"])


def _historical_payload(
    group_type: str,
    claim_groups: dict[str, set[str]],
    references: dict[str, set[str]],
    field_name: str,
    plural_name: str,
) -> dict[str, Any]:
    values = claim_groups.get(group_type, set())
    counts = _historical_counts(values, references.get(group_type, set()))
    count, seen, unseen, any_unseen, all_unseen = counts
    return {
        f"eval__{field_name}_count": count,
        f"eval__{field_name}_seen_count": seen,
        f"eval__{field_name}_unseen_count": unseen,
        f"eval__any_{field_name}_unseen": any_unseen,
        f"eval__all_{plural_name}_unseen": all_unseen,
    }


def summarize_evaluation_cohorts(cohorts: pd.DataFrame) -> dict[str, Any]:
    """Return aggregate cohort counts for reports."""

    if cohorts.empty:
        return {"total_claims": 0, "by_split": {}}
    result: dict[str, Any] = {"total_claims": int(len(cohorts)), "by_split": {}}
    for split in ("TRAIN", "VALIDATION", "TEST"):
        subset = cohorts.loc[cohorts["split"] == split]
        result["by_split"][split] = {
            "claims": int(len(subset)),
            "fingerprint_unseen_claims": int(subset["eval__fingerprint_unseen"].sum()),
            "fingerprint_clean_claims": int(subset["eval__fingerprint_clean"].sum()),
            "unseen_truck_claims": int(subset["eval__truck_unseen"].sum()),
            "unseen_production_batch_claims": int(subset["eval__production_batch_unseen"].sum()),
            "unseen_service_center_claims": int(subset["eval__service_center_unseen"].sum()),
            "any_unseen_supplier_claims": int(subset["eval__any_historical_supplier_unseen"].sum()),
            "all_unseen_supplier_claims": int(
                subset["eval__all_historical_suppliers_unseen"].sum()
            ),
            "any_unseen_component_lot_claims": int(subset["eval__any_component_lot_unseen"].sum()),
            "all_unseen_component_lot_claims": int(subset["eval__all_component_lots_unseen"].sum()),
            "any_unseen_component_batch_claims": int(
                subset["eval__any_component_batch_unseen"].sum()
            ),
            "all_unseen_component_batch_claims": int(
                subset["eval__all_component_batches_unseen"].sum()
            ),
        }
    return result


def fingerprint_overlap_summary(
    assignments: pd.DataFrame,
    exposure: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate duplicate/scenario fingerprint overlap diagnostics."""

    validate_assignment_frame(assignments)
    fingerprints = exposure.loc[exposure["group_type"] == "safe_scenario_fingerprint"]
    train = set(fingerprints.loc[fingerprints["split"] == "TRAIN", "group_value_hash"])
    validation = set(fingerprints.loc[fingerprints["split"] == "VALIDATION", "group_value_hash"])
    test = set(fingerprints.loc[fingerprints["split"] == "TEST", "group_value_hash"])
    development = train | validation
    validation_overlap = validation & train
    test_train_overlap = test & train
    test_development_overlap = test & development
    validation_claims = set(
        fingerprints.loc[
            (fingerprints["split"] == "VALIDATION")
            & fingerprints["group_value_hash"].isin(validation_overlap),
            "warranty_claim_key",
        ]
    )
    test_train_claims = set(
        fingerprints.loc[
            (fingerprints["split"] == "TEST")
            & fingerprints["group_value_hash"].isin(test_train_overlap),
            "warranty_claim_key",
        ]
    )
    test_development_claims = set(
        fingerprints.loc[
            (fingerprints["split"] == "TEST")
            & fingerprints["group_value_hash"].isin(test_development_overlap),
            "warranty_claim_key",
        ]
    )
    validation_count = int((assignments["split"] == "VALIDATION").sum())
    test_count = int((assignments["split"] == "TEST").sum())
    return {
        "validation_fingerprints_seen_in_train": len(validation_overlap),
        "validation_affected_claims": len(validation_claims),
        "validation_affected_percentage": _percentage(len(validation_claims), validation_count),
        "test_fingerprints_seen_in_train": len(test_train_overlap),
        "test_affected_claims_seen_in_train": len(test_train_claims),
        "test_affected_percentage_seen_in_train": _percentage(len(test_train_claims), test_count),
        "test_fingerprints_seen_in_development": len(test_development_overlap),
        "test_affected_claims_seen_in_development": len(test_development_claims),
        "test_affected_percentage_seen_in_development": _percentage(
            len(test_development_claims), test_count
        ),
        "validation_fingerprint_clean_claims": validation_count - len(validation_claims),
        "test_fingerprint_clean_claims": test_count - len(test_development_claims),
        "overlap_severity": "WARNING",
    }


def _percentage(numerator: int, denominator: int) -> float:
    return round((numerator / denominator) * 100.0, 6) if denominator else 0.0
