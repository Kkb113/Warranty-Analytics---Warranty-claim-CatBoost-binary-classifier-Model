"""Target-free group exposure and overlap diagnostics for Phase 6."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..feature_mart.common import deterministic_sort
from .assignments import validate_assignment_frame
from .models import SplitError

GROUP_COLUMNS = [
    "warranty_claim_key",
    "group_type",
    "group_value_hash",
]
EXPOSURE_COLUMNS = [
    "warranty_claim_key",
    "split",
    "group_type",
    "group_value_hash",
    "first_seen_split",
    "seen_in_train",
    "seen_in_validation",
    "seen_in_development",
    "unseen_in_train",
    "unseen_in_development",
    "is_model_feature",
]
SPLIT_ORDER = {"TRAIN": 0, "VALIDATION": 1, "TEST": 2}


def _group_key(group_type: Any, group_value_hash: Any) -> tuple[str, str]:
    return (str(group_type), str(group_value_hash))


def normalize_group_membership(
    assignments: pd.DataFrame,
    group_membership: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize Phase 5 lineage to one claim × one group membership."""

    validate_assignment_frame(assignments)
    missing = sorted(set(GROUP_COLUMNS) - set(group_membership.columns))
    if missing:
        raise SplitError(
            f"Phase 5 group membership is missing required columns: {', '.join(missing)}"
        )
    if group_membership["warranty_claim_key"].isna().any():
        raise SplitError("Phase 5 group membership contains null claim keys.")
    if (
        group_membership["group_type"].isna().any()
        or group_membership["group_value_hash"].isna().any()
    ):
        raise SplitError("Phase 5 group membership contains null group identity values.")
    if (
        "is_model_feature" in group_membership
        and group_membership["is_model_feature"].eq(True).any()
    ):
        raise SplitError("Phase 5 group membership contains model-feature metadata set to true.")
    claim_splits = assignments[["warranty_claim_key", "split"]].copy()
    if not bool(
        group_membership["warranty_claim_key"].isin(claim_splits["warranty_claim_key"]).all()
    ):
        raise SplitError("Group membership references a claim outside the Phase 5 split input.")
    normalized = group_membership[GROUP_COLUMNS].copy()
    normalized["group_type"] = normalized["group_type"].astype(str)
    normalized["group_value_hash"] = normalized["group_value_hash"].astype(str)
    normalized = normalized.drop_duplicates(GROUP_COLUMNS, keep="first")
    normalized = normalized.merge(
        claim_splits, on="warranty_claim_key", how="left", validate="many_to_one"
    )
    if normalized["split"].isna().any():
        raise SplitError("Group membership could not be assigned to a split.")
    return deterministic_sort(normalized, ["warranty_claim_key", "group_type", "group_value_hash"])


def build_group_exposure(
    assignments: pd.DataFrame,
    group_membership: pd.DataFrame,
) -> pd.DataFrame:
    """Build target-free seen/unseen exposure flags for every group membership."""

    normalized = normalize_group_membership(assignments, group_membership)
    if normalized.empty:
        return pd.DataFrame(columns=EXPOSURE_COLUMNS)
    group_splits: dict[tuple[str, str], set[str]] = {}
    for group_type, group_hash, split in normalized[
        ["group_type", "group_value_hash", "split"]
    ].itertuples(index=False, name=None):
        group_splits.setdefault(_group_key(group_type, group_hash), set()).add(str(split))
    rows: list[dict[str, Any]] = []
    for claim_key, split, group_type, group_hash in normalized[
        ["warranty_claim_key", "split", "group_type", "group_value_hash"]
    ].itertuples(index=False, name=None):
        key = _group_key(group_type, group_hash)
        seen_splits = group_splits[key]
        seen_train = "TRAIN" in seen_splits
        seen_validation = "VALIDATION" in seen_splits
        seen_development = seen_train or seen_validation
        first_seen = min(seen_splits, key=lambda item: SPLIT_ORDER[item])
        rows.append(
            {
                "warranty_claim_key": claim_key,
                "split": str(split),
                "group_type": str(group_type),
                "group_value_hash": str(group_hash),
                "first_seen_split": first_seen,
                "seen_in_train": seen_train,
                "seen_in_validation": seen_validation,
                "seen_in_development": seen_development,
                "unseen_in_train": not seen_train,
                "unseen_in_development": not seen_development,
                "is_model_feature": False,
            }
        )
    exposure = pd.DataFrame(rows, columns=EXPOSURE_COLUMNS)
    return deterministic_sort(exposure, ["warranty_claim_key", "group_type", "group_value_hash"])


def group_sets(exposure: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    """Return distinct hashed group sets by group type and split."""

    if exposure.empty:
        return {}
    missing = sorted(set(EXPOSURE_COLUMNS) - set(exposure.columns))
    if missing:
        raise SplitError(f"Group exposure is missing required columns: {', '.join(missing)}")
    result: dict[str, dict[str, set[str]]] = {}
    for group_type, group_hash, split in exposure[
        ["group_type", "group_value_hash", "split"]
    ].itertuples(index=False, name=None):
        result.setdefault(str(group_type), {}).setdefault(str(split), set()).add(str(group_hash))
    return result


def _claim_ids_for_group_state(
    exposure: pd.DataFrame,
    *,
    group_type: str,
    split: str,
    seen_column: str,
) -> set[Any]:
    subset = exposure.loc[
        (exposure["group_type"] == group_type)
        & (exposure["split"] == split)
        & exposure[seen_column].eq(True)
    ]
    return set(subset["warranty_claim_key"].tolist())


def summarize_group_overlap(exposure: pd.DataFrame) -> dict[str, Any]:
    """Create aggregate group-overlap summaries without target rates or raw IDs."""

    if exposure.empty:
        return {"group_types": {}, "available_group_types": []}
    sets = group_sets(exposure)
    summaries: dict[str, Any] = {}
    for group_type in sorted(sets):
        by_split = sets[group_type]
        train = by_split.get("TRAIN", set())
        validation = by_split.get("VALIDATION", set())
        test = by_split.get("TEST", set())
        development = train | validation
        validation_seen = validation & train
        test_seen_train = test & train
        test_seen_development = test & development
        validation_seen_claims = _claim_ids_for_group_state(
            exposure,
            group_type=group_type,
            split="VALIDATION",
            seen_column="seen_in_train",
        )
        validation_unseen_claims = set(
            exposure.loc[
                (exposure["group_type"] == group_type)
                & (exposure["split"] == "VALIDATION")
                & exposure["unseen_in_train"].eq(True),
                "warranty_claim_key",
            ].tolist()
        )
        test_seen_claims = _claim_ids_for_group_state(
            exposure,
            group_type=group_type,
            split="TEST",
            seen_column="seen_in_development",
        )
        test_unseen_claims = set(
            exposure.loc[
                (exposure["group_type"] == group_type)
                & (exposure["split"] == "TEST")
                & exposure["unseen_in_development"].eq(True),
                "warranty_claim_key",
            ].tolist()
        )
        summaries[group_type] = {
            "train_unique_groups": len(train),
            "validation_unique_groups": len(validation),
            "test_unique_groups": len(test),
            "validation_groups_seen_in_train": len(validation_seen),
            "validation_groups_unseen_in_train": len(validation - train),
            "test_groups_seen_in_train": len(test_seen_train),
            "test_groups_seen_in_development": len(test_seen_development),
            "test_groups_unseen_in_development": len(test - development),
            "claims_in_seen_groups": len(validation_seen_claims | test_seen_claims),
            "claims_in_unseen_groups": len(validation_unseen_claims | test_unseen_claims),
            "validation_claims_in_seen_groups": len(validation_seen_claims),
            "validation_claims_in_unseen_groups": len(validation_unseen_claims),
            "test_claims_in_seen_groups": len(test_seen_claims),
            "test_claims_in_unseen_groups": len(test_unseen_claims),
        }
    return {
        "group_types": summaries,
        "available_group_types": sorted(summaries),
    }


def available_group_type_diagnostics(group_membership: pd.DataFrame) -> dict[str, Any]:
    """Report expected group availability without treating absent optional groups as errors."""

    expected = (
        "truck",
        "truck_model",
        "manufacturing_plant",
        "assembly_line",
        "production_batch",
        "service_center",
        "historical_supplier",
        "historical_component_lot",
        "historical_component_batch",
        "safe_scenario_fingerprint",
    )
    counts = (
        group_membership.groupby("group_type", dropna=False).size().astype(int).to_dict()
        if not group_membership.empty and "group_type" in group_membership
        else {}
    )
    missing = [group_type for group_type in expected if int(counts.get(group_type, 0)) == 0]
    return {
        "expected_group_types": list(expected),
        "available_group_types": sorted(str(item) for item in counts if int(counts[item]) > 0),
        "row_counts": {str(key): int(value) for key, value in counts.items()},
        "missing_optional_group_types": missing,
    }
