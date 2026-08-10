"""Phase 6 split-control lineage and target-free scenario fingerprints."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from .common import deterministic_sort
from .models import FeatureMartError, MartContract


def canonical_value(value: Any) -> str:
    """Render one value deterministically, including a canonical null token."""

    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return "<NULL>"
    if isinstance(value, pd.Timestamp):
        return str(value.isoformat())
    if isinstance(value, (datetime, date)):
        return str(value.isoformat())
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def sha256_values(values: Iterable[Any]) -> str:
    """Hash canonical values using a stable field separator."""

    payload = "\x1f".join(canonical_value(value) for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_safe_scenario_fingerprint(
    frame: pd.DataFrame,
    input_columns: list[str] | tuple[str, ...],
) -> pd.Series:
    """Hash only explicitly documented safe direct/control values."""

    missing = sorted(set(input_columns) - set(frame.columns))
    if missing:
        raise FeatureMartError(
            f"Safe scenario fingerprint inputs are missing: {', '.join(missing)}"
        )
    if not input_columns:
        raise FeatureMartError("Safe scenario fingerprint input columns cannot be empty.")
    return frame[list(input_columns)].apply(lambda row: sha256_values(row.tolist()), axis=1)


def build_group_membership(
    snapshot: pd.DataFrame,
    component_history: pd.DataFrame,
    mart_contract: MartContract,
) -> pd.DataFrame:
    """Create non-model group relationships for later duplicate-aware splitting."""

    if "warranty_claim_key" not in snapshot:
        raise FeatureMartError("Group lineage requires warranty_claim_key in the snapshot.")
    rows: list[dict[str, Any]] = []

    direct_groups = {
        "truck": "lineage__truck_key",
        "truck_model": "lineage__truck_model_key",
        "manufacturing_plant": "truck__manufacturing_plant",
        "assembly_line": "truck__assembly_line",
        "production_batch": "lineage__production_batch_id",
        "service_center": "lineage__service_center_key",
    }
    for group_type, column in direct_groups.items():
        if column not in snapshot:
            continue
        for claim_key, value in snapshot[["warranty_claim_key", column]].itertuples(index=False):
            if pd.isna(value):
                continue
            rendered = canonical_value(value)
            rows.append(
                {
                    "warranty_claim_key": claim_key,
                    "group_type": group_type,
                    "group_value": rendered,
                    "group_value_hash": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                    "source": "claim_snapshot",
                    "is_model_feature": False,
                }
            )

    historical_groups = {
        "historical_supplier": "supplier_key",
        "historical_component_lot": "component_lot_no",
        "historical_component_batch": "production_batch_id",
    }
    if not component_history.empty:
        for group_type, column in historical_groups.items():
            if column not in component_history:
                continue
            for claim_key, value in component_history[
                ["current_warranty_claim_key", column]
            ].itertuples(index=False):
                if pd.isna(value):
                    continue
                rendered = canonical_value(value)
                rows.append(
                    {
                        "warranty_claim_key": claim_key,
                        "group_type": group_type,
                        "group_value": rendered,
                        "group_value_hash": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                        "source": "component_installation_history",
                        "is_model_feature": False,
                    }
                )

    fingerprint_inputs = mart_contract.safety_rules.get("safe_scenario_fingerprint_input_columns")
    if not isinstance(fingerprint_inputs, list) or not all(
        isinstance(item, str) for item in fingerprint_inputs
    ):
        raise FeatureMartError("The mart contract must document fingerprint input columns.")
    if any("target" in item.casefold() for item in fingerprint_inputs):
        raise FeatureMartError("Target columns cannot enter the safe scenario fingerprint.")
    fingerprints = build_safe_scenario_fingerprint(snapshot, fingerprint_inputs)
    for claim_key, fingerprint in zip(snapshot["warranty_claim_key"], fingerprints, strict=True):
        rows.append(
            {
                "warranty_claim_key": claim_key,
                "group_type": "safe_scenario_fingerprint",
                "group_value": fingerprint,
                "group_value_hash": fingerprint,
                "source": "phase5_safe_scenario",
                "is_model_feature": False,
            }
        )

    output = pd.DataFrame(
        rows,
        columns=[
            "warranty_claim_key",
            "group_type",
            "group_value",
            "group_value_hash",
            "source",
            "is_model_feature",
        ],
    )
    return deterministic_sort(
        output,
        ["warranty_claim_key", "group_type", "group_value_hash"],
    )
