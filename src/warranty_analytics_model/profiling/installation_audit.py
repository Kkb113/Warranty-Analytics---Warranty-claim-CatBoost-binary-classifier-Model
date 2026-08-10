"""As-of component-installation diagnostics for Phase 3."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

INSTALLATION_GROUP_COLUMNS = (
    "component_lot_no",
    "production_batch_id",
    "supplier_key",
)

_MATCHED_COLUMNS = {
    "installation_key": "installation_key",
    "component_lot_no": "installation_component_lot_no",
    "production_batch_id": "installation_production_batch_id",
    "supplier_key": "installation_supplier_key",
    "installed_date": "installation_installed_date",
}


def _empty_match_frame(claims: pd.DataFrame) -> pd.DataFrame:
    """Return one diagnostic row per claim with empty installation context."""

    output = claims.reset_index(drop=True).copy()
    output["installation_match_status"] = pd.Series(pd.NA, index=output.index, dtype="string")
    for column in _MATCHED_COLUMNS.values():
        output[column] = pd.Series(pd.NA, index=output.index, dtype="object")
    return output


def _diagnostic_date(claims: pd.DataFrame) -> pd.Series:
    """Use failure date when present and claim date as the row-level fallback."""

    claim_dates = (
        pd.to_datetime(claims["claim_date"], errors="coerce")
        if "claim_date" in claims
        else pd.Series(pd.NaT, index=claims.index)
    )
    failure_dates = (
        pd.to_datetime(claims["failure_date"], errors="coerce")
        if "failure_date" in claims
        else pd.Series(pd.NaT, index=claims.index)
    )
    if "failure_date" not in claims:
        return claim_dates
    return failure_dates.fillna(claim_dates)


def _same_group_values(frame: pd.DataFrame, columns: Iterable[str]) -> bool:
    """Return whether all audited grouping values are identical, including nulls."""

    available = [column for column in columns if column in frame]
    if not available:
        return True
    for column in available:
        values = frame[column].map(lambda value: "<missing>" if pd.isna(value) else str(value))
        if values.nunique(dropna=False) > 1:
            return False
    return True


def _select_deterministically(frame: pd.DataFrame) -> pd.Series:
    """Select a stable row after latest-date and grouping ambiguity are resolved."""

    sort_columns = [
        column
        for column in (
            "installation_key",
            "component_serial_no",
            "component_lot_no",
            "production_batch_id",
            "supplier_key",
        )
        if column in frame.columns
    ]
    if sort_columns:
        return frame.sort_values(sort_columns, kind="mergesort", na_position="first").iloc[0]
    return frame.iloc[0]


def match_component_installations_asof(
    claims: pd.DataFrame,
    installations: pd.DataFrame | None,
    *,
    grouping_columns: Iterable[str] = INSTALLATION_GROUP_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach at most one historical installation to each claim for diagnostics.

    The matching key is ``truck_key`` plus ``causal_component_key``. The
    diagnostic as-of date is ``failure_date`` when present and non-null, with
    ``claim_date`` as a row-level fallback. Only installations with
    ``installed_date <= diagnostic_as_of_date`` are eligible. The latest
    eligible installation is selected; same-date rows are marked ambiguous and
    are excluded from purity context unless their audited grouping values are
    identical.

    This function is diagnostic only. It does not approve
    ``causal_component_key`` or any installation attribute as a production
    prediction-time feature.
    """

    output = _empty_match_frame(claims)
    output["installation_match_status"] = pd.Series(
        "no_causal_component", index=output.index, dtype="string"
    )
    key_columns = {"truck_key", "causal_component_key"}
    has_keys = key_columns.issubset(claims.columns)
    causal_mask = (
        claims["truck_key"].notna() & claims["causal_component_key"].notna()
        if has_keys
        else pd.Series(False, index=output.index)
    )
    claims_with_causal_component = int(causal_mask.sum())
    diagnostics: dict[str, Any] = {
        "as_of_rule": "failure_date when available, otherwise claim_date",
        "claims_with_causal_component": claims_with_causal_component,
        "matched_as_of_installation": 0,
        "unmatched_as_of_installation": claims_with_causal_component,
        "ambiguous_as_of_installation": 0,
        "future_installations_excluded": 0,
        "claims_with_multiple_historical_installations": 0,
    }
    output.loc[causal_mask, "installation_match_status"] = "unmatched"

    required_installation_columns = {"truck_key", "component_key", "installed_date"}
    if installations is None or not required_installation_columns.issubset(installations.columns):
        return output, diagnostics
    if not claims_with_causal_component:
        return output, diagnostics

    claim_dates = _diagnostic_date(claims)
    claim_keys = claims.loc[causal_mask, ["truck_key", "causal_component_key"]].copy()
    claim_keys["_phase3_claim_row"] = list(output.index[causal_mask.to_numpy()])
    claim_keys["_diagnostic_as_of_date"] = claim_dates.loc[causal_mask]
    claim_keys = claim_keys.reset_index(drop=True)
    installation_columns = [
        column
        for column in (
            "installation_key",
            "truck_key",
            "component_key",
            "component_serial_no",
            "component_lot_no",
            "production_batch_id",
            "supplier_key",
            "installed_date",
        )
        if column in installations.columns
    ]
    installation_rows = installations[installation_columns].copy()
    installation_rows["installed_date"] = pd.to_datetime(
        installation_rows["installed_date"], errors="coerce"
    )
    candidates = claim_keys.merge(
        installation_rows,
        left_on=["truck_key", "causal_component_key"],
        right_on=["truck_key", "component_key"],
        how="left",
        sort=False,
        validate="many_to_many",
    )
    candidate_dates = candidates["installed_date"].notna()
    as_of_dates = candidates["_diagnostic_as_of_date"].notna()
    future_mask = (
        candidate_dates
        & as_of_dates
        & (candidates["installed_date"] > candidates["_diagnostic_as_of_date"])
    )
    diagnostics["future_installations_excluded"] = int(future_mask.sum())
    eligible = candidates[
        candidate_dates
        & as_of_dates
        & (candidates["installed_date"] <= candidates["_diagnostic_as_of_date"])
    ].copy()
    if eligible.empty:
        return output, diagnostics

    historical_counts = eligible.groupby("_phase3_claim_row", observed=True).size()
    diagnostics["claims_with_multiple_historical_installations"] = int(
        (historical_counts > 1).sum()
    )

    ambiguous_count = 0
    matched_count = 0
    unmatched_count = claims_with_causal_component
    for claim_row, claim_candidates in eligible.groupby("_phase3_claim_row", observed=True):
        latest_date = claim_candidates["installed_date"].max()
        latest = claim_candidates[claim_candidates["installed_date"] == latest_date]
        ambiguous = len(latest) > 1
        if ambiguous:
            ambiguous_count += 1
        safe_to_collapse = _same_group_values(latest, grouping_columns)
        if ambiguous and not safe_to_collapse:
            output.at[int(claim_row), "installation_match_status"] = "ambiguous"
            continue
        selected = _select_deterministically(latest)
        output.at[int(claim_row), "installation_match_status"] = "matched"
        matched_count += 1
        unmatched_count -= 1
        for source, target in _MATCHED_COLUMNS.items():
            if source in selected.index:
                output.at[int(claim_row), target] = selected[source]

    diagnostics["matched_as_of_installation"] = matched_count
    diagnostics["unmatched_as_of_installation"] = unmatched_count
    diagnostics["ambiguous_as_of_installation"] = ambiguous_count
    return output, diagnostics
