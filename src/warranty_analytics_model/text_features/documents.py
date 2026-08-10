"""Deterministic strict-as-of historical text document construction."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .models import Phase8Inputs, TextFeatureError, TextFeatureSettings
from .normalize import normalize_description

WINDOWS: tuple[tuple[str, int | None], ...] = (("6m", 6), ("12m", 12), ("24m", 24), ("all", None))


def _as_key(value: Any) -> Any:
    """Keep numeric claim keys stable while avoiding pandas scalar surprises."""

    return value.item() if hasattr(value, "item") else value


def _window_rows(
    rows: pd.DataFrame,
    claim_date: pd.Timestamp,
    months: int | None,
) -> pd.DataFrame:
    if months is None:
        return rows.loc[rows["prior_claim__claim_date"] < claim_date]
    lower = claim_date - pd.DateOffset(months=months)
    return rows.loc[
        (rows["prior_claim__claim_date"] >= lower) & (rows["prior_claim__claim_date"] < claim_date)
    ]


def build_historical_documents(
    inputs: Phase8Inputs,
    settings: TextFeatureSettings,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build four claim-level documents from the only approved text source."""

    prior = inputs.prior_claim_history.copy()
    assignments = inputs.assignments[["warranty_claim_key", "claim_date", "split"]].copy()
    assignments["claim_date"] = pd.to_datetime(assignments["claim_date"], errors="coerce")
    prior["prior_claim__claim_date"] = pd.to_datetime(
        prior["prior_claim__claim_date"], errors="coerce"
    )
    if assignments["claim_date"].isna().any():
        raise TextFeatureError("Phase 6 assignments contain invalid claim dates.")
    if prior["prior_claim__claim_date"].isna().any():
        raise TextFeatureError("Phase 5 prior history contains invalid prior claim dates.")
    if (
        prior["current_warranty_claim_key"].isna().any()
        or prior["prior_warranty_claim_key"].isna().any()
    ):
        raise TextFeatureError("Phase 5 prior history contains null claim keys.")

    claim_dates = dict(
        (_as_key(key), date)
        for key, date in assignments[["warranty_claim_key", "claim_date"]].itertuples(
            index=False, name=None
        )
    )
    prior_keys = set(claim_dates)
    source_keys = set(_as_key(value) for value in prior["current_warranty_claim_key"])
    unknown_current_keys = sorted(source_keys - prior_keys)
    if unknown_current_keys:
        raise TextFeatureError(
            "Prior history contains current claim keys outside Phase 6 membership."
        )

    joined_dates = prior["current_warranty_claim_key"].map(claim_dates)
    same_day_mask = prior["prior_claim__claim_date"] == joined_dates
    future_mask = prior["prior_claim__claim_date"] > joined_dates
    current_record_mask = prior["prior_warranty_claim_key"] == prior["current_warranty_claim_key"]
    same_day_count = int(same_day_mask.sum())
    future_count = int(future_mask.sum())
    current_record_count = int(current_record_mask.sum())
    if same_day_count or future_count or current_record_count:
        raise TextFeatureError(
            "Phase 8 strict-before history blocks unsafe prior records: "
            f"same_day={same_day_count}, future={future_count}, "
            f"current_records={current_record_count}."
        )

    prior["_normalized_description"] = prior["prior_failure__failure_description"].map(
        lambda value: normalize_description(value, settings)
    )
    grouped: dict[Any, pd.DataFrame] = {
        _as_key(key): group.sort_values(
            ["prior_claim__claim_date", "prior_warranty_claim_key"],
            kind="mergesort",
        )
        for key, group in prior.groupby("current_warranty_claim_key", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for claim_key, claim_date, split in assignments.sort_values(
        ["warranty_claim_key"], kind="mergesort"
    ).itertuples(index=False, name=None):
        key = _as_key(claim_key)
        claim_rows = grouped.get(key, prior.iloc[0:0].copy())
        base = {
            "warranty_claim_key": claim_key,
            "split": str(split),
            "claim__claim_date": pd.Timestamp(claim_date),
        }
        for window, months in WINDOWS:
            selected = _window_rows(claim_rows, pd.Timestamp(claim_date), months)
            descriptions = [
                str(value)
                for value in selected["_normalized_description"].tolist()
                if isinstance(value, str) and value
            ]
            base[f"prior_failure_text__{window}__document"] = (
                settings.document_separator.join(descriptions) if descriptions else None
            )
        rows.append(base)
    frame = pd.DataFrame(rows)
    expected = 4
    document_columns = [f"prior_failure_text__{window}__document" for window, _ in WINDOWS]
    if len(document_columns) != expected or len(frame) != len(assignments):
        raise TextFeatureError("Phase 8 document construction did not preserve claim grain.")
    return frame, {
        "historical_text_rows": int(len(prior)),
        "normalized_nonempty_descriptions": int(prior["_normalized_description"].notna().sum()),
        "same_day_text_records": same_day_count,
        "future_text_records": future_count,
        "current_record_text_records": current_record_count,
        "unknown_current_claim_keys": len(unknown_current_keys),
        "document_columns": document_columns,
    }
