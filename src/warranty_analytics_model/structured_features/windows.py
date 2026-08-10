"""As-of windows and deterministic aggregation helpers."""

from __future__ import annotations

import pandas as pd

from .models import StructuredFeatureError


def numeric(values: pd.Series) -> pd.Series:
    """Convert numeric-looking SQL values without filling nulls."""

    return pd.to_numeric(values, errors="coerce")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return NULL for invalid denominators and never produce infinity."""

    left = numeric(numerator).astype("Float64")
    right = numeric(denominator).astype("Float64")
    left, right = left.align(right, join="outer")
    valid = right.notna() & (right > 0)
    result = pd.Series(pd.NA, index=left.index, dtype="Float64")
    result.loc[valid] = left.loc[valid] / right.loc[valid]
    return result


def month_index(values: pd.Series) -> pd.Series:
    """Map dates to actual calendar-month positions."""

    parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.year * 12 + parsed.dt.month


def add_claim_dates(
    history: pd.DataFrame,
    claims: pd.DataFrame,
    event_column: str,
    *,
    telemetry: bool = False,
) -> pd.DataFrame:
    """Attach claim dates and enforce strict-before Phase 5 as-of semantics."""

    key = "current_warranty_claim_key"
    claim_key = "warranty_claim_key"
    if key not in history or event_column not in history:
        return history.iloc[0:0].copy()
    claim_dates = claims[[claim_key, "claim__claim_date"]].rename(
        columns={claim_key: key, "claim__claim_date": "_claim_date"}
    )
    merged = history.merge(claim_dates, on=key, how="inner", validate="many_to_one")
    event_dates = pd.to_datetime(merged[event_column], errors="coerce")
    if telemetry:
        event_dates = event_dates + pd.offsets.MonthEnd(0)
    claim_dates_series = pd.to_datetime(merged["_claim_date"], errors="coerce")
    if (
        (event_dates.notna()) & (claim_dates_series.notna()) & (event_dates >= claim_dates_series)
    ).any():
        raise StructuredFeatureError(
            f"Unsafe same-day/future history detected in {event_column}; Phase 7 is blocked."
        )
    merged["_event_date"] = event_dates
    merged["_claim_date"] = claim_dates_series
    return merged.loc[merged["_event_date"].notna()].copy()


def window_mask(frame: pd.DataFrame, window: str) -> pd.Series:
    """Return a safe lower-bound mask for a fixed or all-history window."""

    if window == "all":
        return pd.Series(True, index=frame.index)
    months = int(window.removesuffix("m"))
    lower = frame["_claim_date"] - pd.DateOffset(months=months)
    return frame["_event_date"] >= lower


def population_std(values: pd.Series, minimum: int = 2) -> float | None:
    """Return population standard deviation only with enough observations."""

    clean = numeric(values).dropna()
    if len(clean) < minimum:
        return None
    return float(clean.std(ddof=0))


def per_claim_slope(
    frame: pd.DataFrame,
    value_column: str,
    *,
    minimum: int = 3,
) -> pd.Series:
    """Calculate a per-claim least-squares slope on actual calendar months."""

    if frame.empty or value_column not in frame.columns:
        return pd.Series(dtype="Float64")
    work = frame[["current_warranty_claim_key", "_event_date", value_column]].copy()
    work["_x"] = month_index(work["_event_date"])
    work["_y"] = numeric(work[value_column])
    work = work.dropna(subset=["_x", "_y"])
    if work.empty:
        return pd.Series(dtype="Float64")
    work = (
        work.groupby(["current_warranty_claim_key", "_x"], as_index=False, sort=True)["_y"]
        .mean()
        .rename(columns={"_y": "_value"})
    )

    def slope(group: pd.DataFrame) -> float | None:
        if len(group) < minimum:
            return None
        x = group["_x"].astype(float)
        y = group["_value"].astype(float)
        denominator = float(((x - x.mean()) ** 2).sum())
        if denominator <= 0:
            return None
        return float(((x - x.mean()) * (y - y.mean())).sum() / denominator)

    return work.groupby("current_warranty_claim_key", sort=True).apply(slope, include_groups=False)


def latest_series(
    frame: pd.DataFrame,
    value_column: str,
    key_column: str,
) -> pd.Series:
    """Select latest value with stable source-key tie-breaking."""

    if frame.empty or value_column not in frame.columns:
        return pd.Series(dtype="object")
    sort_cols = ["current_warranty_claim_key", "_event_date"]
    if key_column in frame:
        sort_cols.append(key_column)
    ordered = frame.sort_values(sort_cols, kind="mergesort", na_position="last")
    return ordered.drop_duplicates("current_warranty_claim_key", keep="last").set_index(
        "current_warranty_claim_key"
    )[value_column]


def deterministic_mode(values: pd.Series) -> str | None:
    """Choose highest frequency, then lexical value order."""

    clean = values.dropna().astype(str)
    if clean.empty:
        return None
    counts = clean.value_counts()
    top = counts[counts == counts.max()].index.astype(str).tolist()
    return str(sorted(top)[0])


def expected_completed_months(
    in_service: pd.Series,
    claim_date: pd.Series,
    months: int | None,
) -> pd.Series:
    """Compute completed calendar months available before each claim."""

    service = pd.to_datetime(in_service, errors="coerce")
    claim = pd.to_datetime(claim_date, errors="coerce")
    start = service.dt.to_period("M")
    if months is not None:
        requested = (claim - pd.DateOffset(months=months)).dt.to_period("M")
        start = start.where(start.notna() & (start > requested), requested)
    end = (claim - pd.Timedelta(days=1)).dt.to_period("M")
    values: list[int | None] = []
    for lower, upper in zip(start, end, strict=True):
        if pd.isna(lower) or pd.isna(upper) or lower > upper:
            values.append(None)
        else:
            values.append(int(upper.ordinal - lower.ordinal + 1))
    return pd.Series(values, index=claim.index, dtype="Float64")
