"""Column-level distribution and missingness diagnostics."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

_SENSITIVE_NAME_PARTS = (
    "vin",
    "customer_name",
    "technician",
    "inspector",
    "serial",
    "notes",
    "description",
    "complaint",
    "diagnostic_summary",
    "repair_notes",
)


def is_sensitive_column(column: str) -> bool:
    """Identify values that must never be copied into generated reports."""

    normalized = column.casefold()
    return any(part in normalized for part in _SENSITIVE_NAME_PARTS)


def normalize_text(value: object) -> str:
    """Normalize text for aggregate duplicate analysis only."""

    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def _json_value(value: object) -> object:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    return value


def _safe_category(value: object, column: str) -> object:
    if is_sensitive_column(column):
        normalized = normalize_text(value)
        if not normalized:
            return "<missing>"
        return f"<redacted:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:12]}>"
    return _json_value(value)


def _percentile_map(values: pd.Series, percentiles: Iterable[float]) -> dict[str, object]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {f"p{int(point * 100):02d}": None for point in percentiles}
    quantiles = numeric.quantile(list(percentiles))
    return {f"p{int(point * 100):02d}": _json_value(quantiles.loc[point]) for point in percentiles}


def profile_series(
    series: pd.Series,
    *,
    column_name: str | None = None,
    top_categories: int = 20,
    percentiles: Iterable[float] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
    rare_category_thresholds: Iterable[int] = (1, 5, 10, 20),
    reference_date: pd.Timestamp | None = None,
) -> dict[str, object]:
    """Profile one series without retaining raw sensitive values."""

    name = column_name or str(series.name or "column")
    row_count = int(len(series))
    null_count = int(series.isna().sum())
    non_null_count = row_count - null_count
    result: dict[str, object] = {
        "column": name,
        "row_count": row_count,
        "non_null_count": non_null_count,
        "null_count": null_count,
        "null_percentage": round(null_count / row_count * 100, 6) if row_count else 0.0,
    }

    date_values = pd.to_datetime(series, errors="coerce")
    is_date = pd.api.types.is_datetime64_any_dtype(series) or (
        ("date" in name.casefold() or name.casefold().endswith("_at"))
        and date_values.notna().sum() > 0
    )
    if is_date:
        valid_dates = date_values.dropna()
        reference = reference_date or pd.Timestamp.now(tz="UTC").tz_localize(None)
        if valid_dates.dt.tz is not None:
            valid_dates = valid_dates.dt.tz_localize(None)
        result.update(
            {
                "data_type": "date",
                "min_date": valid_dates.min().isoformat() if not valid_dates.empty else None,
                "max_date": valid_dates.max().isoformat() if not valid_dates.empty else None,
                "future_date_count": int((valid_dates > reference).sum())
                if not valid_dates.empty
                else 0,
            }
        )
        return result

    numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    if numeric:
        values = pd.to_numeric(series, errors="coerce")
        non_null = values.dropna()
        result.update(
            {
                "data_type": "numeric",
                "min": _json_value(non_null.min()) if not non_null.empty else None,
                "max": _json_value(non_null.max()) if not non_null.empty else None,
                "mean": _json_value(non_null.mean()) if not non_null.empty else None,
                "median": _json_value(non_null.median()) if not non_null.empty else None,
                "std": _json_value(non_null.std(ddof=1)) if len(non_null) > 1 else 0.0,
                "zero_count": int((non_null == 0).sum()),
                "negative_count": int((non_null < 0).sum()),
            }
        )
        result.update(_percentile_map(values, percentiles))
        return result

    text_values = series.astype("string")
    normalized = text_values.map(normalize_text)
    non_null_text = text_values.dropna()
    lengths = non_null_text.map(len)
    looks_textual = pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
    if looks_textual:
        counts = series.value_counts(dropna=True)
        normalized_counts = normalized[normalized != ""].value_counts()
        top = [
            {"value": _safe_category(value, name), "count": int(count)}
            for value, count in counts.head(top_categories).items()
        ]
        thresholds = {
            str(int(limit)): int((normalized_counts < limit).sum())
            for limit in rare_category_thresholds
        }
        result.update(
            {
                "data_type": "categorical"
                if series.nunique(dropna=True) <= max(50, top_categories * 2)
                else "text",
                "distinct_count": int(series.nunique(dropna=True)),
                "top_values": top,
                "most_frequent_count": int(counts.iloc[0]) if not counts.empty else 0,
                "rare_category_counts": thresholds,
                "singleton_category_count": int((normalized_counts == 1).sum()),
                "empty_string_count": int((text_values.fillna("").str.strip() == "").sum()),
                "empty_string_percentage": round(
                    (text_values.fillna("").str.strip() == "").sum() / row_count * 100, 6
                )
                if row_count
                else 0.0,
                "average_text_length": float(lengths.mean()) if not lengths.empty else 0.0,
                "median_text_length": float(lengths.median()) if not lengths.empty else 0.0,
                "maximum_text_length": int(lengths.max()) if not lengths.empty else 0,
                "exact_duplicate_text_percentage": round(
                    (normalized.duplicated(keep=False) & normalized.ne("")).sum() / row_count * 100,
                    6,
                )
                if row_count
                else 0.0,
                "normalized_distinct_count": int(normalized[normalized != ""].nunique()),
            }
        )
        return result

    result["data_type"] = "other"
    result["distinct_count"] = int(series.nunique(dropna=True))
    return result
