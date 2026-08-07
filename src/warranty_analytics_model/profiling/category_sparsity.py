"""Category-frequency and target-coverage diagnostics."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def category_sparsity(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    target_column: str = "high_cost_claim_flag",
    thresholds: Iterable[int] = (1, 5, 10, 20),
) -> list[dict[str, object]]:
    """Measure category support and target coverage without collapsing values."""

    target = (
        pd.to_numeric(frame[target_column], errors="coerce") if target_column in frame else None
    )
    output: list[dict[str, object]] = []
    for column in columns:
        if column not in frame:
            continue
        values = frame[column]
        counts = values.value_counts(dropna=False)
        non_null_counts = values.dropna().value_counts()
        item: dict[str, object] = {
            "field": column,
            "distinct_categories": int(values.nunique(dropna=True)),
            "records": int(len(values)),
            "top_category_coverage_percentage": round(float(counts.iloc[0] / len(values) * 100), 6)
            if len(values)
            else 0.0,
            "categories_with_one_record": int((non_null_counts == 1).sum()),
            "categories_below_threshold": {
                str(int(threshold)): int((non_null_counts < threshold).sum())
                for threshold in thresholds
            },
            "target_coverage": [],
        }
        if target is not None:
            grouped = pd.DataFrame({"category": values, "target": target}).dropna(subset=["target"])
            grouped = grouped[grouped.target.isin([0, 1])]
            rates = grouped.groupby("category", dropna=False, observed=True)["target"].agg(
                records="size", positive_rate="mean"
            )
            item["target_coverage"] = [
                {
                    "records": int(row.records),
                    "positive_rate_percentage": round(float(row.positive_rate) * 100, 6),
                }
                for _, row in rates.sort_values("records", ascending=False).head(100).iterrows()
            ]
        output.append(item)
    return output
