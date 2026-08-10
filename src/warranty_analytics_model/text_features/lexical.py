"""Small deterministic lexical candidates and aggregate-only text diagnostics."""

from __future__ import annotations

from statistics import median
from typing import Any

import pandas as pd

from .models import TextBuildResult, TextFeatureDefinition, TextFeatureSettings, TextTier
from .normalize import technical_tokens
from .source_policy import ALLOWED_PHASE8_TEXT_VALUE_SOURCES, APPROVED_PHASE8_TEXT_ARTIFACT

WINDOWS = ("6m", "12m", "24m", "all")
LEXICAL_METRICS = (
    "description_count",
    "unique_description_count",
    "character_count",
    "token_count",
    "unique_token_count",
    "avg_description_character_count",
    "avg_description_token_count",
)


def _finite_summary(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None}
    return {
        "min": int(min(values)),
        "mean": round(float(sum(values) / len(values)), 6),
        "median": round(float(median(values)), 6),
        "max": int(max(values)),
    }


def _description_list(document: Any, separator: str) -> list[str]:
    if not isinstance(document, str) or not document:
        return []
    return document.split(separator)


def _lexical_values(document: Any, separator: str) -> dict[str, Any]:
    descriptions = _description_list(document, separator)
    tokens = [token for description in descriptions for token in technical_tokens(description)]
    return {
        "description_count": len(descriptions),
        "unique_description_count": len(set(descriptions)),
        "character_count": len(document) if isinstance(document, str) else 0,
        "token_count": len(tokens),
        "unique_token_count": len(set(tokens)),
        "avg_description_character_count": (
            sum(len(description) for description in descriptions) / len(descriptions)
            if descriptions
            else None
        ),
        "avg_description_token_count": (len(tokens) / len(descriptions) if descriptions else None),
    }


def _quality_for_window(
    frame: pd.DataFrame,
    window: str,
    *,
    separator: str,
) -> dict[str, Any]:
    train = frame.loc[frame["split"] == "TRAIN"]
    document_column = f"prior_failure_text__{window}__document"
    values = train[document_column].tolist()
    nonempty = [value for value in values if isinstance(value, str) and value]
    lexical = [_lexical_values(value, separator) for value in nonempty]
    descriptions = [
        description for value in nonempty for description in _description_list(value, separator)
    ]
    unique_descriptions = set(descriptions)
    duplicate_rate = 1.0 - len(unique_descriptions) / len(descriptions) if descriptions else 0.0
    top_share = (
        max(descriptions.count(description) for description in unique_descriptions)
        / len(descriptions)
        if descriptions
        else 0.0
    )
    coverage = 100.0 * len(nonempty) / len(train) if len(train) else 0.0
    warnings: list[str] = []
    if coverage < 5.0:
        warnings.append("LOW_TEXT_COVERAGE")
    if descriptions and len(unique_descriptions) <= max(1, int(len(descriptions) * 0.10)):
        warnings.append("HIGH_TEXT_TAXONOMY_REPETITION")
    return {
        "train_claim_count": int(len(train)),
        "claims_with_non_null_text": int(len(nonempty)),
        "coverage_percentage": round(coverage, 6),
        "total_descriptions": int(len(descriptions)),
        "average_descriptions_per_claim_with_text": round(len(descriptions) / len(nonempty), 6)
        if nonempty
        else None,
        "document_character_count": _finite_summary([len(value) for value in nonempty]),
        "document_token_count": _finite_summary([item["token_count"] for item in lexical]),
        "unique_normalized_descriptions": int(len(unique_descriptions)),
        "normalized_description_cardinality": int(len(unique_descriptions)),
        "duplicate_description_rate": round(duplicate_rate, 6),
        "top_description_share": round(top_share, 6),
        "maximum_character_length": max((len(value) for value in nonempty), default=0),
        "maximum_token_length": max((item["token_count"] for item in lexical), default=0),
        "warnings": warnings,
    }


def build_lexical_features(
    documents: pd.DataFrame,
    settings: TextFeatureSettings,
) -> TextBuildResult:
    """Create raw document and lexical candidates without fitting any transform."""

    frame = documents.copy()
    definitions: list[TextFeatureDefinition] = []
    controls = (
        "current_warranty_claim_key",
        "prior_warranty_claim_key",
        "prior_claim__claim_date",
        "claim__claim_date",
    )
    value_sources = tuple(sorted(ALLOWED_PHASE8_TEXT_VALUE_SOURCES))
    for window in WINDOWS:
        name = f"prior_failure_text__{window}__document"
        tier: TextTier = "TEXT_EXTENDED" if window == "6m" else "TEXT_CORE"
        definitions.append(
            TextFeatureDefinition(
                feature_name=name,
                tier=tier,
                feature_type="text",
                source_artifacts=(APPROVED_PHASE8_TEXT_ARTIFACT,),
                source_columns=value_sources,
                value_sources=value_sources,
                control_sources=controls,
                window=window,
                aggregation="chronological_concatenation",
                transform=("normalize", "strict_as_of_filter", "deterministic_sort", "join"),
                separator=settings.document_separator,
                is_model_feature=True,
                notes="Canonical raw historical failure-description document; recurrence preserved.",
            )
        )
    for window in WINDOWS:
        document_column = f"prior_failure_text__{window}__document"
        for metric in LEXICAL_METRICS:
            name = f"text__{window}__{metric}"
            lexical_tier: TextTier = "TEXT_EXTENDED" if window == "6m" else "TEXT_CORE"
            values = frame[document_column].map(
                lambda value, separator=settings.document_separator, metric_name=metric: (
                    _lexical_values(value, separator)[metric_name]
                )
            )
            frame[name] = values
            definitions.append(
                TextFeatureDefinition(
                    feature_name=name,
                    tier=lexical_tier,
                    feature_type="numeric",
                    source_artifacts=(APPROVED_PHASE8_TEXT_ARTIFACT,),
                    source_columns=value_sources,
                    value_sources=value_sources,
                    control_sources=controls,
                    window=window,
                    aggregation="description_document_lexical_count",
                    transform=("normalize", "tokenize", "count"),
                    is_model_feature=True,
                    notes="Target-independent lexical statistic; averages are null with no descriptions.",
                )
            )
    presence_name = "text__has_prior_failure_description"
    frame[presence_name] = frame["prior_failure_text__all__document"].notna()
    definitions.append(
        TextFeatureDefinition(
            feature_name=presence_name,
            tier="TEXT_CORE",
            feature_type="boolean",
            source_artifacts=(APPROVED_PHASE8_TEXT_ARTIFACT,),
            source_columns=value_sources,
            value_sources=value_sources,
            control_sources=controls,
            window="all",
            aggregation="nonempty_text_presence",
            transform=("normalize", "presence"),
            is_model_feature=True,
            notes="True when at least one non-empty historical description exists.",
        )
    )
    quality: dict[str, Any] = {
        window: _quality_for_window(frame, window, separator=settings.document_separator)
        for window in WINDOWS
    }
    quality["train_feature_warnings"] = sorted(
        {warning for item in quality.values() for warning in item["warnings"]}
    )
    quality["raw_text_report_exposure"] = False
    quality["text_source"] = "prior_failure__failure_description"
    source_coverage = {
        "approved": [
            {
                "source": "prior_failure__failure_description",
                "artifact": APPROVED_PHASE8_TEXT_ARTIFACT,
                "policy": "ALLOW_HISTORICAL_POC",
                "status": "USED",
            }
        ],
        "deferred": [
            {"source": "current-claim narrative fields", "status": "PROHIBITED"},
            {"source": "failure taxonomy fields", "status": "STRUCTURED_PHASE7_ONLY"},
        ],
        "prohibited_model_feature_count": 0,
    }
    return TextBuildResult(
        frame=frame,
        definitions=definitions,
        quality=quality,
        source_coverage=source_coverage,
    )
