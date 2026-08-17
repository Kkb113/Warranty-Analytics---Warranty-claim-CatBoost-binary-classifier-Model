"""Target-independent Phase 14 plan construction."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import pandas as pd

from ..catboost_optimization.provenance import canonical_json_sha256
from .config import Phase14Settings, configuration_sha256
from .input import KEY, Phase14Resolved, train_oof_scores

DOMAIN_COLUMNS: dict[str, tuple[str, ...]] = {
    "truck_model": (
        "truck_model__model_name",
        "truck_model__engine_platform",
        "truck_model__cab_type",
    ),
    "truck_age": ("vehicle__days_since_delivery", "claim__months_in_service"),
    "mileage": ("claim__odometer_miles_at_failure", "telemetry__all__mileage_month__mean"),
    "warranty_policy": ("warranty_policy__coverage_months", "claim__warranty_coverage_status"),
    "customer_history": (
        "prior_claim__all__unique_failure_category_count",
        "prior_claim__days_since_last_claim",
    ),
    "prior_claim_history": (
        "prior_claim__last_failure_category",
        "prior_claim__last_failure_system",
    ),
    "service_center": ("service_location__climate_zone",),
    "region_location": ("service_location__region", "service_location__location"),
}


def _finite_numeric(values: pd.Series) -> np.ndarray:
    result = np.asarray(pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64"))
    return cast(np.ndarray, result[np.isfinite(result)])


def _dedupe_edges(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, dtype="float64")
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return []
    return [float(value) for value in np.unique(finite)]


def _categorical_values(train: pd.Series, limit: int) -> list[str]:
    normalized = train.astype("string").fillna("__MISSING__")
    values = cast(
        list[str], normalized.value_counts(dropna=False).head(int(limit)).index.astype(str).tolist()
    )
    if "__MISSING__" not in values:
        values.append("__MISSING__")
    if "__OTHER__" not in values:
        values.append("__OTHER__")
    return values


def _numeric_definition(name: str, train: pd.Series) -> dict[str, Any]:
    values = _finite_numeric(train)
    edges = _dedupe_edges(
        np.quantile(values, [0.0, 0.25, 0.50, 0.75, 1.0]) if len(values) else np.array([])
    )
    return {"kind": "numeric_quantile", "column": name, "edges": edges, "source": "TRAIN_ONLY"}


def _feature_kind(name: str, spec: Any) -> str:
    if name in spec.categorical_features or name in spec.text_features:
        return "categorical"
    return "numeric"


def build_analysis_plan(resolved: Phase14Resolved, settings: Phase14Settings) -> dict[str, Any]:
    """Construct every boundary before the caller is allowed to load validation labels."""

    train = resolved.train_features
    validation = resolved.validation_features
    oof = train_oof_scores(resolved)
    if oof[KEY].duplicated().any():
        raise ValueError("TRAIN OOF score keys are duplicated.")
    feature_names = list(resolved.feature_names)
    registry: list[dict[str, Any]] = []
    definitions: list[dict[str, Any]] = []

    def register(
        name: str,
        kind: str,
        column: str | None,
        definition: dict[str, Any] | None,
        *,
        available: bool = True,
        reason: str | None = None,
    ) -> None:
        registry.append(
            {
                "slice_id": name,
                "kind": kind,
                "column": column,
                "status": "AVAILABLE" if available else "UNAVAILABLE",
                "prediction_time_available": bool(available),
                "target_independent": True,
                "reason": reason,
            }
        )
        if definition is not None:
            definitions.append({"slice_id": name, **definition})

    date_col = "claim__claim_date"
    if date_col in validation.columns:
        register(
            "temporal_month", "temporal", date_col, {"kind": "calendar_month", "column": date_col}
        )
        register(
            "temporal_quarter",
            "temporal",
            date_col,
            {"kind": "calendar_quarter", "column": date_col},
        )
        ordered = validation[[KEY, date_col]].copy().sort_values([date_col, KEY], kind="mergesort")
        n = len(ordered)
        third = np.zeros(n, dtype="int8")
        third[n // 3 : 2 * n // 3] = 1
        third[2 * n // 3 :] = 2
        definitions.append(
            {
                "slice_id": "temporal_chronological_thirds",
                "kind": "chronological_thirds",
                "column": date_col,
                "membership_keys": {
                    label: [int(key) for key in ordered.loc[third == index, KEY].tolist()]
                    for index, label in enumerate(("EARLY", "MIDDLE", "LATE"))
                },
                "source": "VALIDATION_FEATURES_AND_DATE_ONLY",
            }
        )
        registry.append(
            {
                "slice_id": "temporal_chronological_thirds",
                "kind": "temporal",
                "column": date_col,
                "status": "AVAILABLE",
                "prediction_time_available": True,
                "target_independent": True,
            }
        )
    else:
        for name in ("temporal_month", "temporal_quarter", "temporal_chronological_thirds"):
            register(
                name,
                "temporal",
                date_col,
                None,
                available=False,
                reason="claim__claim_date unavailable",
            )

    oof_values = _finite_numeric(oof["probability"])
    score_edges = _dedupe_edges(
        np.quantile(oof_values, np.linspace(0, 1, 11)) if len(oof_values) else np.array([])
    )
    register(
        "risk_score_band",
        "risk_score",
        None,
        {"kind": "risk_score_band", "edges": score_edges, "source": "TRAIN_OOF"},
        available=bool(score_edges),
    )
    register(
        "risk_score_decile",
        "risk_score",
        None,
        {"kind": "risk_score_decile", "edges": score_edges, "source": "TRAIN_OOF"},
        available=bool(score_edges),
    )

    missing_features = [name for name in feature_names if name in train.columns]
    register(
        "feature_missingness_band",
        "missingness",
        None,
        {
            "kind": "feature_missingness_band",
            "feature_names": missing_features,
            "edges": [0.0, 1.0, 3.0, 10.0, float(max(10, len(missing_features)))],
            "source": "TRAIN_FEATURE_SCHEMA",
        },
        available=bool(missing_features),
    )
    for name, candidates in DOMAIN_COLUMNS.items():
        column = next(
            (candidate for candidate in candidates if candidate in validation.columns), None
        )
        if column is None:
            register(
                name,
                "domain",
                None,
                None,
                available=False,
                reason="no exact approved prediction-time column",
            )
            continue
        spec = next(
            (
                component.feature_set
                for component in resolved.components
                if column in component.feature_set.feature_names
            ),
            resolved.components[0].feature_set,
        )
        kind = _feature_kind(column, spec)
        definition = (
            {
                "kind": "categorical",
                "column": column,
                "categories": _categorical_values(train[column], settings.top_train_categories),
                "source": "TRAIN_ONLY",
            }
            if kind == "categorical"
            else _numeric_definition(column, train[column])
        )
        register(name, "domain", column, definition)

    # Register exact, approved model features. Definitions are frozen from
    # TRAIN feature distributions only; validation labels are not read here.
    seen = 0
    for component in resolved.components:
        for name in component.feature_set.feature_names:
            if name not in train.columns or name in {"claim__claim_date", KEY, "split"}:
                continue
            kind = _feature_kind(name, component.feature_set)
            slice_id = f"feature:{name}"
            if kind == "categorical":
                definition = {
                    "kind": "categorical",
                    "column": name,
                    "categories": _categorical_values(train[name], settings.top_train_categories),
                    "source": "TRAIN_ONLY",
                }
            else:
                definition = _numeric_definition(name, train[name])
            register(slice_id, "feature", name, definition)
            seen += 1

    plan = {
        "phase": 14,
        "phase13_run_id": resolved.phase13_manifest["run_id"],
        "slice_registry": registry,
        "slice_definitions": definitions,
        "feature_names": feature_names,
        "risk_score_train_oof_rows": int(len(oof)),
        "temporal_sort": ["claim__claim_date", KEY],
        "support_policy": {
            "min_slice_rows": settings.min_slice_rows,
            "min_slice_positives_for_ranking": settings.min_slice_positives_ranking,
            "min_slice_negatives_for_ranking": settings.min_slice_negatives_ranking,
            "min_slice_positives_for_bootstrap": settings.min_slice_positives_bootstrap,
        },
        "bootstrap_policy": {
            "seed": settings.seed,
            "overall_replicates": settings.overall_replicates,
            "material_slice_replicates": settings.material_slice_replicates,
            "confidence_level": settings.confidence_level,
            "method": "STRATIFIED_PERCENTILE",
        },
        "drift_policy": {
            "psi_low": settings.psi_low,
            "psi_high": settings.psi_high,
            "missingness_material_delta": settings.missingness_material_delta,
            "missingness_high_delta": settings.missingness_high_delta,
        },
        "threshold_diagnostic_policy": {
            "diagnostic_only": True,
            "multipliers": list(settings.threshold_multipliers),
        },
        "readiness_policy": {
            "minimum_ap_over_prevalence": settings.minimum_ap_over_prevalence,
            "minimum_roc_auc": settings.minimum_roc_auc,
        },
        "constructed_feature_definition_count": int(seen),
        "validation_targets_accessed": False,
        "test_targets_accessed": False,
        "configuration_sha256": configuration_sha256(),
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    plan["slice_registry_sha256"] = canonical_json_sha256(registry)
    plan["slice_definition_sha256"] = canonical_json_sha256(definitions)
    plan["temporal_definition_sha256"] = canonical_json_sha256(
        [
            item
            for item in definitions
            if item.get("kind") in {"calendar_month", "calendar_quarter", "chronological_thirds"}
        ]
    )
    plan["analysis_plan_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in plan.items()
            if not key.endswith("_sha256") and key != "created_at_utc"
        }
    )
    return plan


__all__ = ["DOMAIN_COLUMNS", "build_analysis_plan"]
