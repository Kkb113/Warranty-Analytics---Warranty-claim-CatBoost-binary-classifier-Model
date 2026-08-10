"""Vectorized, target-independent structured feature construction."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .models import (
    FeatureBuildResult,
    FeatureDefinition,
    StructuredFeatureError,
    StructuredFeatureSettings,
)
from .windows import (
    add_claim_dates,
    deterministic_mode,
    expected_completed_months,
    latest_series,
    month_index,
    numeric,
    per_claim_slope,
    population_std,
    safe_divide,
    window_mask,
)

CLAIM_KEY = "warranty_claim_key"
TARGET = "target__high_cost_claim_flag"
CONTROL_DATE_COLUMNS = (
    "claim__claim_date",
    "truck__build_date",
    "truck__delivery_date",
    "truck__in_service_date",
)
TEXT_COLUMNS = {
    "complaint_description",
    "diagnostic_summary",
    "technician_notes",
    "repair_notes",
    "prior_failure__failure_description",
}
RESTRICTED_TOKENS = {
    "production_batch_id",
    "component_lot_no",
    "supplier_key",
    "service_center_key",
    "truck_key",
    "vin",
    "serial",
    "technician",
    "inspector",
    "scenario",
    "group_hash",
    "cohort",
}


class FeatureRegistry:
    """Collect values and metadata while enforcing deterministic names."""

    def __init__(self, claim_keys: pd.Index) -> None:
        self.claim_keys = claim_keys
        self.values: dict[str, pd.Series] = {}
        self.definitions: list[FeatureDefinition] = []

    def add(self, values: pd.Series, definition: FeatureDefinition) -> None:
        """Add one aligned feature and reject duplicate or target-dependent definitions."""

        if definition.feature_name in self.values:
            raise StructuredFeatureError(
                f"Duplicate Phase 7 feature name: {definition.feature_name}"
            )
        if definition.target_dependent or definition.fitted_transformation is not None:
            raise StructuredFeatureError(f"Unsafe feature definition: {definition.feature_name}")
        series = values.copy()
        if not isinstance(series, pd.Series):
            series = pd.Series(series, index=self.claim_keys)
        elif isinstance(series.index, pd.RangeIndex) and len(series) == len(self.claim_keys):
            series = pd.Series(series.to_numpy(), index=self.claim_keys, name=series.name)
        self.values[definition.feature_name] = series.reindex(self.claim_keys)
        self.definitions.append(definition)

    def add_model(
        self,
        values: pd.Series,
        name: str,
        *,
        family: str,
        tier: str,
        feature_type: str,
        source_artifacts: Iterable[str],
        source_columns: Iterable[str],
        value_sources: Iterable[str] | None = None,
        control_sources: Iterable[str] = (CLAIM_KEY, "claim__claim_date"),
        window: str | None = None,
        aggregation: str | None = None,
        null_behavior: str = "preserve_null",
        minimum_observations: int | None = None,
        phase4_source_policy: str = "ALLOW_HISTORICAL_POC",
        formula: str = "",
        notes: str = "",
    ) -> None:
        """Add a safe model feature with complete lineage metadata."""

        self.add(
            values,
            FeatureDefinition(
                feature_name=name,
                family=family,
                tier=tier,  # type: ignore[arg-type]
                feature_type=feature_type,  # type: ignore[arg-type]
                source_artifacts=tuple(source_artifacts),
                source_columns=tuple(source_columns),
                value_sources=tuple(value_sources or source_columns),
                control_sources=tuple(control_sources),
                window=window,
                aggregation=aggregation,
                null_behavior=null_behavior,
                minimum_observations=minimum_observations,
                phase4_source_policy=phase4_source_policy,
                formula=formula,
                notes=notes,
            ),
        )

    def add_control(
        self,
        values: pd.Series,
        name: str,
        *,
        feature_type: str,
        source_artifacts: Iterable[str],
        source_columns: Iterable[str],
        is_lineage: bool = False,
        notes: str = "",
    ) -> None:
        """Add a control/reference column that cannot be a model feature."""

        self.add(
            values,
            FeatureDefinition(
                feature_name=name,
                family="control",
                tier="CONTROL",
                feature_type=feature_type,  # type: ignore[arg-type]
                source_artifacts=tuple(source_artifacts),
                source_columns=tuple(source_columns),
                value_sources=(),
                control_sources=(name,),
                null_behavior="preserve_null",
                is_model_feature=False,
                is_control=True,
                is_lineage=is_lineage,
                phase4_source_policy="CONTROL_ONLY",
                formula="control/reference only",
                notes=notes,
            ),
        )

    def frame(self) -> pd.DataFrame:
        """Return controls, CORE features, and EXTENDED features in stable order."""

        controls = [item.feature_name for item in self.definitions if item.tier == "CONTROL"]
        core = [item.feature_name for item in self.definitions if item.tier == "CORE"]
        extended = [item.feature_name for item in self.definitions if item.tier == "EXTENDED"]
        return pd.DataFrame(
            {name: self.values[name].to_numpy() for name in controls + core + extended},
            index=self.claim_keys,
        ).reset_index(drop=True)


def _feature_type(values: pd.Series, *, date_control: bool = False) -> str:
    if date_control or pd.api.types.is_datetime64_any_dtype(values):
        return "date_control"
    if pd.api.types.is_bool_dtype(values):
        return "boolean"
    if pd.api.types.is_numeric_dtype(values):
        return "numeric"
    return "categorical"


def _direct_value(values: pd.Series) -> pd.Series:
    """Preserve safe values while normalizing numeric SQL object columns only."""

    if pd.api.types.is_object_dtype(values):
        parsed = numeric(values)
        if values.notna().any() and parsed.notna().sum() == values.notna().sum():
            return parsed
        return values.astype("string")
    return values.copy()


def _metric(frame: pd.DataFrame, column: str, operation: str) -> pd.Series:
    """Calculate one grouped historical metric without filling measurements."""

    if frame.empty:
        return pd.Series(dtype="Float64")
    keys = frame["current_warranty_claim_key"]
    if operation == "count":
        return frame.groupby(keys, sort=True).size().astype("Int64")
    values = numeric(frame[column])
    grouped = values.groupby(keys, sort=True)
    if operation == "sum":
        return grouped.sum(min_count=1)
    if operation == "mean":
        return grouped.mean()
    if operation == "min":
        return grouped.min()
    if operation == "max":
        return grouped.max()
    if operation == "std":
        return grouped.agg(population_std)
    if operation == "nunique":
        return frame.groupby(keys, sort=True)[column].nunique(dropna=True)
    raise StructuredFeatureError(f"Unsupported historical aggregation: {operation}")


def _boolean_count(frame: pd.DataFrame, column: str) -> pd.Series:
    """Count true flags while preserving null for an empty group via reindexing later."""

    values = numeric(frame[column])
    return values.groupby(frame["current_warranty_claim_key"], sort=True).sum(min_count=1)


def _event_dates(claims: pd.DataFrame, frame: pd.DataFrame, event_column: str) -> pd.DataFrame:
    return add_claim_dates(frame, claims, event_column)


def _windows(settings: StructuredFeatureSettings) -> tuple[str, ...]:
    values = tuple(f"{item}m" for item in settings.windows_months)
    return values + (("all",) if settings.include_all_history else ())


def _add_metric(
    registry: FeatureRegistry,
    values: pd.Series,
    name: str,
    *,
    family: str,
    tier: str,
    source_artifact: str,
    source_columns: Iterable[str],
    window: str | None = None,
    aggregation: str | None = None,
    feature_type: str = "numeric",
    formula: str = "",
    notes: str = "",
    minimum_observations: int | None = None,
) -> None:
    if aggregation in {"count", "nunique", "nunique_month"}:
        values = values.reindex(registry.claim_keys).fillna(0)
    registry.add_model(
        values,
        name,
        family=family,
        tier=tier,
        feature_type=feature_type,
        source_artifacts=(source_artifact,),
        source_columns=source_columns,
        window=window,
        aggregation=aggregation,
        formula=formula,
        notes=notes,
        minimum_observations=minimum_observations,
    )


def _add_direct(registry: FeatureRegistry, snapshot: pd.DataFrame) -> None:
    """Handoff all safe non-date direct Tier A values as CORE features."""

    for column in snapshot.columns:
        if column in {CLAIM_KEY, TARGET, *CONTROL_DATE_COLUMNS} or column.startswith("lineage__"):
            continue
        if not column.startswith(
            (
                "claim__",
                "truck__",
                "truck_model__",
                "warranty_policy__",
                "claim_calendar__",
                "service_location__",
            )
        ):
            continue
        values = _direct_value(snapshot[column])
        registry.add_model(
            values,
            column,
            family="direct",
            tier="CORE",
            feature_type=_feature_type(values),
            source_artifacts=("claim_snapshot",),
            source_columns=(column,),
            value_sources=(column,),
            formula="direct Phase 5 safe claim-time value",
            notes="Phase 4-approved direct value; categorical values remain unencoded.",
            phase4_source_policy="ALLOW_BASELINE_POC",
        )


def _add_lifecycle(registry: FeatureRegistry, claims: pd.DataFrame) -> None:
    claim_date = pd.to_datetime(claims["claim__claim_date"], errors="coerce")
    build = pd.to_datetime(claims["truck__build_date"], errors="coerce")
    delivery = pd.to_datetime(claims["truck__delivery_date"], errors="coerce")
    service = pd.to_datetime(claims["truck__in_service_date"], errors="coerce")
    for name, values, formula in (
        ("vehicle__age_days", (claim_date - build).dt.days, "claim_date - build_date"),
        (
            "vehicle__days_since_delivery",
            (claim_date - delivery).dt.days,
            "claim_date - delivery_date",
        ),
        (
            "vehicle__days_in_service",
            (claim_date - service).dt.days,
            "claim_date - in_service_date",
        ),
        (
            "vehicle__build_to_delivery_days",
            (delivery - build).dt.days,
            "delivery_date - build_date",
        ),
        (
            "vehicle__delivery_to_in_service_days",
            (service - delivery).dt.days,
            "in_service_date - delivery_date",
        ),
    ):
        _add_metric(
            registry,
            values,
            name,
            family="lifecycle",
            tier="CORE",
            source_artifact="claim_snapshot",
            source_columns=(
                "claim__claim_date",
                "truck__build_date",
                "truck__delivery_date",
                "truck__in_service_date",
            ),
            aggregation="date_arithmetic",
            formula=formula,
            notes="Date arithmetic only; raw date controls remain available separately.",
        )


def _add_usage_and_warranty(registry: FeatureRegistry, claims: pd.DataFrame) -> None:
    months = numeric(claims["claim__months_in_service"])
    miles = numeric(claims["claim__odometer_miles_at_failure"])
    hours = numeric(claims["claim__engine_hours_at_failure"])
    coverage_months = numeric(claims["warranty_policy__coverage_months"])
    coverage_miles = numeric(claims["warranty_policy__coverage_miles"])
    coverage_hours = numeric(claims["warranty_policy__coverage_engine_hours"])
    for name, values, columns, formula in (
        (
            "usage__miles_per_month_in_service",
            safe_divide(miles, months),
            ("claim__odometer_miles_at_failure", "claim__months_in_service"),
            "odometer_miles_at_failure / months_in_service",
        ),
        (
            "usage__engine_hours_per_month_in_service",
            safe_divide(hours, months),
            ("claim__engine_hours_at_failure", "claim__months_in_service"),
            "engine_hours_at_failure / months_in_service",
        ),
        (
            "usage__miles_per_engine_hour",
            safe_divide(miles, hours),
            ("claim__odometer_miles_at_failure", "claim__engine_hours_at_failure"),
            "odometer_miles_at_failure / engine_hours_at_failure",
        ),
    ):
        _add_metric(
            registry,
            values,
            name,
            family="usage",
            tier="CORE",
            source_artifact="claim_snapshot",
            source_columns=columns,
            formula=formula,
            notes="Invalid or non-positive denominator returns NULL; no imputation or clipping.",
        )
    ratios = {
        "warranty__months_utilization_ratio": (
            months,
            coverage_months,
            "months_in_service / coverage_months",
        ),
        "warranty__mileage_utilization_ratio": (
            miles,
            coverage_miles,
            "odometer_miles_at_failure / coverage_miles",
        ),
        "warranty__engine_hours_utilization_ratio": (
            hours,
            coverage_hours,
            "engine_hours_at_failure / coverage_engine_hours",
        ),
    }
    ratio_values: list[pd.Series] = []
    ratio_columns: list[str] = []
    for name, (numerator, denominator, formula) in ratios.items():
        values = safe_divide(numerator, denominator)
        ratio_values.append(values)
        ratio_columns.append(name)
        source = {
            "warranty__months_utilization_ratio": (
                "claim__months_in_service",
                "warranty_policy__coverage_months",
            ),
            "warranty__mileage_utilization_ratio": (
                "claim__odometer_miles_at_failure",
                "warranty_policy__coverage_miles",
            ),
            "warranty__engine_hours_utilization_ratio": (
                "claim__engine_hours_at_failure",
                "warranty_policy__coverage_engine_hours",
            ),
        }[name]
        _add_metric(
            registry,
            values,
            name,
            family="warranty",
            tier="CORE",
            source_artifact="claim_snapshot",
            source_columns=source,
            formula=formula,
            notes="Invalid or non-positive denominator returns NULL; values above 1 are preserved.",
        )
    for name, values, source, formula in (
        (
            "warranty__months_remaining",
            coverage_months - months,
            ("warranty_policy__coverage_months", "claim__months_in_service"),
            "coverage_months - months_in_service",
        ),
        (
            "warranty__miles_remaining",
            coverage_miles - miles,
            ("warranty_policy__coverage_miles", "claim__odometer_miles_at_failure"),
            "coverage_miles - odometer_miles_at_failure",
        ),
        (
            "warranty__engine_hours_remaining",
            coverage_hours - hours,
            ("warranty_policy__coverage_engine_hours", "claim__engine_hours_at_failure"),
            "coverage_engine_hours - engine_hours_at_failure",
        ),
    ):
        _add_metric(
            registry,
            values,
            name,
            family="warranty",
            tier="CORE",
            source_artifact="claim_snapshot",
            source_columns=source,
            formula=formula,
        )
    stacked = pd.concat(ratio_values, axis=1)
    _add_metric(
        registry,
        stacked.max(axis=1, skipna=True).where(stacked.notna().any(axis=1)),
        "warranty__max_utilization_ratio",
        family="warranty",
        tier="CORE",
        source_artifact="claim_snapshot",
        source_columns=(
            "warranty_policy__coverage_months",
            "warranty_policy__coverage_miles",
            "warranty_policy__coverage_engine_hours",
        ),
        aggregation="max",
        formula="max(months_utilization_ratio, mileage_utilization_ratio, engine_hours_utilization_ratio)",
    )


def _add_telemetry(
    registry: FeatureRegistry,
    claims: pd.DataFrame,
    telemetry: pd.DataFrame,
    settings: StructuredFeatureSettings,
) -> None:
    source = "telemetry_history"
    prepared = add_claim_dates(telemetry, claims, "telemetry__month_start_date", telemetry=True)
    additive = (
        "telemetry__mileage_month",
        "telemetry__engine_hours_month",
        "telemetry__idle_hours_month",
        "telemetry__low_oil_pressure_events",
        "telemetry__brake_air_pressure_alerts",
        "telemetry__battery_voltage_alerts",
        "telemetry__fault_code_count",
        "telemetry__harsh_braking_events",
    )
    continuous = (
        "telemetry__avg_engine_temp",
        "telemetry__max_engine_temp",
        "telemetry__avg_oil_pressure",
        "telemetry__avg_payload_weight",
        "telemetry__fuel_efficiency_mpg",
        "telemetry__route_severity_score",
        "telemetry__maintenance_compliance_score",
    )
    all_measures = additive + continuous + ("telemetry__total_odometer_miles",)
    window_event_frames: dict[str, pd.DataFrame] = {}
    for window in _windows(settings):
        current = prepared.loc[window_mask(prepared, window)].copy()
        window_event_frames[window] = current
        observed = current.groupby("current_warranty_claim_key", sort=True)["_event_date"].nunique()
        _add_metric(
            registry,
            observed,
            f"telemetry__{window}__months_observed",
            family="telemetry",
            tier="CORE",
            source_artifact=source,
            source_columns=("telemetry__month_start_date",),
            window=window,
            aggregation="nunique_month",
            formula="count of safe completed telemetry month_end values",
        )
        if window != "all":
            expected = expected_completed_months(
                claims["truck__in_service_date"],
                claims["claim__claim_date"],
                int(window.removesuffix("m")),
            )
            expected.index = registry.claim_keys
            coverage = safe_divide(observed.reindex(registry.claim_keys), expected)
            _add_metric(
                registry,
                coverage,
                f"telemetry__{window}__coverage_ratio",
                family="telemetry",
                tier="CORE",
                source_artifact=source,
                source_columns=(
                    "telemetry__month_start_date",
                    "truck__in_service_date",
                    "claim__claim_date",
                ),
                window=window,
                aggregation="coverage_ratio",
                formula="months_observed / eligible_completed_months",
                notes="Claim month is excluded; denominator uses eligible calendar months.",
            )
        for column in additive:
            if column not in current:
                continue
            for operation in ("sum", "mean"):
                _add_metric(
                    registry,
                    _metric(current, column, operation),
                    f"telemetry__{window}__{column.removeprefix('telemetry__')}__{operation}",
                    family="telemetry",
                    tier="CORE",
                    source_artifact=source,
                    source_columns=(column, "telemetry__month_start_date"),
                    window=window,
                    aggregation=operation,
                    formula=f"{operation}({column}) over safe completed telemetry months",
                )
            for operation in ("max", "std"):
                _add_metric(
                    registry,
                    _metric(current, column, operation),
                    f"telemetry__{window}__{column.removeprefix('telemetry__')}__{operation}",
                    family="telemetry",
                    tier="EXTENDED",
                    source_artifact=source,
                    source_columns=(column, "telemetry__month_start_date"),
                    window=window,
                    aggregation=operation,
                    formula=f"{operation}({column}) over safe completed telemetry months",
                    notes="Population standard deviation; at least two observations required.",
                    minimum_observations=2 if operation == "std" else None,
                )
        for column in continuous:
            if column not in current:
                continue
            if window in {"3m", "6m", "12m"}:
                tier = "CORE"
                operations: tuple[str, ...] = ("mean",)
            else:
                tier = "EXTENDED"
                operations = ("mean", "min", "max", "std")
            for operation in operations:
                _add_metric(
                    registry,
                    _metric(current, column, operation),
                    f"telemetry__{window}__{column.removeprefix('telemetry__')}__{operation}",
                    family="telemetry",
                    tier=tier,
                    source_artifact=source,
                    source_columns=(column, "telemetry__month_start_date"),
                    window=window,
                    aggregation=operation,
                    formula=f"{operation}({column}) over safe completed telemetry months",
                    notes="Population standard deviation; at least two observations required."
                    if operation == "std"
                    else "Missing observations remain NULL.",
                    minimum_observations=2 if operation == "std" else None,
                )
        engine = (
            _metric(current, "telemetry__engine_hours_month", "sum")
            if "telemetry__engine_hours_month" in current
            else pd.Series(dtype="Float64")
        )
        idle = (
            _metric(current, "telemetry__idle_hours_month", "sum")
            if "telemetry__idle_hours_month" in current
            else pd.Series(dtype="Float64")
        )
        _add_metric(
            registry,
            safe_divide(idle.reindex(registry.claim_keys), engine.reindex(registry.claim_keys)),
            f"telemetry__{window}__idle_engine_hour_ratio",
            family="telemetry",
            tier="CORE",
            source_artifact=source,
            source_columns=("telemetry__idle_hours_month", "telemetry__engine_hours_month"),
            window=window,
            aggregation="ratio",
            formula="SUM(idle_hours_month) / SUM(engine_hours_month)",
            notes="Denominator <= 0 returns NULL.",
        )
    for column in all_measures:
        if column not in prepared:
            continue
        latest = latest_series(prepared, column, "telemetry_month_key")
        if column == "telemetry__total_odometer_miles":
            _add_metric(
                registry,
                numeric(latest),
                "telemetry__latest_total_odometer_miles",
                family="telemetry",
                tier="CORE",
                source_artifact=source,
                source_columns=(column, "telemetry__month_start_date"),
                aggregation="latest",
                formula="latest safe completed total_odometer_miles",
            )
        _add_metric(
            registry,
            _direct_value(latest),
            f"telemetry__latest__{column.removeprefix('telemetry__')}",
            family="telemetry",
            tier="CORE",
            source_artifact=source,
            source_columns=(column, "telemetry__month_start_date"),
            aggregation="latest",
            feature_type="numeric",
            formula="latest safe completed telemetry month value",
        )
    claims_indexed = claims.set_index(CLAIM_KEY)
    latest_date = latest_series(prepared, "_event_date", "telemetry_month_key")
    days_since = (
        claims_indexed["claim__claim_date"]
        .pipe(pd.to_datetime, errors="coerce")
        .sub(pd.to_datetime(latest_date, errors="coerce"))
        .dt.days
    )
    _add_metric(
        registry,
        days_since,
        "telemetry__days_since_latest_completed_month",
        family="telemetry",
        tier="CORE",
        source_artifact=source,
        source_columns=("telemetry__month_start_date", "claim__claim_date"),
        aggregation="recency",
        formula="claim_date - latest completed telemetry month_end",
    )
    all_frame = window_event_frames.get("all", prepared.iloc[0:0])
    if not all_frame.empty:
        min_index = (
            all_frame.assign(_month_index=month_index(all_frame["_event_date"]))
            .groupby("current_warranty_claim_key", sort=True)["_month_index"]
            .min()
        )
        max_index = (
            all_frame.assign(_month_index=month_index(all_frame["_event_date"]))
            .groupby("current_warranty_claim_key", sort=True)["_month_index"]
            .max()
        )
        observed = all_frame.groupby("current_warranty_claim_key", sort=True)[
            "_event_date"
        ].nunique()
        span = (max_index - min_index + 1).astype("Float64")
        missing = span - observed.reindex(span.index).astype("Float64")
    else:
        span = pd.Series(dtype="Float64")
        missing = pd.Series(dtype="Float64")
    _add_metric(
        registry,
        span,
        "telemetry__history_span_months",
        family="telemetry",
        tier="CORE",
        source_artifact=source,
        source_columns=("telemetry__month_start_date",),
        aggregation="calendar_span",
        formula="max(month_index) - min(month_index) + 1",
    )
    _add_metric(
        registry,
        missing,
        "telemetry__missing_months_within_observed_span",
        family="telemetry",
        tier="CORE",
        source_artifact=source,
        source_columns=("telemetry__month_start_date",),
        aggregation="missing_months",
        formula="history_span_months - months_observed",
    )
    odometer = "telemetry__total_odometer_miles"
    for window in ("3m", "6m", "12m", "24m"):
        current = window_event_frames[window]
        if odometer not in current:
            continue
        latest = latest_series(current, odometer, "telemetry_month_key")
        earliest = (
            current.sort_values(
                ["current_warranty_claim_key", "_event_date", "telemetry_month_key"],
                kind="mergesort",
            )
            .drop_duplicates("current_warranty_claim_key", keep="first")
            .set_index("current_warranty_claim_key")[odometer]
        )
        _add_metric(
            registry,
            numeric(latest).sub(numeric(earliest)),
            f"telemetry__{window}__odometer_change",
            family="telemetry",
            tier="CORE",
            source_artifact=source,
            source_columns=(odometer, "telemetry__month_start_date"),
            window=window,
            aggregation="latest_minus_earliest",
            formula="latest total_odometer_miles - earliest total_odometer_miles",
        )
    trend_columns = (
        "telemetry__mileage_month",
        "telemetry__engine_hours_month",
        "telemetry__fault_code_count",
        "telemetry__avg_engine_temp",
        "telemetry__avg_oil_pressure",
        "telemetry__fuel_efficiency_mpg",
        "telemetry__route_severity_score",
        "telemetry__maintenance_compliance_score",
    )
    for window in ("6m", "12m", "24m"):
        current = window_event_frames[window]
        for column in trend_columns:
            if column not in current:
                continue
            slope = per_claim_slope(current, column, minimum=settings.slope_min_observations)
            _add_metric(
                registry,
                slope,
                f"telemetry__{window}__{column.removeprefix('telemetry__')}__slope",
                family="telemetry",
                tier="EXTENDED",
                source_artifact=source,
                source_columns=(column, "telemetry__month_start_date"),
                window=window,
                aggregation="least_squares_slope",
                formula="per-claim least-squares slope on year * 12 + month",
                minimum_observations=settings.slope_min_observations,
                notes="Same-month values are averaged before slope calculation; missing months are not reindexed.",
            )
            latest = latest_series(current, column, "telemetry_month_key")
            earliest = (
                current.sort_values(
                    ["current_warranty_claim_key", "_event_date", "telemetry_month_key"],
                    kind="mergesort",
                )
                .drop_duplicates("current_warranty_claim_key", keep="first")
                .set_index("current_warranty_claim_key")[column]
            )
            change = numeric(latest).sub(numeric(earliest))
            counts = current.groupby("current_warranty_claim_key")[column].count()
            change = change.where(counts.reindex(change.index).ge(2))
            _add_metric(
                registry,
                change,
                f"telemetry__{window}__{column.removeprefix('telemetry__')}__change",
                family="telemetry",
                tier="EXTENDED",
                source_artifact=source,
                source_columns=(column, "telemetry__month_start_date"),
                window=window,
                aggregation="latest_minus_earliest",
                formula="latest value - earliest value within safe window",
                minimum_observations=2,
            )


def _add_event_family(
    registry: FeatureRegistry,
    claims: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    family: str,
    artifact: str,
    date_column: str,
    key_column: str,
    type_column: str,
    odometer_column: str,
    engine_hours_column: str,
    settings: StructuredFeatureSettings,
    maintenance: bool,
) -> pd.DataFrame:
    prepared = add_claim_dates(frame, claims, date_column)
    for window in _windows(settings):
        current = prepared.loc[window_mask(prepared, window)].copy()
        event_prefix = f"{family}__{window}__"
        _add_metric(
            registry,
            _metric(current, key_column, "count"),
            f"{event_prefix}event_count",
            family=family,
            tier="CORE",
            source_artifact=artifact,
            source_columns=(date_column, key_column),
            window=window,
            aggregation="count",
            formula="count of safe historical event rows",
        )
        if maintenance:
            for name, values, formula, source in (
                (
                    "scheduled_count",
                    _boolean_count(current, "maintenance__scheduled_flag"),
                    "count(scheduled_flag = true)",
                    "maintenance__scheduled_flag",
                ),
                (
                    "unscheduled_count",
                    _boolean_count(
                        current.assign(_inverse=~current["maintenance__scheduled_flag"]), "_inverse"
                    ),
                    "count(scheduled_flag = false)",
                    "maintenance__scheduled_flag",
                ),
                (
                    "on_time_count",
                    _boolean_count(current, "maintenance__completed_on_time_flag"),
                    "count(completed_on_time_flag = true)",
                    "maintenance__completed_on_time_flag",
                ),
                (
                    "overdue_event_count",
                    _metric(
                        current.loc[numeric(current["maintenance__overdue_days"]) > 0],
                        key_column,
                        "count",
                    ),
                    "count(overdue_days > 0)",
                    "maintenance__overdue_days",
                ),
            ):
                _add_metric(
                    registry,
                    values,
                    f"{event_prefix}{name}",
                    family=family,
                    tier="CORE",
                    source_artifact=artifact,
                    source_columns=(source, date_column),
                    window=window,
                    aggregation="count",
                    formula=formula,
                )
        if maintenance and window in {"6m", "12m", "24m", "all"}:
            cost = _metric(current, "maintenance__maintenance_cost", "sum")
            overdue_mean = _metric(current, "maintenance__overdue_days", "mean")
            overdue_max = _metric(current, "maintenance__overdue_days", "max")
            for name, values, column, operation in (
                ("maintenance_cost_sum", cost, "maintenance__maintenance_cost", "sum"),
                ("overdue_days_mean", overdue_mean, "maintenance__overdue_days", "mean"),
                ("overdue_days_max", overdue_max, "maintenance__overdue_days", "max"),
            ):
                _add_metric(
                    registry,
                    values,
                    f"{event_prefix}{name}",
                    family=family,
                    tier="CORE",
                    source_artifact=artifact,
                    source_columns=(column, date_column),
                    window=window,
                    aggregation=operation,
                    formula=f"{operation}({column}) over safe events; no events remains NULL",
                )
            event_count = _metric(current, key_column, "count")
            scheduled = _boolean_count(current, "maintenance__scheduled_flag")
            on_time = _boolean_count(current, "maintenance__completed_on_time_flag")
            for name, values, formula, column in (
                (
                    "scheduled_ratio",
                    safe_divide(
                        scheduled.reindex(registry.claim_keys),
                        event_count.reindex(registry.claim_keys),
                    ),
                    "scheduled_count / event_count",
                    "maintenance__scheduled_flag",
                ),
                (
                    "on_time_ratio",
                    safe_divide(
                        on_time.reindex(registry.claim_keys),
                        event_count.reindex(registry.claim_keys),
                    ),
                    "on_time_count / event_count",
                    "maintenance__completed_on_time_flag",
                ),
            ):
                _add_metric(
                    registry,
                    values,
                    f"{event_prefix}{name}",
                    family=family,
                    tier="CORE",
                    source_artifact=artifact,
                    source_columns=(column, date_column),
                    window=window,
                    aggregation="ratio",
                    formula=formula,
                    notes="No qualifying events returns NULL.",
                )
    latest_date = latest_series(prepared, "_event_date", key_column)
    claim_indexed = claims.set_index(CLAIM_KEY)
    claim_date = pd.to_datetime(claim_indexed["claim__claim_date"], errors="coerce")
    _add_metric(
        registry,
        claim_date.sub(pd.to_datetime(latest_date, errors="coerce")).dt.days,
        f"{family}__days_since_last_event",
        family=family,
        tier="CORE",
        source_artifact=artifact,
        source_columns=(date_column, "claim__claim_date"),
        aggregation="recency",
        formula="claim_date - latest safe event date",
    )
    for output, value_column, formula in (
        (
            f"{family}__miles_since_last_event",
            odometer_column,
            f"claim__odometer_miles_at_failure - latest {odometer_column}",
        ),
        (
            f"{family}__engine_hours_since_last_event",
            engine_hours_column,
            f"claim__engine_hours_at_failure - latest {engine_hours_column}",
        ),
    ):
        if value_column not in prepared:
            continue
        latest_value = latest_series(prepared, value_column, key_column)
        claim_value = numeric(
            claim_indexed["claim__odometer_miles_at_failure"]
            if "miles" in output
            else claim_indexed["claim__engine_hours_at_failure"]
        )
        _add_metric(
            registry,
            claim_value.sub(numeric(latest_value)),
            output,
            family=family,
            tier="CORE",
            source_artifact=artifact,
            source_columns=(value_column, date_column),
            aggregation="latest_difference",
            formula=formula,
        )
    if type_column in prepared:
        type_count = prepared.groupby("current_warranty_claim_key", sort=True)[type_column].nunique(
            dropna=True
        )
        _add_metric(
            registry,
            type_count,
            f"{family}__all__unique_type_count",
            family=family,
            tier="CORE",
            source_artifact=artifact,
            source_columns=(type_column, date_column),
            window="all",
            aggregation="nunique",
            formula=f"unique non-null {type_column} values over all safe history",
        )
        last_type = latest_series(prepared, type_column, key_column)
        _add_metric(
            registry,
            last_type.astype("string"),
            f"{family}__last_type",
            family=family,
            tier="CORE",
            feature_type="categorical",
            source_artifact=artifact,
            source_columns=(type_column, date_column, key_column),
            aggregation="latest",
            formula="latest safe event type; source key breaks same-date ties",
            notes="Categorical values remain strings and are not encoded.",
        )
        if maintenance:
            mode = prepared.groupby("current_warranty_claim_key", sort=True)[type_column].agg(
                deterministic_mode
            )
            dominant = prepared.groupby(
                ["current_warranty_claim_key", type_column], dropna=True
            ).size()
            dominant_share = (
                dominant.groupby(level=0)
                .max()
                .div(prepared.groupby("current_warranty_claim_key").size())
            )
            _add_metric(
                registry,
                mode.astype("string"),
                "maintenance__all__dominant_type",
                family=family,
                tier="EXTENDED",
                feature_type="categorical",
                source_artifact=artifact,
                source_columns=(type_column, date_column),
                window="all",
                aggregation="mode",
                formula="highest frequency maintenance type; lexical tie break",
                notes="Mode is target-independent and deterministic.",
            )
            _add_metric(
                registry,
                dominant_share,
                "maintenance__all__dominant_type_share",
                family=family,
                tier="EXTENDED",
                source_artifact=artifact,
                source_columns=(type_column, date_column),
                window="all",
                aggregation="dominant_share",
                formula="dominant maintenance type count / event count",
            )
    if not prepared.empty:
        intervals = prepared.sort_values(
            ["current_warranty_claim_key", "_event_date", key_column], kind="mergesort"
        ).copy()
        intervals["_inter_event_days"] = (
            intervals.groupby("current_warranty_claim_key")["_event_date"].diff().dt.days
        )
        interval_mean = intervals.groupby("current_warranty_claim_key")["_inter_event_days"].mean()
        interval_std = intervals.groupby("current_warranty_claim_key")["_inter_event_days"].agg(
            population_std
        )
    else:
        interval_mean = pd.Series(dtype="Float64")
        interval_std = pd.Series(dtype="Float64")
    _add_metric(
        registry,
        interval_mean,
        f"{family}__mean_inter_event_days",
        family=family,
        tier="EXTENDED",
        source_artifact=artifact,
        source_columns=(date_column, key_column),
        aggregation="mean_interval",
        formula="mean actual calendar days between safe events",
        minimum_observations=2,
    )
    _add_metric(
        registry,
        interval_std,
        f"{family}__std_inter_event_days",
        family=family,
        tier="EXTENDED",
        source_artifact=artifact,
        source_columns=(date_column, key_column),
        aggregation="std_interval",
        formula="population std of actual inter-event days",
        minimum_observations=2,
        notes="Population standard deviation (ddof=0).",
    )
    return prepared


def _add_component(
    registry: FeatureRegistry,
    claims: pd.DataFrame,
    frame: pd.DataFrame,
    settings: StructuredFeatureSettings,
) -> None:
    artifact = "component_installation_history"
    date_column = "component_installation__installed_date"
    prepared = add_claim_dates(frame, claims, date_column)
    for window in ("6m", "12m", "24m", "all"):
        current = prepared.loc[window_mask(prepared, window)]
        _add_metric(
            registry,
            _metric(current, "installation_key", "count"),
            f"component__{window}__installation_event_count",
            family="component",
            tier="CORE",
            source_artifact=artifact,
            source_columns=("installation_key", date_column),
            window=window,
            aggregation="count",
            formula="count of historical installation events; not active components",
        )
    event_count = _metric(prepared, "installation_key", "count")
    for name, values, formula, column in (
        (
            "component__all__rework_count",
            _boolean_count(prepared, "component_installation__rework_flag"),
            "count(rework_flag = true)",
            "component_installation__rework_flag",
        ),
        (
            "component__all__safety_critical_count",
            _boolean_count(prepared, "component__is_safety_critical"),
            "count(is_safety_critical = true)",
            "component__is_safety_critical",
        ),
    ):
        _add_metric(
            registry,
            values,
            name,
            family="component",
            tier="CORE",
            source_artifact=artifact,
            source_columns=(column, date_column),
            aggregation="count",
            formula=formula,
        )
    for name, numerator, formula, column in (
        (
            "component__all__rework_ratio",
            _boolean_count(prepared, "component_installation__rework_flag"),
            "rework_count / installation_event_count",
            "component_installation__rework_flag",
        ),
        (
            "component__all__safety_critical_ratio",
            _boolean_count(prepared, "component__is_safety_critical"),
            "safety_critical_count / installation_event_count",
            "component__is_safety_critical",
        ),
    ):
        _add_metric(
            registry,
            safe_divide(
                numerator.reindex(registry.claim_keys), event_count.reindex(registry.claim_keys)
            ),
            name,
            family="component",
            tier="CORE",
            source_artifact=artifact,
            source_columns=(column, date_column),
            aggregation="ratio",
            formula=formula,
            notes="No qualifying events returns NULL.",
        )
    for name, column in (
        ("component__all__unique_system_count", "component__component_system"),
        ("component__all__unique_category_count", "component__component_category"),
        (
            "component__all__quality_check_status_unique_count",
            "component_installation__quality_check_status",
        ),
    ):
        _add_metric(
            registry,
            _metric(prepared, column, "nunique"),
            name,
            family="component",
            tier="CORE",
            source_artifact=artifact,
            source_columns=(column, date_column),
            aggregation="nunique",
            formula=f"unique non-null {column} values over all safe installations",
        )
    for name, column, tier, operations in (
        (
            "inspection_score",
            "component_installation__inspection_score",
            "CORE",
            ("mean", "min", "max"),
        ),
        ("torque_value", "component_installation__torque_value", "EXTENDED", ("mean", "std")),
        ("unit_cost", "component__unit_cost", "CORE", ("mean", "max", "sum")),
        ("standard_life_miles", "component__standard_life_miles", "EXTENDED", ("mean", "min")),
        ("standard_life_months", "component__standard_life_months", "CORE", ("mean", "min")),
    ):
        if column not in prepared:
            continue
        for operation in operations:
            _add_metric(
                registry,
                _metric(prepared, column, operation),
                f"component__all__{name}_{operation}",
                family="component",
                tier=tier,
                source_artifact=artifact,
                source_columns=(column, date_column),
                window="all",
                aggregation=operation,
                formula=f"{operation}({column}) over all safe installation history",
            )
    age_days = (
        pd.to_datetime(prepared["_claim_date"], errors="coerce")
        - pd.to_datetime(prepared["_event_date"], errors="coerce")
    ).dt.days
    prepared = prepared.assign(
        _component_age_days=age_days, _component_age_months=age_days / 30.4375
    )
    for name, column, operation in (
        ("component__all__component_age_days_mean", "_component_age_days", "mean"),
        ("component__all__component_age_days_min", "_component_age_days", "min"),
        ("component__all__component_age_days_max", "_component_age_days", "max"),
    ):
        _add_metric(
            registry,
            _metric(prepared, column, operation),
            name,
            family="component",
            tier="CORE",
            source_artifact=artifact,
            source_columns=(date_column, "claim__claim_date"),
            window="all",
            aggregation=operation,
            formula="claim_date - installed_date in days",
        )
    latest = latest_series(prepared, date_column, "installation_key")
    claim_date = pd.to_datetime(claims.set_index(CLAIM_KEY)["claim__claim_date"], errors="coerce")
    _add_metric(
        registry,
        claim_date.sub(pd.to_datetime(latest, errors="coerce")).dt.days,
        "component__days_since_latest_installation",
        family="component",
        tier="CORE",
        source_artifact=artifact,
        source_columns=(date_column, "claim__claim_date"),
        aggregation="recency",
        formula="claim_date - latest safe installation date",
    )
    if "component__standard_life_months" in prepared:
        utilization = safe_divide(
            prepared["_component_age_months"], prepared["component__standard_life_months"]
        )
        prepared = prepared.assign(_life_utilization=utilization)
        for name, operation in (
            ("component__all__life_utilization_mean", "mean"),
            ("component__all__life_utilization_max", "max"),
        ):
            _add_metric(
                registry,
                _metric(prepared, "_life_utilization", operation),
                name,
                family="component",
                tier="EXTENDED",
                source_artifact=artifact,
                source_columns=(date_column, "component__standard_life_months"),
                window="all",
                aggregation=operation,
                formula="component_age_months / standard_life_months",
                notes="Engineering utilization ratio; not a failure threshold.",
            )
        at_one = numeric(prepared["_life_utilization"]).ge(1).astype("Int64")
        at_one_count = at_one.groupby(prepared["current_warranty_claim_key"], sort=True).sum(
            min_count=1
        )
        _add_metric(
            registry,
            safe_divide(
                at_one_count.reindex(registry.claim_keys), event_count.reindex(registry.claim_keys)
            ),
            "component__all__life_utilization_at_or_above_1_ratio",
            family="component",
            tier="EXTENDED",
            source_artifact=artifact,
            source_columns=(date_column, "component__standard_life_months"),
            window="all",
            aggregation="ratio",
            formula="count(component_age_months / standard_life_months >= 1) / installation_event_count",
        )


def _add_prior_claims(
    registry: FeatureRegistry,
    claims: pd.DataFrame,
    frame: pd.DataFrame,
    settings: StructuredFeatureSettings,
) -> None:
    artifact = "prior_claim_history"
    date_column = "prior_claim__claim_date"
    prepared = add_claim_dates(frame, claims, date_column)
    for window in _windows(settings):
        current = prepared.loc[window_mask(prepared, window)]
        _add_metric(
            registry,
            _metric(current, "prior_warranty_claim_key", "count"),
            f"prior_claim__{window}__claim_count",
            family="prior_claim",
            tier="CORE",
            source_artifact=artifact,
            source_columns=("prior_warranty_claim_key", date_column),
            window=window,
            aggregation="count",
            formula="count of safe prior claims",
        )
        if window in {"6m", "12m", "24m"}:
            for name, column in (
                ("safety_related_count", "prior_failure__safety_related_flag"),
                ("recall_related_count", "prior_failure__recall_related_flag"),
            ):
                _add_metric(
                    registry,
                    _boolean_count(current, column),
                    f"prior_claim__{window}__{name}",
                    family="prior_claim",
                    tier="CORE",
                    source_artifact=artifact,
                    source_columns=(column, date_column),
                    window=window,
                    aggregation="count",
                    formula=f"count({column} = true)",
                )
            for name, column in (
                ("unique_failure_system_count", "prior_failure__failure_system"),
                ("unique_failure_category_count", "prior_failure__failure_category"),
            ):
                _add_metric(
                    registry,
                    _metric(current, column, "nunique"),
                    f"prior_claim__{window}__{name}",
                    family="prior_claim",
                    tier="EXTENDED",
                    source_artifact=artifact,
                    source_columns=(column, date_column),
                    window=window,
                    aggregation="nunique",
                    formula=f"unique non-null {column} values",
                )
    event_count = _metric(prepared, "prior_warranty_claim_key", "count")
    for name, column in (
        ("unique_failure_code_count", "prior_failure__failure_code"),
        ("unique_failure_system_count", "prior_failure__failure_system"),
        ("unique_failure_category_count", "prior_failure__failure_category"),
    ):
        _add_metric(
            registry,
            _metric(prepared, column, "nunique"),
            f"prior_claim__all__{name}",
            family="prior_claim",
            tier="CORE",
            source_artifact=artifact,
            source_columns=(column, date_column),
            window="all",
            aggregation="nunique",
            formula=f"unique non-null {column} values over all safe prior claims",
        )
    for name, column in (
        ("safety_related_count", "prior_failure__safety_related_flag"),
        ("recall_related_count", "prior_failure__recall_related_flag"),
    ):
        count = _boolean_count(prepared, column)
        _add_metric(
            registry,
            count,
            f"prior_claim__all__{name}",
            family="prior_claim",
            tier="CORE",
            source_artifact=artifact,
            source_columns=(column, date_column),
            window="all",
            aggregation="count",
            formula=f"count({column} = true)",
        )
        _add_metric(
            registry,
            safe_divide(
                count.reindex(registry.claim_keys), event_count.reindex(registry.claim_keys)
            ),
            f"prior_claim__all__{name.removesuffix('_count')}_ratio",
            family="prior_claim",
            tier="CORE",
            source_artifact=artifact,
            source_columns=(column, date_column),
            window="all",
            aggregation="ratio",
            formula=f"{name} / prior_claim__all__claim_count",
        )
    key_columns = (
        ("prior_claim__last_failure_code", "prior_failure__failure_code"),
        ("prior_claim__last_failure_system", "prior_failure__failure_system"),
        ("prior_claim__last_failure_category", "prior_failure__failure_category"),
        ("prior_claim__last_severity_level", "prior_failure__severity_level"),
    )
    for name, column in key_columns:
        _add_metric(
            registry,
            latest_series(prepared, column, "prior_warranty_claim_key").astype("string"),
            name,
            family="prior_claim",
            tier="CORE" if "severity" not in name else "EXTENDED",
            feature_type="categorical",
            source_artifact=artifact,
            source_columns=(column, date_column, "prior_warranty_claim_key"),
            aggregation="latest",
            formula="latest safe taxonomy value; source key breaks same-date ties",
            notes="Categorical values remain strings; severity_level is treated as categorical.",
        )
    claim_indexed = claims.set_index(CLAIM_KEY)
    latest_date = latest_series(prepared, date_column, "prior_warranty_claim_key")
    _add_metric(
        registry,
        pd.to_datetime(claim_indexed["claim__claim_date"], errors="coerce")
        .sub(pd.to_datetime(latest_date, errors="coerce"))
        .dt.days,
        "prior_claim__days_since_last_claim",
        family="prior_claim",
        tier="CORE",
        source_artifact=artifact,
        source_columns=(date_column, "claim__claim_date"),
        aggregation="recency",
        formula="claim_date - latest prior claim date",
    )
    if not prepared.empty:
        intervals = prepared.sort_values(
            ["current_warranty_claim_key", "_event_date", "prior_warranty_claim_key"],
            kind="mergesort",
        ).copy()
        intervals["_inter_claim_days"] = (
            intervals.groupby("current_warranty_claim_key")["_event_date"].diff().dt.days
        )
        mean = intervals.groupby("current_warranty_claim_key")["_inter_claim_days"].mean()
        std = intervals.groupby("current_warranty_claim_key")["_inter_claim_days"].agg(
            population_std
        )
    else:
        mean = pd.Series(dtype="Float64")
        std = pd.Series(dtype="Float64")
    _add_metric(
        registry,
        mean,
        "prior_claim__mean_inter_claim_days",
        family="prior_claim",
        tier="EXTENDED",
        source_artifact=artifact,
        source_columns=(date_column, "prior_warranty_claim_key"),
        aggregation="mean_interval",
        formula="mean days between prior claims",
        minimum_observations=2,
    )
    _add_metric(
        registry,
        std,
        "prior_claim__std_inter_claim_days",
        family="prior_claim",
        tier="EXTENDED",
        source_artifact=artifact,
        source_columns=(date_column, "prior_warranty_claim_key"),
        aggregation="std_interval",
        formula="population std of days between prior claims",
        minimum_observations=2,
        notes="Population standard deviation (ddof=0).",
    )


def _add_presence(
    registry: FeatureRegistry, claims: pd.DataFrame, prepared: dict[str, pd.DataFrame]
) -> None:
    for source_name, frame in prepared.items():
        name = f"history__has_{source_name}"
        if source_name == "component_installation":
            source_name = "component_installation"
        present = (
            frame.groupby("current_warranty_claim_key")
            .size()
            .reindex(registry.claim_keys)
            .fillna(0)
            .gt(0)
        )
        registry.add_model(
            present.astype(bool),
            name,
            family="history_coverage",
            tier="CORE",
            feature_type="boolean",
            source_artifacts=(f"{source_name}_history",),
            source_columns=("current_warranty_claim_key",),
            formula="safe historical row count > 0",
            notes="False means no qualifying history; the claim remains in the output.",
        )


def build_feature_matrix(
    frames: dict[str, pd.DataFrame],
    assignments: pd.DataFrame,
    settings: StructuredFeatureSettings | None = None,
) -> FeatureBuildResult:
    """Build one deterministic row per assigned claim without receiving the target."""

    settings = settings or StructuredFeatureSettings()
    if "claim_snapshot" not in frames:
        raise StructuredFeatureError("claim_snapshot is required for Phase 7 feature construction.")
    snapshot = frames["claim_snapshot"].copy()
    required = {CLAIM_KEY, "claim__claim_date"}
    missing = sorted(required - set(snapshot.columns))
    if missing:
        raise StructuredFeatureError("Snapshot is missing: " + ", ".join(missing))
    claims = snapshot.sort_values(CLAIM_KEY, kind="mergesort").drop_duplicates(CLAIM_KEY).copy()
    claims["claim__claim_date"] = pd.to_datetime(claims["claim__claim_date"], errors="coerce")
    if claims["claim__claim_date"].isna().any():
        raise StructuredFeatureError("Claim dates are required for target-independent features.")
    if assignments[CLAIM_KEY].duplicated().any():
        raise StructuredFeatureError("Phase 6 assignments contain duplicate claim keys.")
    assignment_view = assignments[[CLAIM_KEY, "split"]].copy()
    claims = claims.merge(assignment_view, on=CLAIM_KEY, how="inner", validate="one_to_one")
    if len(claims) != len(snapshot):
        raise StructuredFeatureError("Phase 7 assignments do not cover exactly one row per claim.")
    registry = FeatureRegistry(pd.Index(claims[CLAIM_KEY], name=CLAIM_KEY))
    registry.add_control(
        claims[CLAIM_KEY],
        CLAIM_KEY,
        feature_type="numeric",
        source_artifacts=("claim_snapshot",),
        source_columns=(CLAIM_KEY,),
        is_lineage=True,
        notes="Claim identity control; never a model feature.",
    )
    registry.add_control(
        claims["split"].astype("string"),
        "split",
        feature_type="categorical",
        source_artifacts=("split_assignments",),
        source_columns=("split",),
        is_lineage=True,
        notes="Frozen Phase 6 split control; never a model feature.",
    )
    for column in CONTROL_DATE_COLUMNS:
        if column in claims:
            registry.add_control(
                pd.to_datetime(claims[column], errors="coerce"),
                column,
                feature_type="date_control",
                source_artifacts=("claim_snapshot",),
                source_columns=(column,),
                notes="Raw date retained for lineage/control; derived numeric features feed models.",
            )
    _add_direct(registry, claims)
    _add_lifecycle(registry, claims)
    _add_usage_and_warranty(registry, claims)
    telemetry = frames.get("telemetry_history", pd.DataFrame())
    maintenance = frames.get("maintenance_history", pd.DataFrame())
    service = frames.get("service_history", pd.DataFrame())
    component = frames.get("component_installation_history", pd.DataFrame())
    prior = frames.get("prior_claim_history", pd.DataFrame())
    _add_telemetry(registry, claims, telemetry, settings)
    maintenance_ready = _add_event_family(
        registry,
        claims,
        maintenance,
        family="maintenance",
        artifact="maintenance_history",
        date_column="maintenance__maintenance_date",
        key_column="maintenance_event_key",
        type_column="maintenance__maintenance_type",
        odometer_column="maintenance__odometer_miles",
        engine_hours_column="maintenance__engine_hours",
        settings=settings,
        maintenance=True,
    )
    service_ready = _add_event_family(
        registry,
        claims,
        service,
        family="service",
        artifact="service_history",
        date_column="service__service_date",
        key_column="service_event_key",
        type_column="service__service_type",
        odometer_column="service__odometer_miles",
        engine_hours_column="service__engine_hours",
        settings=settings,
        maintenance=False,
    )
    _add_component(registry, claims, component, settings)
    _add_prior_claims(registry, claims, prior, settings)
    _add_presence(
        registry,
        claims,
        {
            "telemetry": add_claim_dates(
                telemetry, claims, "telemetry__month_start_date", telemetry=True
            ),
            "maintenance": maintenance_ready,
            "service": service_ready,
            "component_installation": add_claim_dates(
                component, claims, "component_installation__installed_date"
            ),
            "prior_claim": add_claim_dates(prior, claims, "prior_claim__claim_date"),
        },
    )
    output = registry.frame()
    if TARGET in output.columns:
        raise StructuredFeatureError("Target appeared in the Phase 7 feature matrix.")
    model_columns = [item.feature_name for item in registry.definitions if item.is_model_feature]
    if any(any(token in name.casefold() for token in RESTRICTED_TOKENS) for name in model_columns):
        raise StructuredFeatureError(
            "Restricted identifier appears in a Phase 7 model feature name."
        )
    if any(any(text in name.casefold() for text in TEXT_COLUMNS) for name in model_columns):
        raise StructuredFeatureError("Text-derived feature appears in the Phase 7 model matrix.")
    return FeatureBuildResult(frame=output, definitions=registry.definitions, warnings=[])
