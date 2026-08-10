"""Temporal, telemetry, maintenance, service, repair, and component audits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _violation(
    table: str,
    rule: str,
    mask: pd.Series,
    frame: pd.DataFrame,
    *,
    severity: str = "WARNING",
    classification: str = "provisional process check",
) -> dict[str, object]:
    count = int(mask.fillna(False).sum())
    dates: list[str] = []
    for column in frame.columns:
        if "date" in str(column).casefold() or str(column).casefold().endswith("_at"):
            values = pd.to_datetime(frame.loc[mask.fillna(False), column], errors="coerce").dropna()
            if not values.empty:
                dates.extend([values.min().isoformat(), values.max().isoformat()])
                break
    return {
        "table": table,
        "rule": rule,
        "violation_count": count,
        "violation_percentage": round(count / len(frame) * 100, 6) if len(frame) else 0.0,
        "example_date_range": dates[:2],
        "severity": severity if count else "INFO",
        "classification": classification,
    }


def temporal_rules(frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    """Run documented strong and provisional date-order checks."""

    rows: list[dict[str, object]] = []
    truck = frames.get("dbo.dim_truck")
    if truck is not None:
        for earlier, later, rule, severity, classification in (
            (
                "build_date",
                "delivery_date",
                "delivery_date_before_build_date",
                "ERROR",
                "strong logical check",
            ),
            (
                "build_date",
                "in_service_date",
                "in_service_date_before_build_date",
                "ERROR",
                "strong logical check",
            ),
        ):
            if earlier in truck and later in truck:
                left = pd.to_datetime(truck[earlier], errors="coerce")
                right = pd.to_datetime(truck[later], errors="coerce")
                rows.append(
                    _violation(
                        "dbo.dim_truck",
                        rule,
                        right < left,
                        truck,
                        severity=severity,
                        classification=classification,
                    )
                )

    claims = frames.get("dbo.fact_warranty_claim")
    if claims is not None:
        for earlier, later, rule in (
            ("repair_start_date", "repair_end_date", "repair_end_date_before_repair_start_date"),
            ("failure_date", "repair_start_date", "repair_start_date_before_failure_date"),
        ):
            if earlier in claims and later in claims:
                left = pd.to_datetime(claims[earlier], errors="coerce")
                right = pd.to_datetime(claims[later], errors="coerce")
                rows.append(
                    _violation(
                        "dbo.fact_warranty_claim",
                        rule,
                        right < left,
                        claims,
                        severity="ERROR" if "repair_end" in rule else "WARNING",
                        classification="strong logical check"
                        if "repair_end" in rule
                        else "expected process check requiring confirmation",
                    )
                )
        for column, rule in (
            ("days_to_failure", "negative_days_to_failure"),
            ("days_to_repair", "negative_days_to_repair"),
        ):
            if column in claims:
                values = pd.to_numeric(claims[column], errors="coerce")
                rows.append(
                    _violation(
                        "dbo.fact_warranty_claim",
                        rule,
                        values < 0,
                        claims,
                        severity="ERROR",
                        classification="strong logical check",
                    )
                )

    installation = frames.get("dbo.fact_component_installation")
    if (
        installation is not None
        and "installed_date" in installation
        and "failure_date" in installation
    ):
        installed = pd.to_datetime(installation["installed_date"], errors="coerce")
        failure = pd.to_datetime(installation["failure_date"], errors="coerce")
        rows.append(
            _violation(
                "dbo.fact_component_installation",
                "component_installed_after_failure",
                installed > failure,
                installation,
                severity="WARNING",
                classification="expected process check requiring confirmation",
            )
        )
    for table_name in ("dbo.fact_maintenance_event", "dbo.fact_service_event"):
        frame = frames.get(table_name)
        if frame is not None and "maintenance_date" in frame and "claim_date" in frame:
            rows.append(
                _violation(
                    table_name,
                    "maintenance_after_claim_date",
                    pd.to_datetime(frame["maintenance_date"], errors="coerce")
                    > pd.to_datetime(frame["claim_date"], errors="coerce"),
                    frame,
                    classification="informational temporal diagnostic",
                )
            )
        if frame is not None and "service_date" in frame and "claim_date" in frame:
            rows.append(
                _violation(
                    table_name,
                    "service_after_claim_date",
                    pd.to_datetime(frame["service_date"], errors="coerce")
                    > pd.to_datetime(frame["claim_date"], errors="coerce"),
                    frame,
                    classification="informational temporal diagnostic",
                )
            )
    telemetry = frames.get("dbo.fact_telemetry_monthly")
    if telemetry is not None and "month_start_date" in telemetry and "claim_date" in telemetry:
        rows.append(
            _violation(
                "dbo.fact_telemetry_monthly",
                "telemetry_after_claim_date",
                pd.to_datetime(telemetry["month_start_date"], errors="coerce")
                > pd.to_datetime(telemetry["claim_date"], errors="coerce"),
                telemetry,
                classification="informational temporal diagnostic",
            )
        )
    return rows


def telemetry_quality(frame: pd.DataFrame) -> dict[str, object]:
    """Profile telemetry sequences and flag logical/statistical issues."""

    if frame.empty:
        return {"available": True, "records": 0, "truck_count": 0, "issues": []}
    result: dict[str, Any] = {
        "available": True,
        "records": int(len(frame)),
        "truck_count": 0,
        "issues": [],
        "measurements": {},
    }
    truck_column = "truck_key" if "truck_key" in frame else None
    if truck_column:
        result["truck_count"] = int(frame[truck_column].nunique())
        ordered = frame.copy()
        ordered["_date"] = pd.to_datetime(ordered.get("month_start_date"), errors="coerce")
        ordered = ordered.sort_values([truck_column, "_date"])
        duplicate_month = ordered.duplicated([truck_column, "_date"], keep=False)
        result["duplicate_truck_month_records"] = int(duplicate_month.sum())
        missing_gaps = 0
        decreases = 0
        for _, group in ordered.groupby(truck_column, observed=True):
            dates = group["_date"].dropna().drop_duplicates().sort_values()
            if len(dates) > 1:
                gaps = dates.diff().dt.days.dropna()
                missing_gaps += int((gaps > 31).sum())
            if "total_odometer_miles" in group:
                decreases += int(group["total_odometer_miles"].diff().lt(0).sum())
        result["missing_month_gap_count"] = missing_gaps
        result["odometer_decrease_count"] = decreases
    for column in (
        "total_odometer_miles",
        "engine_hours_month",
        "idle_hours_month",
        "avg_engine_temp",
        "max_engine_temp",
        "avg_oil_pressure",
        "low_oil_pressure_events",
        "brake_air_pressure_alerts",
        "battery_voltage_alerts",
        "fault_code_count",
        "harsh_braking_events",
        "avg_payload_weight",
        "fuel_efficiency_mpg",
        "route_severity_score",
        "maintenance_compliance_score",
    ):
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        non_null = values.dropna()
        result["measurements"][column] = {
            "null_count": int(values.isna().sum()),
            "zero_count": int((non_null == 0).sum()),
            "negative_count": int((non_null < 0).sum()),
            "minimum": float(non_null.min()) if not non_null.empty else None,
            "maximum": float(non_null.max()) if not non_null.empty else None,
            "p01": float(non_null.quantile(0.01)) if not non_null.empty else None,
            "p99": float(non_null.quantile(0.99)) if not non_null.empty else None,
            "outlier_count_iqr": _iqr_outlier_count(non_null),
        }
    issues: list[dict[str, Any]] = result["issues"]
    if result.get("duplicate_truck_month_records", 0):
        issues.append(
            {
                "issue": "duplicate_truck_month_records",
                "severity": "ERROR",
                "count": result["duplicate_truck_month_records"],
            }
        )
    if result.get("missing_month_gap_count", 0):
        issues.append(
            {
                "issue": "missing_month_gaps",
                "severity": "WARNING",
                "count": result["missing_month_gap_count"],
            }
        )
    if result.get("odometer_decrease_count", 0):
        issues.append(
            {
                "issue": "odometer_decreases",
                "severity": "ERROR",
                "count": result["odometer_decrease_count"],
            }
        )
    if {"engine_hours_month", "idle_hours_month"}.issubset(frame.columns):
        engine_hours = pd.to_numeric(frame["engine_hours_month"], errors="coerce")
        idle_hours = pd.to_numeric(frame["idle_hours_month"], errors="coerce")
        comparable = engine_hours.notna() & idle_hours.notna()
        idle_over_engine = int((comparable & (idle_hours > engine_hours)).sum())
        result["idle_hours_over_engine_hours_count"] = idle_over_engine
        if idle_over_engine:
            issues.append(
                {
                    "issue": "idle_hours_exceed_engine_hours",
                    "field": "idle_hours_month",
                    "severity": "WARNING",
                    "count": idle_over_engine,
                }
            )
    for column, measurement in result["measurements"].items():
        if measurement["negative_count"]:
            issues.append(
                {
                    "issue": "negative_measurement",
                    "field": column,
                    "severity": "ERROR",
                    "count": measurement["negative_count"],
                }
            )
    return result


def _iqr_outlier_count(values: pd.Series) -> int:
    if len(values) < 4:
        return 0
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return 0
    return int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())


def maintenance_quality(frame: pd.DataFrame) -> dict[str, object]:
    """Summarize maintenance sequence and process consistency."""

    if frame.empty:
        return {"available": True, "records": 0}
    output: dict[str, object] = {
        "available": True,
        "records": int(len(frame)),
        "truck_count": int(frame["truck_key"].nunique()) if "truck_key" in frame else None,
        "date_range": _date_range(frame, "maintenance_date"),
        "duplicate_event_records": int(frame.duplicated(keep=False).sum()),
        "missing_technician_notes": int(frame["technician_notes"].isna().sum())
        if "technician_notes" in frame
        else None,
        "scheduled_rate_percentage": _flag_rate(frame, "scheduled_flag"),
        "on_time_rate_percentage": _flag_rate(frame, "completed_on_time_flag"),
        "overdue_days": _numeric_summary(frame, "overdue_days"),
        "maintenance_cost": _numeric_summary(frame, "maintenance_cost"),
        "logical_conflict_count": 0,
    }
    if {"completed_on_time_flag", "overdue_days"}.issubset(frame.columns):
        output["logical_conflict_count"] = int(
            (
                (pd.to_numeric(frame["completed_on_time_flag"], errors="coerce") == 1)
                & (pd.to_numeric(frame["overdue_days"], errors="coerce") > 0)
            ).sum()
        )
    return output


def service_repair_quality(
    service: pd.DataFrame | None, repairs: pd.DataFrame | None
) -> dict[str, object]:
    """Summarize service-event and repair-line coverage and arithmetic."""

    output: dict[str, object] = {"service": {}, "repair": {}}
    if service is not None:
        output["service"] = {
            "records": int(len(service)),
            "truck_count": int(service["truck_key"].nunique()) if "truck_key" in service else None,
            "service_center_count": int(service["service_center_key"].nunique())
            if "service_center_key" in service
            else None,
            "duplicate_event_records": int(service.duplicated(keep=False).sum()),
            "date_range": _date_range(service, "service_date"),
            "repeat_visit_rate_percentage": _flag_rate(service, "repeat_visit_flag"),
            "roadside_assistance_rate_percentage": _flag_rate(service, "roadside_assistance_flag"),
            "downtime_hours": _numeric_summary(service, "downtime_hours"),
            "complaint_text_coverage_percentage": _text_coverage(service, "complaint_description"),
            "diagnostic_summary_coverage_percentage": _text_coverage(service, "diagnostic_summary"),
        }
    if repairs is not None:
        output["repair"] = {
            "records": int(len(repairs)),
            "claim_count": int(repairs["warranty_claim_key"].nunique())
            if "warranty_claim_key" in repairs
            else None,
            "service_event_count": int(repairs["service_event_key"].nunique())
            if "service_event_key" in repairs
            else None,
            "duplicate_line_records": int(repairs.duplicated(keep=False).sum()),
            "part_quantity": _numeric_summary(repairs, "part_quantity"),
            "labor_hours": _numeric_summary(repairs, "labor_hours"),
            "labor_rate": _numeric_summary(repairs, "labor_rate"),
            "line_cost": _numeric_summary(repairs, "line_cost"),
            "missing_part_numbers": int(repairs["part_no"].isna().sum())
            if "part_no" in repairs
            else None,
            "missing_technician_ids": int(repairs["technician_id"].isna().sum())
            if "technician_id" in repairs
            else None,
        }
    return output


def component_supplier_quality(
    installations: pd.DataFrame | None,
    components: pd.DataFrame | None,
    suppliers: pd.DataFrame | None,
) -> dict[str, object]:
    """Summarize component genealogy and supplier concentration."""

    output: dict[str, object] = {"installation": {}, "component": {}, "supplier": {}}
    if installations is not None:
        output["installation"] = {
            "records": int(len(installations)),
            "truck_count": int(installations["truck_key"].nunique())
            if "truck_key" in installations
            else None,
            "component_count": int(installations["component_key"].nunique())
            if "component_key" in installations
            else None,
            "supplier_count": int(installations["supplier_key"].nunique())
            if "supplier_key" in installations
            else None,
            "lot_count": int(installations["component_lot_no"].nunique())
            if "component_lot_no" in installations
            else None,
            "rework_rate_percentage": _flag_rate(installations, "rework_flag"),
            "quality_check_status_distribution": _value_counts(
                installations, "quality_check_status"
            ),
            "inspection_score": _numeric_summary(installations, "inspection_score"),
            "torque_value_missing_count": int(installations["torque_value"].isna().sum())
            if "torque_value" in installations
            else None,
            "production_batch_count": int(installations["production_batch_id"].nunique())
            if "production_batch_id" in installations
            else None,
            "station_count": int(installations["station_id"].nunique())
            if "station_id" in installations
            else None,
            "inspector_count": int(installations["inspector_id"].nunique())
            if "inspector_id" in installations
            else None,
        }
    if components is not None:
        output["component"] = {
            "records": int(len(components)),
            "safety_critical_rate_percentage": _flag_rate(components, "is_safety_critical"),
            "system_distribution": _value_counts(components, "component_system"),
            "category_distribution": _value_counts(components, "component_category"),
        }
    if suppliers is not None:
        output["supplier"] = {
            "records": int(len(suppliers)),
            "quality_rating": _numeric_summary(suppliers, "quality_rating"),
            "tier_distribution": _value_counts(suppliers, "supplier_tier"),
            "preferred_rate_percentage": _flag_rate(suppliers, "preferred_supplier_flag"),
        }
    return output


def _date_range(frame: pd.DataFrame, column: str) -> dict[str, object] | None:
    if column not in frame:
        return None
    values = pd.to_datetime(frame[column], errors="coerce").dropna()
    return (
        {"min": values.min().isoformat(), "max": values.max().isoformat()}
        if not values.empty
        else None
    )


def _numeric_summary(frame: pd.DataFrame, column: str) -> dict[str, object] | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return {
        "records": int(len(values)),
        "min": float(values.min()) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
        "median": float(values.median()) if len(values) else None,
    }


def _flag_rate(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return round(float(values.mean() * 100), 6) if len(values) else None


def _text_coverage(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = frame[column].astype("string")
    return round(float((values.notna() & values.str.strip().ne("")).mean() * 100), 6)


def _value_counts(frame: pd.DataFrame, column: str) -> list[dict[str, object]]:
    if column not in frame:
        return []
    counts = frame[column].value_counts(dropna=False).head(20)
    return [
        {"category": "<missing>" if pd.isna(value) else value, "records": int(count)}
        for value, count in counts.items()
    ]
