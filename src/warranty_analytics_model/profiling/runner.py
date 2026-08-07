"""Orchestration for offline fixtures and live Phase 3 profiling."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Settings
from ..database.connection import check_database_connection
from ..database.metadata import collect_schema_metadata
from ..database.schema_contract import load_schema_contract
from ..database.schema_validator import validate_schema
from .association import association_table, missingness_by_target
from .category_sparsity import category_sparsity
from .config import ProfilingSettings
from .extractor import LiveProfileExtractor
from .findings import Finding, FindingSeverity, finding_counts, make_finding, overall_status
from .relational_quality import cost_arithmetic_audit, foreign_key_orphans, missingness_summary
from .reporting import write_phase3_reports
from .synthetic_audit import POST_OUTCOME_COLUMNS, run_synthetic_audit
from .table_profile import profile_tables
from .target_profile import profile_target
from .temporal_quality import (
    component_supplier_quality,
    maintenance_quality,
    service_repair_quality,
    telemetry_quality,
    temporal_rules,
)

TARGET_TABLE = "dbo.fact_warranty_claim"
APPROVED_TABLE_COUNT = 16


def _claim_context(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Join one-to-one dimensions to claims for aggregate diagnostics only."""

    claims = frames.get(TARGET_TABLE, pd.DataFrame()).copy()
    if claims.empty:
        return claims

    def merge_dimension(
        current: pd.DataFrame,
        table_name: str,
        key: str,
        fields: list[str],
        *,
        rename_prefix: str = "",
    ) -> pd.DataFrame:
        dimension = frames.get(table_name)
        if dimension is None or key not in current or key not in dimension:
            return current
        selected = [key] + [field for field in fields if field in dimension.columns]
        selected = list(dict.fromkeys(selected))
        right = dimension[selected].drop_duplicates(key)
        if rename_prefix:
            right = right.rename(
                columns={field: f"{rename_prefix}{field}" for field in selected if field != key}
            )
        else:
            right = right.rename(
                columns={
                    field: field
                    for field in selected
                    if field not in current.columns or field == key
                }
            )
            right = right[
                [
                    column
                    for column in right.columns
                    if column == key or column not in current.columns
                ]
            ]
        return current.merge(right, on=key, how="left", validate="many_to_one")

    claims = merge_dimension(
        claims,
        "dbo.dim_truck",
        "truck_key",
        [
            "truck_model_key",
            "customer_key",
            "manufacturing_plant",
            "assembly_line",
            "production_batch_id",
            "build_date",
            "delivery_date",
            "in_service_date",
        ],
    )
    claims = merge_dimension(
        claims,
        "dbo.dim_truck_model",
        "truck_model_key",
        ["model_name", "model_year", "brand", "segment", "application_type"],
    )
    claims = merge_dimension(
        claims,
        "dbo.dim_service_center",
        "service_center_key",
        ["location_key", "dealer_group", "certified_level", "service_capacity"],
    )
    claims = merge_dimension(
        claims,
        "dbo.dim_component",
        "causal_component_key",
        [
            "component_id",
            "component_system",
            "component_category",
            "supplier_key",
            "is_safety_critical",
        ],
    )
    claims = merge_dimension(
        claims,
        "dbo.dim_failure_code",
        "failure_code_key",
        [
            "failure_code",
            "failure_system",
            "failure_category",
            "severity_level",
            "safety_related_flag",
        ],
    )
    claims = merge_dimension(
        claims,
        "dbo.dim_supplier",
        "supplier_key",
        ["supplier_id", "supplier_region", "supplier_tier", "quality_rating"],
    )
    claims = merge_dimension(
        claims,
        "dbo.dim_customer",
        "customer_key",
        [
            "customer_type",
            "industry",
            "fleet_size",
            "contract_type",
            "account_priority",
            "location_key",
        ],
        rename_prefix="customer_",
    )
    claims = merge_dimension(
        claims,
        "dbo.dim_location",
        "location_key",
        ["region", "climate_zone", "terrain_type"],
    )
    return claims


def _category_columns(frame: pd.DataFrame) -> list[str]:
    candidates = (
        "model_name",
        "truck_model_key",
        "model_year",
        "manufacturing_plant",
        "assembly_line",
        "production_batch_id",
        "component_system",
        "component_category",
        "component_lot_no",
        "supplier_key",
        "service_center_key",
        "region",
        "climate_zone",
        "terrain_type",
        "claim_type",
        "failure_category",
    )
    return [column for column in candidates if column in frame]


def _finding_from_table_profile(profile: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    table = str(profile["table"])
    row_count = int(profile["row_count"])
    duplicate_keys = int(profile.get("duplicate_primary_key_value_count", 0))
    if duplicate_keys:
        findings.append(
            make_finding(
                "DUPLICATE_PRIMARY_KEY",
                "ERROR",
                "relational",
                "Duplicate primary-key values were observed.",
                table=table,
                affected_rows=int(profile.get("duplicate_primary_key_record_count", 0)),
                affected_percentage=float(profile.get("duplicate_primary_key_record_count", 0))
                / row_count
                * 100
                if row_count
                else 0.0,
                evidence={"duplicate_primary_key_values": duplicate_keys},
                modeling_impact="Duplicate records can duplicate labels and violate the claim/unit contract.",
                recommendation="Resolve key validity with the data owner before eligibility rules.",
                required_resolution_phase="Phase 4",
                blocking_for_next_phase=True,
            )
        )
    full_duplicates = int(profile.get("full_duplicate_row_count", 0))
    if full_duplicates:
        findings.append(
            make_finding(
                "DUPLICATE_FULL_ROWS",
                "WARNING",
                "duplicates",
                "Fully duplicated rows were observed.",
                table=table,
                affected_rows=full_duplicates,
                affected_percentage=full_duplicates / row_count * 100 if row_count else 0.0,
                evidence={"full_duplicate_row_count": full_duplicates},
                modeling_impact="Random splits may place identical records on both sides.",
                recommendation="Retain records for now; use duplicate-aware Phase 6 splits.",
                required_resolution_phase="Phase 6",
            )
        )
    for column, column_profile in (profile.get("column_profiles", {}) or {}).items():
        if isinstance(column_profile, dict):
            null_percentage = float(column_profile.get("null_percentage", 0.0))
            if null_percentage >= 50.0:
                findings.append(
                    make_finding(
                        "HIGH_MISSINGNESS",
                        "WARNING",
                        "missingness",
                        "The field is missing for at least half of its rows.",
                        table=table,
                        columns=[str(column)],
                        affected_rows=int(column_profile.get("null_count", 0)),
                        affected_percentage=null_percentage,
                        evidence={"null_percentage": null_percentage},
                        modeling_impact="Missingness may reduce usable history or encode process timing.",
                        recommendation="Confirm capture timing and missingness policy; do not impute in Phase 3.",
                        required_resolution_phase="Phase 4",
                    )
                )
    return findings


def _quality_findings(
    table_profiles: list[dict[str, Any]],
    target: dict[str, Any],
    fk_rows: list[dict[str, Any]],
    temporal_rows: list[dict[str, Any]],
    telemetry: dict[str, Any],
    maintenance: dict[str, Any],
    synthetic: dict[str, Any],
    missingness: list[dict[str, Any]],
    missingness_target: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
) -> list[Finding]:
    findings: list[Finding] = []
    for profile in table_profiles:
        findings.extend(_finding_from_table_profile(profile))
    for row in fk_rows:
        if int(row.get("orphan_count", 0)):
            findings.append(
                make_finding(
                    "FOREIGN_KEY_ORPHAN",
                    "ERROR",
                    "referential_integrity",
                    "Declared foreign-key values do not all resolve to the referenced table.",
                    table=str(row["parent_table"]),
                    columns=[str(row["foreign_key"])],
                    affected_rows=int(row["orphan_count"]),
                    affected_percentage=float(row["orphan_percentage"]),
                    evidence={
                        "referenced_table": row["referenced_table"],
                        "rows_checked": row["rows_checked"],
                    },
                    modeling_impact="Joins can silently lose or multiply claim history.",
                    recommendation="Resolve orphan keys or document an approved business exception.",
                    required_resolution_phase="Phase 4",
                    blocking_for_next_phase=True,
                )
            )
    if int(target.get("null_target_count", 0)):
        findings.append(
            make_finding(
                "NULL_REQUIRED_TARGET",
                "ERROR",
                "target",
                "The required target is null for some claim rows.",
                table=TARGET_TABLE,
                columns=["high_cost_claim_flag"],
                affected_rows=int(target["null_target_count"]),
                affected_percentage=float(target["null_target_count"])
                / int(target.get("claims", 1))
                * 100,
                evidence={"null_target_count": target["null_target_count"]},
                modeling_impact="Supervised eligibility is undefined for null labels.",
                recommendation="Obtain an explicit target eligibility policy before Phase 4.",
                required_resolution_phase="Phase 4",
                blocking_for_next_phase=True,
            )
        )
    if int(target.get("invalid_target_count", 0)):
        findings.append(
            make_finding(
                "INVALID_TARGET_VALUE",
                "ERROR",
                "target",
                "Target values outside the documented binary domain were observed.",
                table=TARGET_TABLE,
                columns=["high_cost_claim_flag"],
                affected_rows=int(target["invalid_target_count"]),
                evidence={"allowed_values": [0, 1]},
                modeling_impact="Invalid labels cannot be used in supervised evaluation.",
                recommendation="Resolve target domain validity with the data owner.",
                required_resolution_phase="Phase 4",
                blocking_for_next_phase=True,
            )
        )
    if str(target.get("class_balance")) == "imbalanced":
        findings.append(
            make_finding(
                "TARGET_IMBALANCE",
                "WARNING",
                "target",
                "The target distribution is not approximately balanced.",
                table=TARGET_TABLE,
                columns=["high_cost_claim_flag"],
                affected_rows=int(target.get("positive_claims", 0)),
                affected_percentage=float(target.get("positive_percentage", 0.0)),
                evidence={
                    "positive_percentage": target.get("positive_percentage"),
                    "negative_percentage": target.get("negative_percentage"),
                },
                modeling_impact="Later evaluation and training may need imbalance-aware baselines and metrics.",
                recommendation="Plan imbalance-aware Phase 5 evaluation after the target and snapshot are approved.",
                required_resolution_phase="Phase 5",
            )
        )
    for row in temporal_rows:
        count = int(row.get("violation_count", 0))
        if count:
            severity: FindingSeverity = "ERROR" if row.get("severity") == "ERROR" else "WARNING"
            findings.append(
                make_finding(
                    "TEMPORAL_ORDER_VIOLATION",
                    severity,
                    "temporal",
                    "A documented temporal ordering diagnostic found violations.",
                    table=str(row.get("table")),
                    affected_rows=count,
                    affected_percentage=float(row.get("violation_percentage", 0.0)),
                    evidence={"rule": row.get("rule"), "classification": row.get("classification")},
                    modeling_impact="Temporal inconsistencies can invalidate as-of histories and splits.",
                    recommendation="Confirm rule strength; resolve strong logical violations before Phase 4.",
                    required_resolution_phase="Phase 4",
                    blocking_for_next_phase=severity == "ERROR",
                )
            )
    for issue in telemetry.get("issues", []) if isinstance(telemetry.get("issues"), list) else []:
        if isinstance(issue, dict):
            findings.append(
                make_finding(
                    "TELEMETRY_SEQUENCE_ISSUE",
                    "ERROR" if issue.get("severity") == "ERROR" else "WARNING",
                    "telemetry",
                    "Telemetry sequence diagnostics found a logical or coverage issue.",
                    table="dbo.fact_telemetry_monthly",
                    affected_rows=int(issue.get("count", 0)),
                    evidence={"issue": issue.get("issue"), "field": issue.get("field")},
                    modeling_impact="Telemetry histories may be unreliable for as-of aggregates.",
                    recommendation="Review sequence completeness and sign constraints with the data owner.",
                    required_resolution_phase="Phase 4",
                    blocking_for_next_phase=issue.get("severity") == "ERROR",
                )
            )
    if int(maintenance.get("logical_conflict_count", 0)):
        findings.append(
            make_finding(
                "MAINTENANCE_PROCESS_CONFLICT",
                "WARNING",
                "maintenance",
                "On-time maintenance rows have materially positive overdue days.",
                table="dbo.fact_maintenance_event",
                affected_rows=int(maintenance["logical_conflict_count"]),
                evidence={"conflict": "completed_on_time_flag=1 and overdue_days>0"},
                modeling_impact="The field relationship may be synthetic or business-process inconsistent.",
                recommendation="Confirm semantics; do not silently correct values.",
                required_resolution_phase="Phase 4",
            )
        )
    identifier = synthetic.get("identifier_audit", {})
    if isinstance(identifier, dict) and "SYNTHETIC_IDENTIFIER_LEAKAGE" in identifier.get(
        "flags", []
    ):
        findings.append(
            make_finding(
                "SYNTHETIC_IDENTIFIER_LEAKAGE",
                "WARNING",
                "synthetic_leakage",
                "Supported identifier prefix/suffix groups are target-pure.",
                table=TARGET_TABLE,
                evidence={"flag": "SYNTHETIC_IDENTIFIER_LEAKAGE"},
                modeling_impact="Identifiers may encode the generator rather than prediction-time risk.",
                recommendation="Exclude identifiers and evaluate unseen groups.",
                required_resolution_phase="Phase 4",
            )
        )
    text = synthetic.get("text_audit", {})
    if isinstance(text, dict) and "SYNTHETIC_TEXT_TEMPLATE_LEAKAGE" in text.get("flags", []):
        findings.append(
            make_finding(
                "SYNTHETIC_TEXT_TEMPLATE_LEAKAGE",
                "WARNING",
                "synthetic_leakage",
                "Repeated normalized text templates strongly align with target values.",
                modeling_impact="Text can reveal generated outcomes and inflate later model performance.",
                recommendation="Exclude or sanitize outcome-bearing text pending availability review.",
                required_resolution_phase="Phase 4",
            )
        )
    group_purity = synthetic.get("group_purity", [])
    pure_count = (
        sum(
            1
            for row in group_purity
            if isinstance(row, dict) and row.get("target_pure") and row.get("meaningful_support")
        )
        if isinstance(group_purity, list)
        else 0
    )
    if pure_count:
        findings.append(
            make_finding(
                "SUPPORTED_TARGET_PURE_GROUP",
                "WARNING",
                "group_purity",
                "At least one supported high-cardinality group is target-pure.",
                table=TARGET_TABLE,
                affected_rows=pure_count,
                evidence={"pure_group_count": pure_count},
                modeling_impact="Random splits may reward memorization of batch, lot, supplier, or component groups.",
                recommendation="Use group-aware Phase 6 evaluation and report support.",
                required_resolution_phase="Phase 6",
            )
        )
    duplicate = synthetic.get("duplicate_audit", {})
    if isinstance(duplicate, dict) and duplicate.get("duplicates_found"):
        findings.append(
            make_finding(
                "DUPLICATE_SCENARIO_FAMILY",
                "WARNING",
                "duplicates",
                "Repeated deterministic scenario fingerprints were observed.",
                evidence={"recommendation": "Use duplicate-aware Phase 6 splits."},
                modeling_impact="Identical or near-identical scenarios can contaminate random train/test splits.",
                recommendation="Retain records and hold out fingerprint families where appropriate.",
                required_resolution_phase="Phase 6",
            )
        )
    for row in missingness:
        if row.get("table") == TARGET_TABLE and float(row.get("null_percentage", 0.0)) >= 50:
            findings.append(
                make_finding(
                    "HIGH_CLAIM_MISSINGNESS",
                    "WARNING",
                    "missingness",
                    "A claim field has substantial missingness.",
                    table=TARGET_TABLE,
                    columns=[str(row["field"])],
                    affected_rows=int(row["null_count"]),
                    affected_percentage=float(row["null_percentage"]),
                    evidence={"null_percentage": row["null_percentage"]},
                    modeling_impact="Claim-time completeness may vary by process stage.",
                    recommendation="Confirm capture timing and missingness policy.",
                    required_resolution_phase="Phase 4",
                )
            )
    for row in missingness_target:
        if row.get("suspected_leakage"):
            findings.append(
                make_finding(
                    "SYNTHETIC_MISSINGNESS_LEAKAGE",
                    "WARNING",
                    "synthetic_leakage",
                    "Missingness rates differ sharply between target classes.",
                    table=TARGET_TABLE,
                    columns=[str(row.get("field"))],
                    evidence={"missing_rate_by_target": row.get("missing_rate_by_target")},
                    modeling_impact="Missingness may encode the target-generation process rather than a claim-time signal.",
                    recommendation="Confirm capture timing before adding missingness indicators in Phase 4.",
                    required_resolution_phase="Phase 4",
                )
            )
    sparse = sum(
        1
        for row in category_rows
        if isinstance(row, dict)
        and any(
            int(value) > 0 for value in (row.get("categories_below_threshold", {}) or {}).values()
        )
    )
    if sparse:
        findings.append(
            make_finding(
                "SPARSE_CATEGORIES",
                "WARNING",
                "category_sparsity",
                "One or more candidate categorical fields contain low-support categories.",
                table=TARGET_TABLE,
                affected_rows=sparse,
                evidence={"fields_with_sparse_categories": sparse},
                modeling_impact="Later categorical encodings may overfit rare groups.",
                recommendation="Plan grouped/unseen-category evaluation after business approval; do not collapse categories in Phase 3.",
                required_resolution_phase="Phase 4",
            )
        )
    return findings


def profile_dataframes(
    frames: Mapping[str, pd.DataFrame],
    *,
    contract: Any | None = None,
    profiling_settings: ProfilingSettings | None = None,
    exact_row_counts: Mapping[str, int] | None = None,
    live_database: dict[str, object] | None = None,
    write_reports: bool = False,
    output_dir: Path | None = None,
    report_formats: tuple[str, ...] = ("json", "markdown"),
) -> dict[str, Any]:
    """Run Phase 3 against in-memory DataFrames, suitable for fixtures and live extracts."""

    config = profiling_settings or ProfilingSettings()
    normalized_frames = {str(name): frame.copy() for name, frame in frames.items()}
    table_profiles = profile_tables(
        normalized_frames,
        contract=contract,
        top_categories=config.top_categories,
        percentiles=tuple(config.percentiles),
        rare_category_thresholds=tuple(config.rare_category_thresholds),
    )
    context = _claim_context(normalized_frames)
    target = profile_target(context)
    fk_rows = foreign_key_orphans(normalized_frames, contract) if contract is not None else []
    temporal_rows = temporal_rules(dict(normalized_frames)) if config.enable_temporal_audit else []
    telemetry = telemetry_quality(
        normalized_frames.get("dbo.fact_telemetry_monthly", pd.DataFrame())
    )
    maintenance = maintenance_quality(
        normalized_frames.get("dbo.fact_maintenance_event", pd.DataFrame())
    )
    service_repair = service_repair_quality(
        normalized_frames.get("dbo.fact_service_event"),
        normalized_frames.get("dbo.fact_repair_line"),
    )
    component_supplier = component_supplier_quality(
        normalized_frames.get("dbo.fact_component_installation"),
        normalized_frames.get("dbo.dim_component"),
        normalized_frames.get("dbo.dim_supplier"),
    )
    missingness = missingness_summary(normalized_frames)
    missingness_target = missingness_by_target(context, columns=context.columns)
    category_rows = category_sparsity(
        context,
        _category_columns(context),
        target_column="high_cost_claim_flag",
        thresholds=config.rare_category_thresholds,
    )
    synthetic = run_synthetic_audit(
        normalized_frames,
        context,
        enable_text=config.enable_text_audit,
        enable_identifiers=config.enable_identifier_audit,
    )
    leakage = synthetic.get("leakage_diagnostics", {})
    associations = association_table(
        context,
        columns=[column for column in context.columns if column not in POST_OUTCOME_COLUMNS],
        leakage_columns=POST_OUTCOME_COLUMNS,
    )
    quality = {
        "foreign_key_orphans": fk_rows,
        "temporal_violations": temporal_rows,
        "telemetry": telemetry,
        "maintenance": maintenance,
        "service_repair": service_repair,
        "component_supplier": component_supplier,
        "cost_arithmetic": cost_arithmetic_audit(
            normalized_frames.get("dbo.fact_repair_line", pd.DataFrame())
        ),
    }
    findings = _quality_findings(
        table_profiles,
        target,
        fk_rows,
        temporal_rows,
        telemetry,
        maintenance,
        synthetic,
        missingness,
        missingness_target,
        category_rows,
    )
    if contract is not None:
        missing_tables = sorted(set(getattr(contract, "table_map", {})) - set(normalized_frames))
        for table in missing_tables:
            findings.append(
                make_finding(
                    "MISSING_APPROVED_TABLE",
                    "ERROR",
                    "scope",
                    "An approved contract table was not available to profile.",
                    table=table,
                    evidence={"approved_table_count": APPROVED_TABLE_COUNT},
                    modeling_impact="The Phase 3 inventory is incomplete.",
                    recommendation="Confirm extraction scope and rerun all approved tables.",
                    required_resolution_phase="Phase 3",
                    blocking_for_next_phase=True,
                )
            )
    counts = finding_counts(findings)
    result: dict[str, Any] = {
        "phase": "Phase 3 — Data Profiling and Synthetic Data Audit",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": overall_status(findings),
        "live_database": live_database or {"executed": False},
        "included_table_count": len(normalized_frames),
        "approved_table_count": len(getattr(contract, "tables", []))
        if contract is not None
        else len(normalized_frames),
        "exact_row_counts": dict(
            exact_row_counts or {name: int(len(frame)) for name, frame in normalized_frames.items()}
        ),
        "table_profiles": table_profiles,
        "target_profile": target,
        "data_quality": quality,
        "missingness": missingness,
        "missingness_by_target": missingness_target,
        "category_sparsity": category_rows,
        "associations": associations,
        "synthetic_data_audit": synthetic,
        "leakage_diagnostics": leakage,
        "findings": [finding.model_dump(mode="json") for finding in findings],
        "finding_counts": counts,
        "excluded_tables": list(getattr(contract, "excluded_tables", []))
        if contract is not None
        else [],
        "scope_confirmations": {
            "database_writes": False,
            "excluded_ml_tables_read": False,
            "production_features_engineered": False,
            "train_test_split_implemented": False,
            "predictive_model_trained": False,
            "raw_sensitive_values_reported": False,
            "phase_0_questions_resolved_by_assumption": False,
        },
    }
    if write_reports:
        if output_dir is None:
            raise ValueError("output_dir is required when write_reports=True")
        result["report_directory"] = str(
            write_phase3_reports(result, output_dir, formats=report_formats)
        )
    return result


def run_live_phase3(
    settings: Settings,
    *,
    profiling_settings: ProfilingSettings | None = None,
    output_dir: Path | None = None,
    report_formats: tuple[str, ...] = ("json", "markdown"),
    no_charts: bool = True,
) -> dict[str, Any]:
    """Validate Phase 2, read only the approved tables, and write Phase 3 reports."""

    config = profiling_settings or ProfilingSettings()
    contract, checksum = load_schema_contract()
    connectivity = check_database_connection(settings.database)
    live = collect_schema_metadata(settings.database, contract)
    validation = validate_schema(
        contract, live, checksum, environment=settings.environment, server=settings.database.server
    )
    if validation.status != "passed":
        raise RuntimeError(f"Schema validation failed with {validation.error_count} error(s).")
    extractor = LiveProfileExtractor(settings.database, contract, chunk_size=config.chunk_size)
    frames, exact_counts = extractor.extract_all()
    report_root = output_dir or config.resolved_output_directory()
    result = profile_dataframes(
        frames,
        contract=contract,
        profiling_settings=config,
        exact_row_counts=exact_counts,
        live_database={
            "executed": True,
            "database": connectivity.actual_database,
            "catalog_readable": connectivity.catalog_readable,
            "schema_validation": validation.status,
            "schema_error_count": validation.error_count,
            "schema_warning_count": validation.warning_count,
            "excluded_objects_reported_by_name": live.excluded_objects,
        },
        write_reports=True,
        output_dir=report_root,
        report_formats=report_formats,
    )
    result["no_charts"] = no_charts
    return result
