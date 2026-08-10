"""Phase 5 orchestration for offline planning and live read-only builds."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Settings
from ..database.connection import check_database_connection
from ..database.metadata import collect_schema_metadata
from ..database.schema_contract import load_schema_contract
from ..database.schema_validator import validate_schema
from ..paths import discover_repository_root
from ..policy.loader import load_phase4_contracts
from ..policy.target_contract import claim_eligibility_mask
from .component_history import build_component_installation_history
from .config import (
    load_feature_mart_settings,
    resolve_mart_output_root,
    resolve_mart_report_root,
)
from .direct_snapshot import build_direct_snapshot
from .extraction_plan import build_extraction_plan
from .extractor import LiveFeatureMartExtractor
from .lineage import build_group_membership
from .maintenance_history import build_maintenance_history
from .manifest import build_column_manifest, build_manifest, write_json, write_parquet
from .mart_contract import (
    assert_mart_contract_valid,
    load_mart_contract,
    validate_mart_contract,
)
from .models import FeatureMartError, FeatureMartSettings, Phase5BuildResult
from .prior_claim_history import build_prior_claim_history
from .repair_history import build_repair_history_index
from .reporting import write_phase5_reports
from .service_history import build_service_history
from .telemetry_history import build_telemetry_history
from .validation import validate_frames, validate_mart_directory


def phase5_run_id() -> str:
    """Create a readable UTC run identifier."""

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _artifact_paths() -> dict[str, str]:
    return {
        "claim_snapshot": "claim_snapshot.parquet",
        "telemetry_history": "history/telemetry_history.parquet",
        "maintenance_history": "history/maintenance_history.parquet",
        "service_history": "history/service_history.parquet",
        "component_installation_history": "history/component_installation_history.parquet",
        "prior_claim_history": "history/prior_claim_history.parquet",
        "repair_history_index": "history/repair_history_index.parquet",
        "claim_group_membership": "lineage/claim_group_membership.parquet",
    }


def _prepare_build_frames(
    frames: dict[str, pd.DataFrame],
    contract: Any,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, Any],
    dict[str, dict[str, int | float]],
    dict[str, Any],
]:
    """Construct all snapshot/bridge frames in deterministic dependency order."""

    direct = build_direct_snapshot(frames, contract)
    eligible_claims = direct.eligible_claims
    telemetry, telemetry_coverage = build_telemetry_history(
        eligible_claims,
        frames["dbo.fact_telemetry_monthly"],
    )
    maintenance, maintenance_coverage = build_maintenance_history(
        eligible_claims,
        frames["dbo.fact_maintenance_event"],
    )
    service, service_coverage = build_service_history(
        eligible_claims,
        frames["dbo.fact_service_event"],
    )
    component, component_coverage, component_join = build_component_installation_history(
        eligible_claims,
        frames["dbo.fact_component_installation"],
        frames["dbo.dim_component"],
    )
    prior, prior_coverage, prior_join = build_prior_claim_history(
        eligible_claims,
        frames["dbo.fact_warranty_claim"],
        frames["dbo.dim_failure_code"],
    )
    repair, repair_coverage = build_repair_history_index(
        eligible_claims,
        frames["dbo.fact_warranty_claim"],
        frames["dbo.fact_repair_line"],
    )
    groups = build_group_membership(direct.snapshot, component, contract)
    artifact_frames = {
        "claim_snapshot": direct.snapshot,
        "telemetry_history": telemetry,
        "maintenance_history": maintenance,
        "service_history": service,
        "component_installation_history": component,
        "prior_claim_history": prior,
        "repair_history_index": repair,
        "claim_group_membership": groups,
    }
    coverage = {
        "telemetry": telemetry_coverage,
        "maintenance": maintenance_coverage,
        "service": service_coverage,
        "component_installation": component_coverage,
        "prior_claim": prior_coverage,
        "repair": repair_coverage,
    }
    joins: dict[str, Any] = dict(direct.join_validation)
    joins["component_to_dimension"] = component_join
    joins["prior_claim_to_failure"] = prior_join
    return artifact_frames, direct.eligibility, coverage, joins


def _write_run_artifacts(
    *,
    run_dir: Path,
    artifact_frames: dict[str, pd.DataFrame],
    settings: FeatureMartSettings,
    contract: Any,
    mart_checksum: str,
    root: Path,
    environment: str,
    source_database: str,
    source_row_counts: dict[str, int],
    eligibility: dict[str, Any],
    history_coverage: dict[str, dict[str, int | float]],
    joins: dict[str, Any],
    source_target_values: pd.Series,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write data and manifests into a temporary run directory."""

    paths = _artifact_paths()
    metadata: dict[str, dict[str, str | int]] = {}
    for artifact, relative_path in paths.items():
        path = run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata[artifact] = write_parquet(
            artifact_frames[artifact],
            path,
            compression=settings.compression,
        )
    column_manifest, field_lineage = build_column_manifest(
        contract,
        artifact_frames,
        paths,
    )
    write_json(run_dir / "column_manifest.json", column_manifest)
    write_json(run_dir / "field_lineage.json", field_lineage)
    manifest = build_manifest(
        root=root,
        contract=contract,
        mart_checksum=mart_checksum,
        environment=environment,
        source_database=source_database,
        source_row_counts=source_row_counts,
        eligibility=eligibility,
        frames=artifact_frames,
        artifact_paths=paths,
        artifact_metadata=metadata,
        bridge_row_counts={
            artifact: int(len(frame))
            for artifact, frame in artifact_frames.items()
            if artifact != "claim_snapshot"
        },
        history_coverage=history_coverage,
        join_validation=joins,
    )
    write_json(run_dir / "manifest.json", manifest)
    validation = validate_frames(
        frames=artifact_frames,
        eligibility=eligibility,
        contract=contract,
        phase4_bundle=load_phase4_contracts(root),
        column_manifest=column_manifest,
        history_coverage=history_coverage,
        direct_join_validation=joins,
        source_target_values=source_target_values,
    )
    manifest["validation_status"] = validation["status"]
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "validation.json", validation)
    return manifest, validation


def build_feature_mart_from_frames(
    *,
    frames: dict[str, pd.DataFrame],
    source_row_counts: dict[str, int],
    root: Path,
    settings: FeatureMartSettings,
    environment: str,
    source_database: str,
    output_root: Path,
    report_root: Path | None = None,
    no_report: bool = False,
    overwrite: bool = False,
    run_id: str | None = None,
) -> Phase5BuildResult:
    """Build a local mart atomically from already extracted source frames."""

    schema_contract, schema_checksum = load_schema_contract(root)
    phase4_bundle = load_phase4_contracts(root)
    contract, contract_checksum = load_mart_contract(root)
    plan = validate_mart_contract(
        schema_contract,
        phase4_bundle,
        schema_contract_checksum=schema_checksum,
        contract=contract,
        contract_checksum=contract_checksum,
    )
    assert_mart_contract_valid(plan)
    if not settings.serialization_format == "parquet":
        raise FeatureMartError("Phase 5 currently supports only Parquet serialization.")
    artifact_frames, eligibility, coverage, joins = _prepare_build_frames(frames, contract)
    eligible_claims = frames["dbo.fact_warranty_claim"].copy()
    eligible_claims["claim_date"] = pd.to_datetime(eligible_claims["claim_date"], errors="coerce")
    truck_keys = frames["dbo.dim_truck"][["truck_key"]]
    eligible_mask = claim_eligibility_mask(
        eligible_claims[["warranty_claim_key", "claim_date", "high_cost_claim_flag", "truck_key"]],
        truck_keys,
    )
    eligible_source = eligible_claims.loc[eligible_mask].set_index("warranty_claim_key")
    source_target_values = pd.to_numeric(
        eligible_source.loc[
            artifact_frames["claim_snapshot"]["warranty_claim_key"],
            "high_cost_claim_flag",
        ],
        errors="coerce",
    ).reset_index(drop=True)
    run_id = run_id or phase5_run_id()
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / run_id
    if final_dir.exists() and not overwrite:
        raise FeatureMartError(f"Completed mart directory already exists: {final_dir}")
    temporary_dir = output_root / f".phase5-{run_id}-{uuid.uuid4().hex}"
    temporary_dir.mkdir(parents=True, exist_ok=False)
    try:
        manifest, validation = _write_run_artifacts(
            run_dir=temporary_dir,
            artifact_frames=artifact_frames,
            settings=settings,
            contract=contract,
            mart_checksum=contract_checksum,
            root=root,
            environment=environment,
            source_database=source_database,
            source_row_counts=source_row_counts,
            eligibility=eligibility,
            history_coverage=coverage,
            joins=joins,
            source_target_values=source_target_values,
        )
        if validation["errors"]:
            raise FeatureMartError("; ".join(validation["errors"]))
        if final_dir.exists():
            shutil.rmtree(final_dir)
        temporary_dir.rename(final_dir)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise

    report_directory: Path | None = None
    if not no_report and report_root is not None:
        write_phase5_reports(manifest, validation, report_root, run_id)
        report_directory = report_root / run_id
    validation = validate_mart_directory(final_dir, project_root=root)
    manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
    return Phase5BuildResult(
        status=validation["status"],
        run_directory=str(final_dir),
        report_directory=str(report_directory) if report_directory else None,
        manifest_path=str(final_dir / "manifest.json"),
        validation_path=str(final_dir / "validation.json"),
        manifest=manifest,
        validation=validation,
        warnings=list(validation.get("warnings", [])),
        errors=list(validation.get("errors", [])),
    )


def run_live_phase5(
    settings: Settings,
    *,
    output_dir: Path | None = None,
    report_dir: Path | None = None,
    no_report: bool = False,
    overwrite: bool = False,
) -> Phase5BuildResult:
    """Run the complete live read-only Phase 5 build."""

    root = discover_repository_root()
    mart_settings = load_feature_mart_settings(root)
    schema_contract, schema_checksum = load_schema_contract(root)
    phase4_bundle = load_phase4_contracts(root)
    mart_contract, mart_checksum = load_mart_contract(root)
    plan_result = validate_mart_contract(
        schema_contract,
        phase4_bundle,
        schema_contract_checksum=schema_checksum,
        contract=mart_contract,
        contract_checksum=mart_checksum,
    )
    assert_mart_contract_valid(plan_result)
    check_database_connection(settings.database)
    live_schema = collect_schema_metadata(settings.database, schema_contract)
    schema_result = validate_schema(
        schema_contract,
        live_schema,
        schema_checksum,
        environment=settings.environment,
        strict=False,
        server=settings.database.server,
    )
    if schema_result.status != "passed":
        raise FeatureMartError("Live schema validation did not pass; Phase 5 is blocked.")
    extraction_plan = build_extraction_plan(schema_contract, phase4_bundle, mart_contract)
    extracted = LiveFeatureMartExtractor(
        settings.database,
        extraction_plan,
        chunk_size=mart_settings.history_chunk_size,
    ).extract()
    claims = extracted["frames"]["dbo.fact_warranty_claim"]
    trucks = extracted["frames"]["dbo.dim_truck"]
    claims_for_eligibility = claims[
        ["warranty_claim_key", "claim_date", "high_cost_claim_flag", "truck_key"]
    ]
    eligibility_mask = claim_eligibility_mask(claims_for_eligibility, trucks[["truck_key"]])
    target = pd.to_numeric(claims["high_cost_claim_flag"], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise FeatureMartError("Live target validation failed; Phase 5 is blocked.")
    if int(eligibility_mask.sum()) == 0:
        raise FeatureMartError("No live claims satisfy Phase 4 eligibility; Phase 5 is blocked.")
    output_root = resolve_mart_output_root(root, mart_settings, output_dir)
    report_root = resolve_mart_report_root(root, mart_settings, report_dir)
    return build_feature_mart_from_frames(
        frames=extracted["frames"],
        source_row_counts=extracted["source_row_counts"],
        root=root,
        settings=mart_settings,
        environment=settings.environment,
        source_database=settings.database.database,
        output_root=output_root,
        report_root=report_root,
        no_report=no_report,
        overwrite=overwrite,
    )


def validate_existing_mart(mart_dir: Path) -> dict[str, Any]:
    """Public wrapper used by the CLI and offline tests."""

    return validate_mart_directory(mart_dir)
