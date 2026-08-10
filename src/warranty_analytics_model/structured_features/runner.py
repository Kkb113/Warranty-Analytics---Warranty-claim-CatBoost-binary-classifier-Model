"""Phase 7 contract, plan, build, and validation orchestration."""

from __future__ import annotations

import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..feature_mart.manifest import write_json
from ..feature_mart.mart_contract import load_mart_contract
from ..paths import discover_repository_root
from .builder import build_feature_matrix
from .config import load_structured_feature_settings
from .contract import load_structured_feature_contract, validate_structured_feature_contract
from .input import phase7_plan_check, verify_frozen_membership
from .manifest import (
    build_run_manifest,
    source_coverage,
    write_feature_artifacts,
)
from .models import StructuredFeatureError
from .reporting import write_phase7_reports
from .validation import validate_feature_directory


def phase7_run_id() -> str:
    """Create a readable UTC Phase 7 run identifier."""

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _resolve(root: Path, value: Path | None, default: str) -> Path:
    configured = value if value is not None else Path(default)
    return configured.resolve() if configured.is_absolute() else (root / configured).resolve()


def phase7_contract_check(project_root: Path | None = None) -> dict[str, Any]:
    """Run the database-independent Phase 7 contract check."""

    return validate_structured_feature_contract(project_root)


def build_phase7(
    mart_dir: Path,
    split_dir: Path,
    *,
    output_dir: Path | None = None,
    report_dir: Path | None = None,
    run_id: str | None = None,
    overwrite: bool = False,
    no_report: bool = False,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Build and atomically publish one immutable Phase 7 feature run."""

    root = discover_repository_root(project_root)
    settings = load_structured_feature_settings(root)
    plan = phase7_plan_check(mart_dir, split_dir, project_root=root)
    if not plan["valid"]:
        raise StructuredFeatureError(
            "Phase 7 plan check blocks the build: " + "; ".join(plan["errors"])
        )
    inputs = plan["inputs"]
    if inputs is None:
        raise StructuredFeatureError("Phase 7 inputs were not loaded after a passing plan check.")
    frozen = verify_frozen_membership(inputs)
    if not frozen["valid"]:
        raise StructuredFeatureError(
            "Phase 6 TEST lock blocks Phase 7: " + "; ".join(frozen["errors"])
        )
    assignments = pd.read_parquet(inputs.split_dir / "split_assignments.parquet")
    feature_frames = {name: frame.copy() for name, frame in inputs.frames.items()}
    if "claim_snapshot" in feature_frames:
        feature_frames["claim_snapshot"] = feature_frames["claim_snapshot"].drop(
            columns=["target__high_cost_claim_flag"], errors="ignore"
        )
    built = build_feature_matrix(feature_frames, assignments, settings)
    _, contract_checksum = load_structured_feature_contract(root)
    mart_contract, _ = load_mart_contract(root)
    coverage = source_coverage(built.definitions, mart_contract)
    warnings = list(dict.fromkeys(list(plan["warnings"]) + built.warnings))
    output_root = _resolve(root, output_dir, settings.output_directory)
    report_root = _resolve(root, report_dir, settings.report_directory)
    selected_run_id = run_id or phase7_run_id()
    final_dir = output_root / selected_run_id
    if final_dir.exists() and not overwrite:
        raise StructuredFeatureError(f"Completed Phase 7 run is immutable: {final_dir}")
    if final_dir.exists() and overwrite:
        shutil.rmtree(final_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_root / f".phase7_{selected_run_id}_{uuid.uuid4().hex}.tmp"
    temporary_dir.mkdir(parents=True, exist_ok=False)
    try:
        parquet_metadata = write_feature_artifacts(
            temporary_dir, frame=built.frame, definitions=built.definitions, settings=settings
        )
        write_json(
            temporary_dir / "manifest.json",
            build_run_manifest(
                root=root,
                inputs=inputs,
                contract_checksum=contract_checksum,
                settings=settings,
                frame=built.frame,
                definitions=built.definitions,
                parquet_metadata=parquet_metadata,
                validation_status="INCOMPLETE",
                warnings=warnings,
                coverage=coverage,
            ),
        )
        validation = validate_feature_directory(temporary_dir, project_root=root, inputs=inputs)
        if validation.get("errors"):
            raise StructuredFeatureError(
                "Phase 7 artifact validation blocks publication: "
                + "; ".join(str(item) for item in validation["errors"])
            )
        final_manifest = build_run_manifest(
            root=root,
            inputs=inputs,
            contract_checksum=contract_checksum,
            settings=settings,
            frame=built.frame,
            definitions=built.definitions,
            parquet_metadata=parquet_metadata,
            validation_status=str(validation.get("status", "PASS")),
            warnings=list(dict.fromkeys(warnings + list(validation.get("warnings", [])))),
            coverage=coverage,
        )
        write_json(temporary_dir / "manifest.json", final_manifest)
        write_json(temporary_dir / "validation.json", validation)
        temporary_dir.replace(final_dir)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    final_validation = validate_feature_directory(final_dir, project_root=root, inputs=inputs)
    final_manifest = json_read(final_dir / "manifest.json")
    if not no_report:
        report_path = report_root / selected_run_id
        quality = json_read(final_dir / "feature_quality.json")
        lineage = json_read(final_dir / "feature_lineage.json")
        write_phase7_reports(
            report_path,
            manifest=final_manifest,
            validation=final_validation,
            quality=quality,
            lineage=lineage,
            coverage=coverage,
        )
    else:
        report_path = None
    return {
        "status": final_validation.get("status", "BLOCKED"),
        "run_directory": str(final_dir),
        "report_directory": str(report_path) if report_path else None,
        "manifest": final_manifest,
        "validation": final_validation,
        "warnings": final_validation.get("warnings", []),
        "errors": final_validation.get("errors", []),
    }


def json_read(path: Path) -> dict[str, Any]:
    """Read one generated JSON object."""

    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StructuredFeatureError(f"Generated JSON must be an object: {path}")
    return payload


def validate_existing_feature_run(
    feature_dir: Path, *, project_root: Path | None = None
) -> dict[str, Any]:
    """Validate a completed Phase 7 run entirely offline."""

    return validate_feature_directory(feature_dir, project_root=project_root)
