"""Phase 8 plan, build, validation, and atomic publication workflow."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..feature_mart.manifest import write_json
from ..paths import discover_repository_root
from .config import load_text_feature_settings
from .contract import load_text_feature_contract, validate_text_feature_contract
from .input import phase8_plan_check
from .lexical import build_lexical_features
from .manifest import (
    build_text_feature_manifest,
    build_text_run_manifest,
    ordered_text_frame,
    write_text_artifact,
)
from .models import Phase8Inputs, TextBuildResult, TextFeatureDefinition, TextFeatureError
from .reporting import write_phase8_reports
from .validation import validate_text_directory


def phase8_contract_check(project_root: Path | None = None) -> dict[str, Any]:
    """Validate the Phase 8 contract without reading generated artifacts."""

    return validate_text_feature_contract(project_root)


def _control_definitions() -> list[TextFeatureDefinition]:
    return [
        TextFeatureDefinition(
            feature_name="warranty_claim_key",
            tier="CONTROL",
            feature_type="numeric",
            source_artifacts=("split_assignments",),
            source_columns=("warranty_claim_key",),
            control_sources=("warranty_claim_key",),
            is_model_feature=False,
            is_control=True,
            notes="Claim identity control only; never a text token or model candidate.",
        ),
        TextFeatureDefinition(
            feature_name="split",
            tier="CONTROL",
            feature_type="categorical",
            source_artifacts=("split_assignments",),
            source_columns=("split",),
            control_sources=("split",),
            is_model_feature=False,
            is_control=True,
            notes="Frozen Phase 6 assignment control only.",
        ),
        TextFeatureDefinition(
            feature_name="claim__claim_date",
            tier="CONTROL",
            feature_type="date_control",
            source_artifacts=("split_assignments",),
            source_columns=("claim__claim_date",),
            control_sources=("claim__claim_date",),
            is_model_feature=False,
            is_control=True,
            notes="Date-level prediction reference control; not a raw text value.",
        ),
    ]


def _with_controls(result: TextBuildResult) -> TextBuildResult:
    result.definitions = _control_definitions() + result.definitions
    return result


def _resolve_output(root: Path, configured: Path | None, default: str) -> Path:
    path = configured or Path(default)
    return path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()


def _write_lineage(path: Path, result: TextBuildResult) -> None:
    write_json(path, {item.feature_name: item.as_dict() for item in result.definitions})


def build_phase8(
    mart_dir: Path,
    split_dir: Path,
    structured_dir: Path,
    *,
    output_dir: Path | None = None,
    report_dir: Path | None = None,
    run_id: str | None = None,
    overwrite: bool = False,
    no_report: bool = False,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Build and atomically publish a deterministic Phase 8 text artifact."""

    root = discover_repository_root(project_root)
    contract_result = validate_text_feature_contract(root)
    if not contract_result.get("valid"):
        raise TextFeatureError(
            "Phase 8 contract blocks the build: " + "; ".join(contract_result["errors"])
        )
    plan = phase8_plan_check(mart_dir, split_dir, structured_dir, project_root=root)
    if not plan.get("valid"):
        raise TextFeatureError("Phase 8 plan check blocks the build: " + "; ".join(plan["errors"]))
    inputs = plan["inputs"]
    if not isinstance(inputs, Phase8Inputs):
        raise TextFeatureError("Phase 8 plan check did not return validated inputs.")
    settings = load_text_feature_settings(root)
    documents, document_audit = _build_documents(inputs, settings)
    result = _with_controls(build_lexical_features(documents, settings))
    result.quality["temporal_audit"] = document_audit
    result.frame = ordered_text_frame(result.frame)
    result.warnings.extend(str(item) for item in plan.get("warnings", []))
    result.warnings.extend(str(item) for item in result.quality.get("train_feature_warnings", []))
    result.warnings.append(
        "UNVERSIONED_FAILURE_DESCRIPTION_DIMENSION: real-data reapproval is required."
    )
    contract, contract_checksum = load_text_feature_contract(root)
    output_root = _resolve_output(root, output_dir, settings.output_directory)
    report_root = _resolve_output(root, report_dir, settings.report_directory)
    chosen_run_id = run_id or "20260810T_PHASE8"
    final_dir = output_root / chosen_run_id
    final_report_dir = report_root / chosen_run_id
    if final_dir.exists() and not overwrite:
        raise TextFeatureError(f"Phase 8 run already exists; choose a new run id: {final_dir}")
    if final_report_dir.exists() and not overwrite and not no_report:
        raise TextFeatureError(
            f"Phase 8 report run already exists; choose a new run id: {final_report_dir}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{chosen_run_id}.", dir=str(output_root)))
    temp_report_root: Path | None = None
    try:
        artifact_metadata = write_text_artifact(
            result.frame,
            temp_dir / "text_features.parquet",
            compression=settings.compression,
        )
        write_json(temp_dir / "text_feature_manifest.json", build_text_feature_manifest(result))
        _write_lineage(temp_dir / "text_feature_lineage.json", result)
        write_json(temp_dir / "text_quality.json", result.quality)
        write_json(temp_dir / "source_coverage.json", result.source_coverage)
        manifest = build_text_run_manifest(
            root=root,
            inputs=inputs,
            result=result,
            settings=settings,
            contract_checksum=contract_checksum,
            artifact_metadata=artifact_metadata,
            warnings=result.warnings,
            validation_status="INCOMPLETE",
        )
        manifest["contract_version"] = contract.get("contract_version")
        manifest["text_feature_names"] = [
            item.feature_name for item in result.definitions if item.is_model_feature
        ]
        write_json(temp_dir / "manifest.json", manifest)
        initial_validation = validate_text_directory(temp_dir, project_root=root)
        if initial_validation.get("errors"):
            raise TextFeatureError(
                "Phase 8 validation failed before publication: "
                + "; ".join(map(str, initial_validation["errors"]))
            )
        if not no_report:
            temp_report_root = Path(
                tempfile.mkdtemp(prefix=f".{chosen_run_id}.", dir=str(report_root))
            )
            write_phase8_reports(
                output_root=temp_report_root,
                run_id=chosen_run_id,
                manifest=manifest,
                quality=result.quality,
                source_coverage=result.source_coverage,
                validation=initial_validation,
            )
        final_validation = validate_text_directory(
            temp_dir,
            project_root=root,
            report_dir=(temp_report_root / chosen_run_id if temp_report_root else None),
        )
        if final_validation.get("errors"):
            raise TextFeatureError(
                "Phase 8 validation failed before publication: "
                + "; ".join(map(str, final_validation["errors"]))
            )
        manifest["validation_status"] = final_validation["status"]
        manifest["warnings"] = final_validation.get("warnings", [])
        write_json(temp_dir / "manifest.json", manifest)
        write_json(temp_dir / "validation.json", final_validation)
        if not no_report and temp_report_root is not None:
            write_phase8_reports(
                output_root=temp_report_root,
                run_id=chosen_run_id,
                manifest=manifest,
                quality=result.quality,
                source_coverage=result.source_coverage,
                validation=final_validation,
            )
        if final_dir.exists():
            if not overwrite:
                raise TextFeatureError(f"Cannot replace existing Phase 8 run: {final_dir}")
            shutil.rmtree(final_dir)
        temp_dir.replace(final_dir)
        published_report_dir: Path | None = None
        if not no_report and temp_report_root is not None:
            published_report_dir = temp_report_root / chosen_run_id
            if final_report_dir.exists():
                if not overwrite:
                    raise TextFeatureError(
                        f"Cannot replace existing Phase 8 reports: {final_report_dir}"
                    )
                shutil.rmtree(final_report_dir)
            published_report_dir.replace(final_report_dir)
            temp_report_root.rmdir()
            published_report_dir = final_report_dir
        return {
            "status": final_validation["status"],
            "run_directory": final_dir,
            "report_directory": published_report_dir,
            "validation": final_validation,
            "manifest": manifest,
            "warnings": final_validation.get("warnings", []),
        }
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        if temp_report_root is not None and temp_report_root.exists():
            shutil.rmtree(temp_report_root, ignore_errors=True)
        raise


def _build_documents(inputs: Phase8Inputs, settings: Any) -> tuple[Any, dict[str, Any]]:
    """Local wrapper keeps the runner call explicit and testable."""

    from .documents import build_historical_documents

    return build_historical_documents(inputs, settings)


def validate_existing_text_run(
    text_dir: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Validate a published Phase 8 artifact without database access."""

    root = discover_repository_root(project_root)
    report_dir = root / "reports" / "phase8_text_features" / text_dir.expanduser().resolve().name
    return validate_text_directory(text_dir, project_root=root, report_dir=report_dir)
