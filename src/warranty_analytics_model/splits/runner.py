"""Phase 6 offline planning, deterministic build, and validation orchestration."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from ..feature_mart.manifest import write_json
from ..paths import discover_repository_root
from .assignments import assignment_date_order_errors, build_split_assignments
from .boundary import determine_boundaries
from .cohorts import (
    build_evaluation_cohorts,
    fingerprint_overlap_summary,
    summarize_evaluation_cohorts,
)
from .config import load_split_settings, settings_as_dict, validate_split_settings
from .group_exposure import (
    available_group_type_diagnostics,
    build_group_exposure,
    summarize_group_overlap,
)
from .input import Phase5MartInput, load_phase5_mart
from .manifest import (
    artifact_metadata,
    assignment_content_sha256,
    build_split_manifest,
    build_test_lock,
    mart_input_fingerprint,
    phase6_run_id,
)
from .models import Phase6BuildResult, SplitError
from .reporting import (
    build_phase6_summary,
    build_split_distribution,
    write_phase6_reports,
)
from .split_contract import load_split_contract, validate_split_contract
from .validation import validate_split_artifacts


def _resolve_path(root: Path, value: Path | None, default: Path) -> Path:
    if value is None:
        return default
    candidate = value.expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _status_from_diagnostics(errors: list[str], warnings: list[str]) -> str:
    return "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS")


def _build_diagnostics(
    *,
    assignments: pd.DataFrame,
    distribution: dict[str, Any],
    fingerprint_overlap: dict[str, Any],
    settings: Any,
    phase5_input: Phase5MartInput,
) -> tuple[list[str], list[str]]:
    """Apply evaluation sufficiency and fraction diagnostics after assignment."""

    errors: list[str] = []
    warnings: list[str] = []
    errors.extend(assignment_date_order_errors(assignments))
    total = int(distribution["total_claims"])
    fractions = {
        split: float(item["row_count"]) / total if total else 0.0
        for split, item in distribution["by_split"].items()
    }
    for split, fraction in fractions.items():
        if fraction < settings.minimum_split_fraction:
            errors.append(
                f"{split} contains {fraction:.6f} of claims, below the {settings.minimum_split_fraction:.6f} minimum."
            )
        requested = settings.requested_fractions[split]
        if abs(fraction - requested) > settings.fraction_warning_tolerance:
            warnings.append(
                f"{split} fraction differs from the requested {requested:.6f} by more than "
                f"{settings.fraction_warning_tolerance:.6f} due to date grouping."
            )
    for split in ("TRAIN", "VALIDATION", "TEST"):
        item = distribution["by_split"][split]
        if item["positive_count"] == 0 or item["negative_count"] == 0:
            errors.append(f"{split} contains only one target class.")
    validation_positive = distribution["by_split"]["VALIDATION"]["positive_count"]
    test_positive = distribution["by_split"]["TEST"]["positive_count"]
    train_positive = distribution["by_split"]["TRAIN"]["positive_count"]
    if validation_positive < settings.min_positive_block_validation:
        errors.append(
            f"VALIDATION contains {validation_positive} positive claims, below the blocking minimum."
        )
    elif validation_positive < settings.min_positive_warning_validation:
        warnings.append(
            f"VALIDATION contains only {validation_positive} positive claims; evaluation sufficiency is limited."
        )
    if test_positive < settings.min_positive_block_test:
        errors.append(f"TEST contains {test_positive} positive claims, below the blocking minimum.")
    elif test_positive < settings.min_positive_warning_test:
        warnings.append(
            f"TEST contains only {test_positive} positive claims; evaluation sufficiency is limited."
        )
    if train_positive < settings.min_positive_warning_train:
        warnings.append(
            f"TRAIN contains only {train_positive} positive claims; training sufficiency is limited."
        )
    if fingerprint_overlap["validation_fingerprints_seen_in_train"]:
        warnings.append("Validation contains repeated safe-scenario fingerprints seen in TRAIN.")
    if fingerprint_overlap["test_fingerprints_seen_in_development"]:
        warnings.append(
            "TEST contains repeated safe-scenario fingerprints seen in TRAIN or VALIDATION; "
            "the clean cohort remains available."
        )
    source_drift = phase5_input.manifest.get("source_drift", {})
    if any(
        isinstance(item, dict) and int(item.get("delta", 0)) != 0 for item in source_drift.values()
    ):
        warnings.append(
            "Phase 5 source/mart counts differ from the prior synthetic baseline; values were not forced."
        )
    warnings.extend(
        str(item) for item in phase5_input.phase5_validation.get("warnings", []) if item
    )
    return list(dict.fromkeys(errors)), list(dict.fromkeys(warnings))


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def phase6_plan_check_from_input(
    phase5_input: Phase5MartInput,
    contract: Any,
    checksum: str,
    settings: Any,
) -> dict[str, Any]:
    """Run the Phase 6 plan gate against an already loaded Phase 5 mart."""

    schema_contract_checksum = phase5_input.schema_contract_checksum
    result = validate_split_contract(
        contract,
        mart_contract=phase5_input.mart_contract,
        mart_contract_checksum=phase5_input.mart_contract_checksum,
        schema_contract_checksum=schema_contract_checksum,
        phase4_bundle=phase5_input.phase4_bundle,
        settings=settings,
        split_contract_checksum_value=checksum,
    )
    errors = list(result.errors)
    warnings = list(result.warnings)
    warnings.extend(
        str(item) for item in phase5_input.phase5_validation.get("warnings", []) if item
    )
    group_diagnostics = available_group_type_diagnostics(
        phase5_input.frames["claim_group_membership"]
    )
    for group_type in group_diagnostics["missing_optional_group_types"]:
        warnings.append(f"Optional group type is not available in the Phase 5 mart: {group_type}")
    settings_errors = validate_split_settings(settings)
    errors.extend(settings_errors)
    return {
        "status": _status_from_diagnostics(errors, warnings),
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "phase5_validation": phase5_input.phase5_validation,
        "split_contract": result.model_dump(mode="json"),
        "split_settings": settings_as_dict(settings),
        "group_types": group_diagnostics,
        "mart_run": phase5_input.mart_dir.name,
        "mart_contract_checksum": phase5_input.mart_contract_checksum,
    }


def phase6_plan_check(
    mart_dir: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Run Phase 5, contract, configuration, and group-availability gates offline."""

    root = discover_repository_root(project_root)
    phase5_input = load_phase5_mart(mart_dir, project_root=root)
    contract, checksum = load_split_contract(root)
    settings = load_split_settings(root)
    return phase6_plan_check_from_input(phase5_input, contract, checksum, settings)


def build_phase6(
    mart_dir: Path,
    *,
    output_dir: Path | None = None,
    report_dir: Path | None = None,
    run_id: str | None = None,
    overwrite: bool = False,
    no_report: bool = False,
    project_root: Path | None = None,
) -> Phase6BuildResult:
    """Build and validate one immutable Phase 6 split run from a Phase 5 mart."""

    root = discover_repository_root(project_root)
    phase5_input = load_phase5_mart(mart_dir, project_root=root)
    settings = load_split_settings(root)
    contract, contract_checksum = load_split_contract(root)
    plan = phase6_plan_check_from_input(phase5_input, contract, contract_checksum, settings)
    if not plan["valid"]:
        raise SplitError("Phase 6 plan check blocks the build: " + "; ".join(plan["errors"]))
    boundaries = determine_boundaries(phase5_input.frames["claim_snapshot"], settings)
    assignments = build_split_assignments(phase5_input.frames["claim_snapshot"], boundaries)
    distribution = build_split_distribution(assignments, phase5_input.frames["claim_snapshot"])
    exposure = build_group_exposure(
        assignments,
        phase5_input.frames["claim_group_membership"],
    )
    cohorts = build_evaluation_cohorts(
        assignments,
        phase5_input.frames["claim_group_membership"],
    )
    fingerprint_overlap = fingerprint_overlap_summary(assignments, exposure)
    errors, warnings = _build_diagnostics(
        assignments=assignments,
        distribution=distribution,
        fingerprint_overlap=fingerprint_overlap,
        settings=settings,
        phase5_input=phase5_input,
    )
    status = _status_from_diagnostics(errors, warnings)
    if errors:
        raise SplitError("Phase 6 split diagnostics block the build: " + "; ".join(errors))

    actual_fractions = {
        split: round(float(item["row_count"]) / int(distribution["total_claims"]), 9)
        for split, item in distribution["by_split"].items()
    }
    selected_run_id = run_id or phase6_run_id()
    output_root = _resolve_path(root, output_dir, root / "artifacts" / "splits")
    final_dir = output_root / selected_run_id
    if final_dir.exists():
        if (final_dir / "split_manifest.json").is_file() or (
            final_dir / "test_lock.json"
        ).is_file():
            raise SplitError(f"Completed Phase 6 split run is immutable: {final_dir}")
        if not overwrite:
            raise SplitError(
                f"Phase 6 output directory already exists; use a new run_id: {final_dir}"
            )
        shutil.rmtree(final_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    temp_dir = output_root / f".phase6_{selected_run_id}_{uuid.uuid4().hex}.tmp"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        assignment_metadata = artifact_metadata(assignments, temp_dir / "split_assignments.parquet")
        exposure_metadata = artifact_metadata(exposure, temp_dir / "group_exposure.parquet")
        cohort_metadata = artifact_metadata(cohorts, temp_dir / "evaluation_cohorts.parquet")
        artifact_metadata_by_name = {
            "split_assignments": assignment_metadata,
            "group_exposure": exposure_metadata,
            "evaluation_cohorts": cohort_metadata,
        }
        manifest = build_split_manifest(
            split_contract_version=contract.contract_version,
            split_contract_checksum=contract_checksum,
            input_mart_run=phase5_input.mart_dir.name,
            input_mart_contract_version=phase5_input.mart_contract.version,
            input_mart_contract_checksum=phase5_input.mart_contract_checksum,
            input_mart_manifest_checksum=phase5_input.mart_manifest_checksum,
            input_mart_relative_path=_relative_path(root, phase5_input.mart_dir),
            input_schema_contract_checksum=phase5_input.schema_contract_checksum,
            input_target_contract_checksum=phase5_input.phase4_bundle.target_checksum,
            input_feature_policy_checksum=phase5_input.phase4_bundle.feature_policy_checksum,
            input_leakage_policy_checksum=phase5_input.phase4_bundle.leakage_checksum,
            claim_snapshot_file_sha256=phase5_input.claim_snapshot_file_sha256,
            claim_snapshot_content_sha256=phase5_input.claim_snapshot_content_sha256,
            group_membership_file_sha256=phase5_input.group_membership_file_sha256,
            group_membership_content_sha256=phase5_input.group_membership_content_sha256,
            total_claims=int(distribution["total_claims"]),
            requested_split_fractions=settings.requested_fractions,
            actual_split_fractions=actual_fractions,
            train_end_date=boundaries.train_end_date.isoformat(),
            validation_end_date=boundaries.validation_end_date.isoformat(),
            counts=distribution["by_split"],
            assignments=assignments,
            group_exposure=exposure,
            evaluation_cohorts=cohorts,
            artifact_metadata_by_name=artifact_metadata_by_name,
            warnings=warnings,
            validation_status=status,
            input_mart_source_drift=phase5_input.manifest.get("source_drift", {}),
        )
        input_fingerprint = mart_input_fingerprint(
            mart_contract_checksum=phase5_input.mart_contract_checksum,
            claim_snapshot_content_sha256=phase5_input.claim_snapshot_content_sha256,
            group_membership_content_sha256=phase5_input.group_membership_content_sha256,
        )
        test_assignments = assignments.loc[assignments["split"] == "TEST"]
        test_lock = build_test_lock(
            split_contract_version=contract.contract_version,
            split_contract_checksum=contract_checksum,
            input_mart_checksum=phase5_input.mart_contract_checksum,
            input_mart_fingerprint=input_fingerprint,
            claim_snapshot_content_sha256=phase5_input.claim_snapshot_content_sha256,
            test_assignments=test_assignments,
            test_assignment_content_sha256=assignment_content_sha256(test_assignments),
            test_start_date=str(distribution["by_split"]["TEST"]["earliest_claim_date"]),
            test_end_date=str(distribution["by_split"]["TEST"]["latest_claim_date"]),
        )
        write_json(temp_dir / "split_manifest.json", manifest)
        write_json(temp_dir / "test_lock.json", test_lock)
        validation = validate_split_artifacts(
            temp_dir,
            project_root=root,
            input_mart=phase5_input,
            expected_exposure=exposure,
            expected_cohorts=cohorts,
        )
        if validation.get("errors"):
            raise SplitError(
                "Phase 6 artifact validation blocks the run: "
                + "; ".join(str(item) for item in validation["errors"])
            )
        write_json(temp_dir / "split_validation.json", validation)
        temp_dir.replace(final_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    final_validation = validate_split_artifacts(
        final_dir,
        project_root=root,
        input_mart=phase5_input,
        expected_exposure=exposure,
        expected_cohorts=cohorts,
    )
    final_status = str(final_validation.get("status", status))
    output_report_dir: Path | None = None
    if not no_report:
        report_root = _resolve_path(root, report_dir, root / "reports" / "phase6_splits")
        output_report_dir = report_root / selected_run_id
        group_overlap = summarize_group_overlap(exposure)
        cohort_summary = summarize_evaluation_cohorts(cohorts)
        summary = build_phase6_summary(
            manifest=manifest,
            validation=final_validation,
            split_distribution=distribution,
            group_overlap=group_overlap,
            cohort_summary=cohort_summary,
            fingerprint_overlap=fingerprint_overlap,
            phase5_validation=phase5_input.phase5_validation,
            test_lock_valid=bool(final_validation.get("checks", {}).get("test_lock_valid")),
        )
        write_phase6_reports(
            output_root=output_report_dir,
            summary=summary,
            split_distribution=distribution,
            group_overlap=group_overlap,
            cohort_summary=cohort_summary,
            validation=final_validation,
        )
    return Phase6BuildResult(
        status=final_status,  # type: ignore[arg-type]
        run_directory=str(final_dir),
        report_directory=str(output_report_dir) if output_report_dir else None,
        manifest_path=str(final_dir / "split_manifest.json"),
        validation_path=str(final_dir / "split_validation.json"),
        manifest=manifest,
        validation=final_validation,
        warnings=list(final_validation.get("warnings", [])),
        errors=list(final_validation.get("errors", [])),
    )


def validate_existing_split(
    split_dir: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Validate an existing immutable Phase 6 run."""

    return validate_split_artifacts(split_dir, project_root=project_root)
