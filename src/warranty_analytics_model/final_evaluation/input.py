"""Explicit Phase 14 resolution and target-safe TEST population loading."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds

from ..baseline_model.input import PROHIBITED_SOURCE_SUFFIXES
from ..calibration_ensemble.input import KEY, TARGET
from ..catboost_optimization.input import CLAIM_DATE
from ..catboost_optimization.provenance import canonical_json_sha256, sha256_file
from ..paths import discover_repository_root
from ..robustness_analysis.input import Phase14Resolved, resolve_phase13_parent


class Phase15InputError(ValueError):
    """Raised when the explicit Phase 14/Test input chain is unsafe."""


@dataclass(frozen=True, slots=True)
class Phase15Resolved:
    root: Path
    phase14_dir: Path
    phase14_manifest: dict[str, Any]
    phase14_validation: dict[str, Any]
    phase14_freeze: dict[str, Any]
    phase14_plan: dict[str, Any]
    phase14_readiness: dict[str, Any]
    phase13: Phase14Resolved
    test_features: pd.DataFrame
    test_assignments: pd.DataFrame
    phase6_manifest: dict[str, Any]
    test_lock: dict[str, Any]
    claim_snapshot_path: Path
    phase14_manifest_sha256: str
    phase14_validation_sha256: str
    phase14_freeze_sha256: str
    phase14_contract_sha256: str
    phase14_configuration_sha256: str
    phase13_manifest_sha256: str
    feature_schema_sha256: str

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.phase13.feature_names

    @property
    def components(self) -> tuple[Any, ...]:
        return self.phase13.components

    @property
    def champion_id(self) -> str:
        return self.phase13.champion_id

    @property
    def champion_type(self) -> str:
        return self.phase13.champion_type

    @property
    def score_space(self) -> str:
        return self.phase13.score_space

    @property
    def threshold(self) -> float:
        return self.phase13.threshold


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase15InputError(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise Phase15InputError(f"{label} must be a JSON object: {path}")
    return payload


def _test_seal(payload: dict[str, Any], label: str) -> list[str]:
    expected = {
        "test_target_rows_loaded": 0,
        "test_predictions_created": 0,
        "test_metrics_computed": False,
        "test_target_access_allowed": False,
        "first_allowed_test_target_phase": 15,
    }
    return [
        f"{label} TEST seal changed: {key}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]


def _phase14_commit_reachable_from_main(root: Path, commit: str) -> bool:
    if not commit or commit == "unknown":
        return False
    for ref in ("refs/remotes/origin/main", "refs/heads/main"):
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", str(commit), ref],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return True
    return False


def _phase14_ci_green(payload: dict[str, Any], parent: dict[str, Any]) -> bool:
    values = (
        payload.get("post_merge_main_quality_ci_green"),
        payload.get("main_quality_ci_green"),
        parent.get("phase14_post_merge_main_quality_ci_green"),
        parent.get("post_merge_main_quality_ci_green"),
    )
    return any(value is True or str(value).upper() == "GREEN" for value in values)


def _build_full_feature_frame(phase13: Phase14Resolved) -> pd.DataFrame:
    inputs = phase13.phase12_lock.phase12_inputs.phase10_inputs.phase9_inputs
    names = list(phase13.feature_names)
    phase7_names = set(inputs.phase7_lineage)
    structured_columns = [KEY] + [name for name in names if name in phase7_names]
    text_columns = [KEY] + [name for name in names if name not in phase7_names]
    structured = inputs.structured_features[structured_columns].copy()
    text = inputs.text_features[text_columns].copy()
    if structured[KEY].duplicated().any() or text[KEY].duplicated().any():
        raise Phase15InputError("Phase 15 feature inputs contain duplicate claim keys.")
    combined = structured.merge(text, on=KEY, how="outer", validate="one_to_one", indicator=True)
    if (combined["_merge"] != "both").any():
        raise Phase15InputError("Phase 15 structured/text feature memberships differ.")
    combined = combined.drop(columns="_merge")
    controls = inputs.assignments[[KEY, "split"]].copy()
    full = controls.merge(combined, on=KEY, how="left", validate="one_to_one", indicator=True)
    if not (full["_merge"] == "both").all():
        raise Phase15InputError("Phase 15 feature membership differs from Phase 6.")
    full = full.drop(columns="_merge")
    if CLAIM_DATE not in full.columns:
        dates = inputs.structured_features[[KEY, CLAIM_DATE]].copy()
        full = full.merge(dates, on=KEY, how="left", validate="one_to_one")
    missing = sorted(set(names) - set(full.columns))
    if missing:
        raise Phase15InputError(
            "TEST feature matrix is missing frozen features: " + ", ".join(missing)
        )
    if TARGET in full.columns:
        raise Phase15InputError("TEST feature matrix contains the target column.")
    return full.sort_values(KEY, kind="mergesort").reset_index(drop=True)


def _schema_payload(phase13: Phase14Resolved) -> dict[str, Any]:
    return {
        component.track: {
            "feature_names": list(component.feature_set.feature_names),
            "numeric_features": list(component.feature_set.numeric_features),
            "categorical_features": list(component.feature_set.categorical_features),
            "boolean_features": list(component.feature_set.boolean_features),
            "text_features": list(component.feature_set.text_features),
            "feature_list_sha256": component.feature_list_sha256,
        }
        for component in phase13.components
    }


def feature_schema_sha256(phase13: Phase14Resolved) -> str:
    return canonical_json_sha256(_schema_payload(phase13))


def _validate_phase14_start(
    root: Path,
    manifest: dict[str, Any],
    validation: dict[str, Any],
    readiness: dict[str, Any],
    parent_resolution: dict[str, Any],
    *,
    require_main_merge: bool,
) -> None:
    if manifest.get("phase") != 14 or not manifest.get("run_id"):
        raise Phase15InputError("Explicit Phase 14 directory is not a valid Phase 14 run.")
    if validation.get("valid") is not True:
        raise Phase15InputError("Phase 14 validation is not valid.")
    if validation.get("hardening_status") not in {"HARDENED_PASS", "HARDENED_PASS_WITH_WARNINGS"}:
        raise Phase15InputError("Phase 14 hardening status is not accepted.")
    if validation.get("errors"):
        raise Phase15InputError("Phase 14 validation contains errors.")
    if readiness.get("status") not in {"READY", "READY_WITH_WARNINGS"}:
        raise Phase15InputError("Phase 14 is not ready for Phase 15.")
    if readiness.get("safe_to_start_phase15") is False:
        raise Phase15InputError("Phase 14 explicitly blocks Phase 15.")
    # Older accepted Phase 14 manifests persisted the seal in validation.json
    # rather than duplicating it at the manifest root.  Reconcile both sources
    # without weakening the values: any present root value must still agree.
    manifest_seal = _test_seal(manifest, "Phase 14 manifest")
    if manifest_seal:
        validation_seal = validation.get("test_seal")
        if not isinstance(validation_seal, dict) or _test_seal(
            validation_seal, "Phase 14 validation"
        ):
            raise Phase15InputError("Phase 14 manifest TEST seal is not closed.")
    if require_main_merge:
        commit = str(manifest.get("git_commit_sha", ""))
        if not _phase14_commit_reachable_from_main(root, commit):
            raise Phase15InputError("Phase 14 implementation is not merged into main.")
        if not _phase14_ci_green(manifest, parent_resolution):
            raise Phase15InputError("Post-merge main Quality CI is not recorded GREEN.")


def build_test_membership_audit(
    assignments: pd.DataFrame,
    phase6_manifest: dict[str, Any],
    test_lock: dict[str, Any],
) -> dict[str, Any]:
    test = assignments.loc[assignments["split"].astype(str) == "TEST"].copy()
    if test.empty or test[KEY].duplicated().any():
        raise Phase15InputError("Phase 6 TEST membership is empty or duplicated.")
    from ..splits.manifest import (
        assignment_content_sha256,
        claim_key_sha256,
        unordered_claim_key_sha256,
    )

    hashes = {
        "split_assignment_sha256": assignment_content_sha256(assignments),
        "ordered_test_claim_keys_sha256": claim_key_sha256(test),
        "unordered_test_claim_keys_sha256": unordered_claim_key_sha256(test),
        "test_assignment_content_sha256": assignment_content_sha256(test),
    }
    expected = {
        "split_assignment_sha256": phase6_manifest.get("split_assignment_sha256"),
        "ordered_test_claim_keys_sha256": test_lock.get("ordered_test_claim_keys_sha256"),
        "unordered_test_claim_keys_sha256": test_lock.get("unordered_test_claim_keys_sha256"),
        "test_assignment_content_sha256": test_lock.get("test_assignment_content_sha256"),
    }
    drift = [key for key, value in expected.items() if value and hashes[key] != value]
    expected_rows = int(test_lock.get("test_row_count", phase6_manifest.get("test_count", -1)))
    if expected_rows != len(test):
        drift.append("test_row_count")
    if drift:
        raise Phase15InputError("Phase 6 TEST membership drifted: " + ", ".join(drift))
    return {
        "phase": 15,
        "target_independent": True,
        "test_targets_accessed": False,
        "expected_test_row_count": len(test),
        **hashes,
        "phase6_expected": expected,
    }


def load_test_targets_after_freeze(
    resolved: Phase15Resolved,
    freeze: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The sole TEST-label loader; callers must present a persisted freeze."""

    if any(
        freeze.get(key) is not expected
        for key, expected in {
            "test_targets_accessed": False,
            "test_predictions_created": False,
            "test_metrics_computed": False,
        }.items()
    ):
        raise Phase15InputError("TEST target access requires an untouched Phase 15 freeze.")
    keys = set(resolved.test_features[KEY].astype(int))
    dataset = ds.dataset(str(resolved.claim_snapshot_path), format="parquet")
    if not {KEY, TARGET}.issubset(dataset.schema.names):
        raise Phase15InputError("Phase 5 claim snapshot lacks the authoritative TEST target.")
    frame = dataset.to_table(
        columns=[KEY, TARGET], filter=ds.field(KEY).isin(sorted(keys))
    ).to_pandas()
    if set(frame[KEY].astype(int)) != keys or frame[KEY].duplicated().any():
        raise Phase15InputError("TEST target membership differs from frozen Phase 6 membership.")
    values = pd.to_numeric(frame[TARGET], errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise Phase15InputError("TEST target is not binary and non-null.")
    frame[TARGET] = values.astype("int8")
    frame = frame[[KEY, TARGET]].sort_values(KEY, kind="mergesort").reset_index(drop=True)
    from ..baseline_model.target import target_content_sha256

    audit = {
        "phase": 15,
        "first_allowed_test_target_phase": 15,
        "first_access_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "access_reason": "FINAL_UNTOUCHED_TEST_EVALUATION",
        "test_target_rows_loaded": int(len(frame)),
        "test_target_claim_key_sha256": canonical_json_sha256(frame[KEY].astype(int).tolist()),
        "test_target_value_sha256": target_content_sha256(frame),
        "first_access_after_phase15_freeze": True,
        "model_selection_using_TEST": False,
        "threshold_tuning_using_TEST": False,
        "calibration_tuning_using_TEST": False,
        "ensemble_tuning_using_TEST": False,
        "feature_selection_using_TEST": False,
    }
    return frame, audit


def resolve_phase14_parent(
    phase14_dir: Path,
    *,
    project_root: Path | None = None,
    require_main_merge: bool = True,
    validate_upstream: bool = True,
) -> Phase15Resolved:
    root = discover_repository_root(project_root or phase14_dir)
    directory = phase14_dir.expanduser().resolve()
    required = (
        "phase14_manifest.json",
        "phase14_analysis_freeze.json",
        "analysis_plan.json",
        "phase13_parent_resolution.json",
        "phase15_readiness.json",
        "validation.json",
        "leakage_recheck.json",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise Phase15InputError("Phase 14 artifacts missing: " + ", ".join(missing))
    manifest = _read_json(directory / "phase14_manifest.json", "Phase 14 manifest")
    validation = _read_json(directory / "validation.json", "Phase 14 validation")
    freeze = _read_json(directory / "phase14_analysis_freeze.json", "Phase 14 freeze")
    plan = _read_json(directory / "analysis_plan.json", "Phase 14 plan")
    readiness = _read_json(directory / "phase15_readiness.json", "Phase 15 readiness")
    parent = _read_json(directory / "phase13_parent_resolution.json", "Phase 13 parent resolution")
    _validate_phase14_start(
        root, manifest, validation, readiness, parent, require_main_merge=require_main_merge
    )
    if (
        freeze.get("development_decisions_frozen") is not True
        or freeze.get("test_targets_accessed") is not False
    ):
        raise Phase15InputError("Phase 14 freeze is not an immutable development freeze.")
    if validate_upstream:
        from ..robustness_analysis.validation import validate_existing_phase14

        replay = validate_existing_phase14(directory, project_root=root)
        if replay.get("valid") is not True or replay.get("errors"):
            raise Phase15InputError("Standalone Phase 14 validator did not pass.")
    phase13_path = Path(str(manifest.get("phase13_dir", "")))
    if not phase13_path.is_absolute():
        phase13_path = (root / phase13_path).resolve()
    if not phase13_path.is_dir():
        raise Phase15InputError("Phase 14 does not identify an accessible Phase 13 artifact.")
    phase13 = resolve_phase13_parent(phase13_path, project_root=root, require_main_merge=False)
    full = _build_full_feature_frame(phase13)
    assignments = (
        phase13.phase12_lock.phase12_inputs.phase10_inputs.phase9_inputs.assignments.copy()
    )
    test_keys = set(assignments.loc[assignments["split"] == "TEST", KEY].astype(int))
    test_features = full.loc[full[KEY].astype(int).isin(test_keys)].copy()
    if set(test_features[KEY].astype(int)) != test_keys or test_features[KEY].duplicated().any():
        raise Phase15InputError("TEST feature membership differs from Phase 6.")
    phase9 = phase13.phase12_lock.phase12_inputs.phase10_inputs.phase9_inputs
    membership = build_test_membership_audit(assignments, phase9.phase6_manifest, phase9.test_lock)
    del membership  # Reconstructed by the runner before the pre-TEST freeze.
    return Phase15Resolved(
        root=root,
        phase14_dir=directory,
        phase14_manifest=manifest,
        phase14_validation=validation,
        phase14_freeze=freeze,
        phase14_plan=plan,
        phase14_readiness=readiness,
        phase13=phase13,
        test_features=test_features.sort_values(KEY, kind="mergesort").reset_index(drop=True),
        test_assignments=assignments.loc[assignments["split"] == "TEST"].copy(),
        phase6_manifest=phase9.phase6_manifest,
        test_lock=phase9.test_lock,
        claim_snapshot_path=phase13.phase12_lock.phase12_inputs.phase10_inputs.claim_snapshot_path,
        phase14_manifest_sha256=sha256_file(directory / "phase14_manifest.json"),
        phase14_validation_sha256=sha256_file(directory / "validation.json"),
        phase14_freeze_sha256=sha256_file(directory / "phase14_analysis_freeze.json"),
        phase14_contract_sha256=str(manifest.get("contract_sha256", "")),
        phase14_configuration_sha256=str(manifest.get("configuration_sha256", "")),
        phase13_manifest_sha256=phase13.phase13_manifest_sha256,
        feature_schema_sha256=feature_schema_sha256(phase13),
    )


def leakage_audit(resolved: Phase15Resolved) -> dict[str, Any]:
    names = list(resolved.feature_names)
    prohibited: list[str] = []
    lineage = (
        resolved.phase13.phase12_lock.phase12_inputs.phase10_inputs.phase9_inputs.phase7_lineage
    )
    for name in names:
        suffix = str(name).split("__")[-1]
        lower = str(name).lower()
        if suffix in PROHIBITED_SOURCE_SUFFIXES or any(
            token in lower
            for token in ("root_cause", "approval_outcome", "repair_end_date", "days_to_repair")
        ):
            prohibited.append(str(name))
        item = lineage.get(name)
        if isinstance(item, dict) and item.get("target_dependent") is not False:
            prohibited.append(str(name))
    values = sorted(set(prohibited))
    return {
        "phase": 15,
        "prohibited_features": values,
        "prohibited_feature_count": len(values),
        "valid": not values,
    }


__all__ = [
    "KEY",
    "TARGET",
    "Phase15InputError",
    "Phase15Resolved",
    "build_test_membership_audit",
    "feature_schema_sha256",
    "leakage_audit",
    "load_test_targets_after_freeze",
    "resolve_phase14_parent",
]
