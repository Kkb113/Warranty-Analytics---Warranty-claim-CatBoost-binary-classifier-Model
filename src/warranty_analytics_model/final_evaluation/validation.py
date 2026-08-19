"""Independent, fail-closed Phase 15 artifact validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..robustness_analysis.config import load_robustness_settings
from ..robustness_analysis.errors import (
    build_error_context,
    error_cohorts,
    error_profile,
    high_confidence_errors,
)
from ..robustness_analysis.input import KEY, TARGET, prepare_scorer, train_oof_scores
from ..robustness_analysis.invariance import prediction_invariance
from ..robustness_analysis.ranking import risk_decile_metrics, topk_lift
from ..robustness_analysis.slices import evaluate_slices
from ..robustness_analysis.temporal import temporal_metrics
from .bootstrap import stratified_bootstrap
from .config import load_final_test_settings
from .contract import phase15_contract_check
from .input import (
    Phase15InputError,
    build_test_membership_audit,
    load_test_targets_after_freeze,
    resolve_phase14_parent,
)
from .metrics import reliability_table, signal_status, test_metrics, validation_test_comparison
from .planning import frozen_test_slice_definitions, test_slice_memberships
from .provenance import artifact_hashes, canonical_sha
from .scoring import build_final_model_policy, score_test_in_batches
from .status import final_model_status

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "phase15_manifest.json",
    "phase14_parent_resolution.json",
    "final_model_policy.json",
    "phase15_evaluation_plan.json",
    "phase15_evaluation_freeze.json",
    "test_membership_audit.json",
    "phase15_test_slice_definitions.json",
    "phase15_test_slice_memberships.parquet",
    "leakage_recheck.json",
    "target_access_audit.json",
    "test_use_audit.json",
    "test_predictions.parquet",
    "test_metrics.json",
    "test_signal_status.json",
    "validation_test_comparison.json",
    "test_threshold_metrics.json",
    "test_confusion_matrix.json",
    "test_topk_lift.json",
    "test_risk_deciles.parquet",
    "ranking_concentration_summary.json",
    "test_bootstrap.parquet",
    "test_bootstrap_summary.json",
    "test_reliability.parquet",
    "test_temporal_metrics.parquet",
    "test_slice_metrics.parquet",
    "test_error_cohorts.parquet",
    "test_error_summary.json",
    "false_negative_summary.json",
    "test_prediction_invariance.json",
    "phase15_final_model_status.json",
    "compute_manifest.json",
    "validation.json",
)


def _read_json(directory: Path, name: str) -> dict[str, Any]:
    value = json.loads((directory / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _close(left: Any, right: Any, tolerance: float = 1.0e-10) -> bool:
    if left is None or right is None:
        return left is right
    try:
        return bool(
            np.isclose(float(left), float(right), rtol=tolerance, atol=tolerance, equal_nan=True)
        )
    except (TypeError, ValueError):
        return bool(left == right)


def _compare_fields(
    errors: list[str], expected: dict[str, Any], actual: dict[str, Any], label: str
) -> None:
    for key, value in expected.items():
        if key not in actual:
            errors.append(f"{label} missing field: {key}")
        elif isinstance(value, (float, int)) and isinstance(actual[key], (float, int)):
            if not _close(value, actual[key]):
                errors.append(f"{label} drifted: {key}")
        elif actual[key] != value:
            errors.append(f"{label} drifted: {key}")


def _compare_frames(expected: pd.DataFrame, actual: pd.DataFrame, label: str) -> list[str]:
    """Compare independently reconstructed tabular evidence.

    Column order and pandas dtype differences are not scientific differences,
    but row order and values are.  Keeping this comparison in the independent
    validator ensures every persisted table can be replayed without trusting
    its own hash or the runner's serialization path.
    """

    left = expected.copy()
    right = actual.copy()
    if set(left.columns) != set(right.columns):
        return [f"Persisted {label} columns are not reproducible."]
    columns = sorted(left.columns)
    left = left.loc[:, columns].reset_index(drop=True)
    right = right.loc[:, columns].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=1.0e-10,
        )
    except AssertionError:
        return [f"Persisted {label} is not independently reproducible."]
    return []


def _compare_nested(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Compare scientific JSON evidence with a strict scalar tolerance."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected an object."]
        object_issues: list[str] = []
        for key, value in expected.items():
            if key not in actual:
                object_issues.append(f"{path}/{key}: missing.")
            else:
                object_issues.extend(_compare_nested(value, actual[key], f"{path}/{key}"))
        return object_issues
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return [f"{path}: list length/content changed."]
        issues: list[str] = []
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            issues.extend(_compare_nested(left, right, f"{path}[{index}]"))
        return issues
    if not _close(expected, actual):
        return [f"{path}: persisted value is not reproducible."]
    return []


def _blocked(errors: list[str], warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "phase": 15,
        "valid": False,
        "status": "BLOCKED",
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings or [])),
        "test_targets_accessed": False,
    }


def validate_existing_phase15(
    phase15_dir: Path,
    *,
    project_root: Path | None = None,
    allow_unpublished: bool = False,
) -> dict[str, Any]:
    """Reconstruct the frozen serving policy and aggregate TEST evidence."""

    directory = phase15_dir.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if not directory.is_dir():
        return _blocked([f"Phase 15 directory is missing: {directory}"])
    required = [name for name in REQUIRED_ARTIFACTS if not (directory / name).is_file()]
    if allow_unpublished:
        required = [
            name for name in required if name not in {"phase15_manifest.json", "validation.json"}
        ]
    if required:
        return _blocked(["Phase 15 artifacts missing: " + ", ".join(required)])
    try:
        contract = phase15_contract_check(project_root or directory)
        if not contract.get("valid"):
            return _blocked(["Phase 15 contract is invalid."] + list(contract.get("errors", [])))
        freeze = _read_json(directory, "phase15_evaluation_freeze.json")
        policy = _read_json(directory, "final_model_policy.json")
        plan = _read_json(directory, "phase15_evaluation_plan.json")
        membership = _read_json(directory, "test_membership_audit.json")
        leakage = _read_json(directory, "leakage_recheck.json")
        target_audit = _read_json(directory, "target_access_audit.json")
        use_audit = _read_json(directory, "test_use_audit.json")
        predictions = pd.read_parquet(directory / "test_predictions.parquet")
        persisted_metrics = _read_json(directory, "test_metrics.json")
        persisted_signal = _read_json(directory, "test_signal_status.json")
        persisted_comparison = _read_json(directory, "validation_test_comparison.json")
        persisted_threshold = _read_json(directory, "test_threshold_metrics.json")
        persisted_confusion = _read_json(directory, "test_confusion_matrix.json")
        persisted_topk = _read_json(directory, "test_topk_lift.json")
        persisted_definitions = _read_json(directory, "phase15_test_slice_definitions.json")
        persisted_invariance = _read_json(directory, "test_prediction_invariance.json")
        persisted_status = _read_json(directory, "phase15_final_model_status.json")
        persisted_manifest = (
            _read_json(directory, "phase15_manifest.json")
            if (directory / "phase15_manifest.json").is_file()
            else {}
        )
        if (
            freeze.get("development_decisions_frozen") is not True
            or freeze.get("freeze_immutable") is not True
        ):
            errors.append("Phase 15 freeze is not marked immutable.")
        for key, expected in {
            "test_targets_accessed": False,
            "test_predictions_created": False,
            "test_metrics_computed": False,
            "test_target_rows_loaded": 0,
            "test_target_access_allowed": False,
        }.items():
            if freeze.get(key) != expected:
                errors.append(f"Phase 15 freeze TEST seal changed: {key}")
        if policy.get("policy") != "REUSE_FROZEN_PHASE14_CHAMPION":
            errors.append("Phase 15 final model policy is not the frozen Phase 14 champion.")
        if (
            policy.get("model_retraining") is not False
            or policy.get("train_validation_refit") is not False
        ):
            errors.append("Phase 15 indicates retraining or refitting.")
        if policy.get("alternative_candidates_evaluated") is not False:
            errors.append("Phase 15 evaluated an alternative candidate.")
        freeze_body = dict(freeze)
        declared_freeze_sha = freeze_body.pop("phase15_evaluation_freeze_sha256", None)
        if declared_freeze_sha != canonical_sha(freeze_body):
            errors.append("Phase 15 evaluation freeze SHA does not match its content.")
        stable_plan = {
            key: value
            for key, value in plan.items()
            if key != "created_at_utc" and not key.endswith("_sha256")
        }
        if plan.get("evaluation_plan_sha256") != canonical_sha(stable_plan):
            errors.append("Phase 15 evaluation plan SHA does not match its stable definition.")
        if plan.get("scoring_policy_count") != 1 or use_audit.get("scoring_policy_count") != 1:
            errors.append("Phase 15 did not use exactly one frozen scoring policy.")
        for key in (
            "model_selection_using_TEST",
            "threshold_tuning_using_TEST",
            "calibration_tuning_using_TEST",
            "ensemble_tuning_using_TEST",
            "feature_selection_using_TEST",
            "class_weight_tuning_using_TEST",
        ):
            if target_audit.get(key) is not False or use_audit.get(key) is not False:
                errors.append(f"TEST-use policy violation: {key}")
        for key, expected in {
            "phase": 15,
            "one_frozen_scoring_policy": True,
            "scoring_policy_count": 1,
            "test_targets_accessed_after_freeze": True,
            "test_predictions_created": True,
            "test_metrics_computed": True,
        }.items():
            if use_audit.get(key) != expected:
                errors.append(f"TEST-use audit drifted: {key}")
        if leakage.get("valid") is not True:
            errors.append("Leakage recheck is invalid.")
        if KEY not in predictions.columns or predictions[KEY].duplicated().any():
            errors.append("TEST predictions do not contain unique claim keys.")
        if "probability" not in predictions.columns:
            errors.append("TEST predictions lack probability.")
        else:
            probabilities = pd.to_numeric(predictions["probability"], errors="coerce")
            if probabilities.isna().any() or not np.isfinite(probabilities.to_numpy()).all():
                errors.append("TEST predictions contain non-finite probabilities.")
            if ((probabilities < 0.0) | (probabilities > 1.0)).any():
                errors.append("TEST predictions contain probabilities outside [0,1].")
        parent_payload = _read_json(directory, "phase14_parent_resolution.json")
        phase14_path = Path(str(parent_payload.get("phase14_dir", "")))
        if not phase14_path.is_absolute():
            phase14_path = (directory / phase14_path).resolve()
        resolved = resolve_phase14_parent(
            phase14_path,
            project_root=project_root or directory,
            require_main_merge=True,
            validate_upstream=True,
        )
        assignments = (
            resolved.phase13.phase12_lock.phase12_inputs.phase10_inputs.phase9_inputs.assignments
        )
        rebuilt_membership = build_test_membership_audit(
            assignments, resolved.phase6_manifest, resolved.test_lock
        )
        _compare_fields(errors, rebuilt_membership, membership, "TEST membership audit")
        expected_policy = build_final_model_policy(resolved)
        _compare_fields(errors, expected_policy, policy, "Final model policy")
        _compare_fields(
            errors,
            {
                "model_sha256": expected_policy["model_sha256"],
                "calibrator_sha256": expected_policy["calibrator_sha256"],
                "feature_list_sha256": expected_policy["feature_list_sha256"],
                "feature_schema_sha256": resolved.feature_schema_sha256,
                "effective_score_space": resolved.score_space,
                "frozen_threshold": float(resolved.threshold),
                "test_expected_row_count": len(resolved.test_features),
            },
            freeze,
            "Phase 15 freeze provenance",
        )
        definitions = frozen_test_slice_definitions(resolved)
        declared_definitions = persisted_definitions.get("definitions")
        if declared_definitions != definitions:
            errors.append("Phase 15 TEST slice definitions changed.")
        if plan.get("slice_definition_sha256") != canonical_sha(definitions):
            errors.append("Phase 15 TEST slice-definition SHA changed.")
        expected_keys = set(resolved.test_features[KEY].astype(int))
        actual_keys = set(predictions[KEY].astype(int))
        if actual_keys != expected_keys:
            errors.append("TEST predictions changed the frozen membership.")
        if len(predictions) != len(expected_keys):
            errors.append("TEST prediction row count does not match frozen membership.")
        # Re-score from the serialized Phase 13 model/calibrator policy.  This
        # is independent of every persisted Phase 15 metric and status file.
        compute = _read_json(directory, "compute_manifest.json")
        rescored = score_test_in_batches(
            resolved,
            resolved.test_features,
            inference_threads=max(1, int(compute.get("catboost_inference_threads", 1))),
        )
        merged = predictions[[KEY, "probability"]].merge(
            rescored[[KEY, "probability"]],
            on=KEY,
            suffixes=("_persisted", "_recomputed"),
            validate="one_to_one",
        )
        delta = (merged["probability_persisted"] - merged["probability_recomputed"]).abs()
        max_delta = float(delta.max()) if len(delta) else 0.0
        tolerance = float(plan.get("invariance_policy", {}).get("tolerance", 1.0e-10))
        if max_delta > tolerance:
            errors.append(f"TEST prediction replay exceeds tolerance: {max_delta}")
        target_frame, rebuilt_target_audit = load_test_targets_after_freeze(resolved, freeze)
        for key in (
            "first_allowed_test_target_phase",
            "test_target_rows_loaded",
            "test_target_claim_key_sha256",
            "test_target_value_sha256",
            "first_access_after_phase15_freeze",
        ):
            if target_audit.get(key) != rebuilt_target_audit.get(key):
                errors.append(f"TEST target-access audit drifted: {key}")
        target_map = target_frame.set_index(KEY)[TARGET]
        ordered = predictions.sort_values(KEY, kind="mergesort").reset_index(drop=True)
        aligned = ordered[KEY].astype(int).map(target_map)
        if aligned.isna().any():
            errors.append("TEST target membership cannot be aligned to predictions.")
            y = np.zeros(len(ordered), dtype="int8")
        else:
            y = aligned.to_numpy(dtype="int8")
        p = ordered["probability"].to_numpy(dtype="float64")
        threshold = float(freeze.get("frozen_threshold", resolved.threshold))
        score_by_key = rescored.set_index(KEY)["probability"]
        rebuilt_slice_memberships = test_slice_memberships(
            definitions,
            resolved.test_features,
            resolved.test_features[KEY].map(score_by_key),
        )
        persisted_slice_memberships = pd.read_parquet(
            directory / "phase15_test_slice_memberships.parquet"
        )
        errors.extend(
            _compare_frames(
                rebuilt_slice_memberships,
                persisted_slice_memberships,
                "TEST slice memberships",
            )
        )
        recomputed_metrics = test_metrics(y, p, threshold)
        metric_keys = (
            "average_precision",
            "roc_auc",
            "log_loss",
            "brier_score",
            "ece_10",
            "mce_10",
            "prevalence",
            "ap_lift_over_prevalence",
            "row_count",
            "positive_count",
            "negative_count",
        )
        _compare_fields(
            errors,
            {key: recomputed_metrics.get(key) for key in metric_keys},
            persisted_metrics,
            "TEST metrics",
        )
        recomputed_signal = signal_status(recomputed_metrics)
        if recomputed_signal != persisted_signal:
            errors.append("TEST signal status does not match independent recomputation.")
        validation_metrics = json.loads(
            (phase14_path / "overall_metrics.json").read_text(encoding="utf-8")
        )
        settings = load_final_test_settings(project_root or directory)
        recomputed_comparison = validation_test_comparison(
            validation_metrics,
            recomputed_metrics,
            moderate_ap_ratio=settings.moderate_ap_ratio,
            moderate_roc_drop=settings.moderate_roc_drop,
        )
        _compare_fields(
            errors, recomputed_comparison, persisted_comparison, "Validation-to-TEST comparison"
        )
        threshold_keys = (
            "tp",
            "fp",
            "tn",
            "fn",
            "precision",
            "recall",
            "specificity",
            "negative_predictive_value",
            "false_positive_rate",
            "false_negative_rate",
            "f1",
            "f2",
            "balanced_accuracy",
            "mcc",
            "predicted_positive_rate",
            "threshold",
        )
        _compare_fields(
            errors,
            {key: recomputed_metrics.get(key) for key in threshold_keys},
            persisted_threshold,
            "TEST threshold metrics",
        )
        _compare_fields(
            errors,
            {key: recomputed_metrics.get(key) for key in ("tp", "fp", "tn", "fn", "threshold")},
            persisted_confusion,
            "TEST confusion matrix",
        )
        recomputed_topk = topk_lift(ordered[KEY], y, p, tuple(settings.top_k))
        if recomputed_topk != persisted_topk:
            errors.append("TEST Top-K results do not match deterministic recomputation.")

        # Replay every persisted diagnostic from the frozen TEST scores and
        # labels.  This is intentionally separate from the runner's output
        # path: a valid hash cannot make a scientifically different table
        # acceptable.
        try:
            test_frame = resolved.test_features.sort_values(KEY, kind="mergesort").reset_index(
                drop=True
            )
            y_series = pd.Series(y, index=test_frame.index, dtype="int8")
            p_series = pd.Series(p, index=test_frame.index, dtype="float64")
            train_oof = train_oof_scores(resolved.phase13)
            reconstructed_deciles = risk_decile_metrics(train_oof, test_frame, y_series, p_series)
            errors.extend(
                _compare_frames(
                    reconstructed_deciles,
                    pd.read_parquet(directory / "test_risk_deciles.parquet"),
                    "TEST risk deciles",
                )
            )
            total_positive = max(int(y.sum()), 1)
            reconstructed_concentration = {
                "positive_share_d10": float(
                    reconstructed_deciles.loc[
                        reconstructed_deciles["decile"] == "D10", "positive_count"
                    ].sum()
                    / total_positive
                ),
                "positive_share_d10_d9": float(
                    reconstructed_deciles.loc[
                        reconstructed_deciles["decile"].isin(["D10", "D9"]), "positive_count"
                    ].sum()
                    / total_positive
                ),
                "positive_share_d10_d8": float(
                    reconstructed_deciles.loc[
                        reconstructed_deciles["decile"].isin(["D10", "D9", "D8"]),
                        "positive_count",
                    ].sum()
                    / total_positive
                ),
            }
            errors.extend(
                _compare_nested(
                    reconstructed_concentration,
                    _read_json(directory, "ranking_concentration_summary.json"),
                    "ranking_concentration_summary",
                )
            )

            compute = _read_json(directory, "compute_manifest.json")
            bootstrap_policy = plan.get("bootstrap_policy", {})
            expected_bootstrap_summary, expected_bootstrap_rows = stratified_bootstrap(
                y,
                p,
                threshold,
                replicates=int(bootstrap_policy.get("replicates", 0)),
                seed=int(bootstrap_policy.get("seed", settings.seed)),
                workers=max(1, int(compute.get("bootstrap_workers", 1))),
                confidence_level=float(
                    bootstrap_policy.get("confidence_level", settings.confidence_level)
                ),
            )
            errors.extend(
                _compare_nested(
                    expected_bootstrap_summary,
                    _read_json(directory, "test_bootstrap_summary.json"),
                    "test_bootstrap_summary",
                )
            )
            errors.extend(
                _compare_frames(
                    pd.DataFrame(expected_bootstrap_rows),
                    pd.read_parquet(directory / "test_bootstrap.parquet"),
                    "TEST bootstrap",
                )
            )

            reconstructed_reliability, reliability_summary = reliability_table(y, p)
            errors.extend(
                _compare_frames(
                    reconstructed_reliability,
                    pd.read_parquet(directory / "test_reliability.parquet"),
                    "TEST reliability",
                )
            )
            reliability_summary_path = directory / "test_reliability_summary.json"
            if reliability_summary_path.is_file():
                errors.extend(
                    _compare_nested(
                        reliability_summary,
                        _read_json(directory, "test_reliability_summary.json"),
                        "test_reliability_summary",
                    )
                )

            robust_settings = load_robustness_settings(resolved.root)
            reconstructed_temporal = temporal_metrics(
                test_frame,
                y_series,
                p_series,
                threshold,
                robust_settings,
                overall=recomputed_metrics,
            )
            errors.extend(
                _compare_frames(
                    reconstructed_temporal,
                    pd.read_parquet(directory / "test_temporal_metrics.parquet"),
                    "TEST temporal metrics",
                )
            )
            temporal_summary_path = directory / "test_temporal_summary.json"
            if temporal_summary_path.is_file():
                errors.extend(
                    _compare_nested(
                        {"row_count": len(reconstructed_temporal)},
                        _read_json(directory, "test_temporal_summary.json"),
                        "test_temporal_summary",
                    )
                )
            reconstructed_slices, reconstructed_slice_summary = evaluate_slices(
                definitions,
                test_frame,
                y_series,
                p_series,
                threshold,
                recomputed_metrics,
                robust_settings,
            )
            errors.extend(
                _compare_frames(
                    reconstructed_slices,
                    pd.read_parquet(directory / "test_slice_metrics.parquet"),
                    "TEST slice metrics",
                )
            )
            slice_summary_path = directory / "test_slice_summary.json"
            if slice_summary_path.is_file():
                errors.extend(
                    _compare_nested(
                        reconstructed_slice_summary,
                        _read_json(directory, "test_slice_summary.json"),
                        "test_slice_summary",
                    )
                )

            reconstructed_cohorts = error_cohorts(ordered[KEY], y, p, threshold)
            errors.extend(
                _compare_frames(
                    reconstructed_cohorts,
                    pd.read_parquet(directory / "test_error_cohorts.parquet"),
                    "TEST error cohorts",
                )
            )
            reconstructed_error_summary = {
                "row_count": int(len(reconstructed_cohorts)),
                "false_positive_count": int(
                    (reconstructed_cohorts["error_type"] == "FALSE_POSITIVE").sum()
                ),
                "false_negative_count": int(
                    (reconstructed_cohorts["error_type"] == "FALSE_NEGATIVE").sum()
                ),
                "true_positive_count": int(
                    (reconstructed_cohorts["error_type"] == "TRUE_POSITIVE").sum()
                ),
                "true_negative_count": int(
                    (reconstructed_cohorts["error_type"] == "TRUE_NEGATIVE").sum()
                ),
                "threshold": threshold,
            }
            context = build_error_context(test_frame, definitions, p_series, resolved.feature_names)
            reconstructed_profile = error_profile(reconstructed_cohorts, context)
            reconstructed_error_summary["profile_row_count"] = int(len(reconstructed_profile))
            errors.extend(
                _compare_nested(
                    reconstructed_error_summary,
                    _read_json(directory, "test_error_summary.json"),
                    "test_error_summary",
                )
            )
            false_negatives = reconstructed_cohorts.loc[
                reconstructed_cohorts["error_type"] == "FALSE_NEGATIVE"
            ]
            actual_positives = reconstructed_cohorts.loc[reconstructed_cohorts["target"] == 1]
            reconstructed_fn_summary = {
                "false_negative_count": int(len(false_negatives)),
                "actual_positive_count": int(len(actual_positives)),
                "false_negative_rate_among_actual_positives": float(
                    len(false_negatives) / len(actual_positives)
                )
                if len(actual_positives)
                else 0.0,
                "mean_false_negative_probability": float(false_negatives["probability"].mean())
                if len(false_negatives)
                else 0.0,
                "max_false_negative_probability": float(false_negatives["probability"].max())
                if len(false_negatives)
                else 0.0,
            }
            errors.extend(
                _compare_nested(
                    reconstructed_fn_summary,
                    _read_json(directory, "false_negative_summary.json"),
                    "false_negative_summary",
                )
            )
            optional_tables = (
                ("high_confidence_errors.parquet", high_confidence_errors(reconstructed_cohorts)),
                ("test_error_profile.parquet", reconstructed_profile),
            )
            for filename, reconstructed in optional_tables:
                path = directory / filename
                if path.is_file():
                    errors.extend(
                        _compare_frames(
                            reconstructed, pd.read_parquet(path), f"TEST {filename[:-8]}"
                        )
                    )

            scorer = prepare_scorer(
                resolved.phase13,
                threads=max(1, int(compute.get("catboost_inference_threads", 1))),
            )
            fresh_scorer = prepare_scorer(
                resolved.phase13,
                threads=max(1, int(compute.get("catboost_inference_threads", 1))),
            )
            reconstructed_invariance = prediction_invariance(
                test_frame,
                scorer,
                fresh_scorer=fresh_scorer,
                batch_sizes=tuple(
                    int(value)
                    for value in plan.get("invariance_policy", {}).get(
                        "batch_sizes", settings.batch_sizes
                    )
                ),
                seed=settings.seed,
                tolerance=tolerance,
            )
            errors.extend(
                _compare_nested(
                    reconstructed_invariance,
                    persisted_invariance,
                    "test_prediction_invariance",
                )
            )
            if reconstructed_invariance.get("valid") is not True:
                errors.append("TEST prediction invariance is invalid.")
        except Exception as exc:
            errors.append(f"Independent Phase 15 diagnostic replay failed: {exc}")
        derived_status = final_model_status(
            recomputed_signal,
            recomputed_comparison,
            provenance_valid=not errors,
            leakage_valid=bool(leakage.get("valid")),
            scoring_valid=max_delta <= tolerance
            and not any("prediction invariance" in item for item in errors),
            test_use_valid=not any(
                "TEST-use policy" in item or "TEST-use audit" in item for item in errors
            ),
            warnings=list(resolved.phase14_readiness.get("warnings", [])),
        )
        if persisted_status.get("final_model_status") != derived_status.get("final_model_status"):
            errors.append("Final model status does not match independent recomputation.")
        if persisted_manifest:
            if persisted_manifest.get("phase") != 15:
                errors.append("Phase 15 manifest has the wrong phase.")
            recorded_hashes = persisted_manifest.get("artifact_file_sha256", {})
            # validation.json is intentionally excluded: it contains the
            # manifest validation result and would otherwise create a hash
            # cycle (the final validation write necessarily changes its hash).
            actual_hashes = artifact_hashes(
                directory, exclude={"phase15_manifest.json", "validation.json"}
            )
            if recorded_hashes and recorded_hashes != actual_hashes:
                errors.append("Phase 15 artifact hash manifest drifted.")
        return {
            "phase": 15,
            "valid": not errors,
            "status": "PASS WITH WARNINGS"
            if warnings and not errors
            else ("PASS" if not errors else "BLOCKED"),
            "errors": sorted(set(errors)),
            "warnings": sorted(set(warnings)),
            "test_targets_accessed": True,
            "test_row_count": int(len(y)),
            "prediction_replay_max_probability_delta": max_delta,
            "independent_final_model_status": derived_status,
        }
    except (OSError, ValueError, KeyError, TypeError, Phase15InputError) as exc:
        errors.append(str(exc))
        return _blocked(errors, warnings)


__all__ = ["REQUIRED_ARTIFACTS", "validate_existing_phase15"]
