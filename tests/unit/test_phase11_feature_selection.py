"""Unit coverage for the Phase 11 safety and deterministic selection helpers."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from warranty_analytics_model.baseline_model.models import FeatureSetSpec
from warranty_analytics_model.feature_selection import config as phase11_config
from warranty_analytics_model.feature_selection import (
    runner as phase11_runner,
)
from warranty_analytics_model.feature_selection import (
    validation as phase11_validation,
)
from warranty_analytics_model.feature_selection.checkpoint import (
    load_valid_checkpoint,
    write_checkpoint,
)
from warranty_analytics_model.feature_selection.config import load_feature_selection_settings
from warranty_analytics_model.feature_selection.contract import (
    load_feature_selection_contract,
    validate_feature_selection_contract,
)
from warranty_analytics_model.feature_selection.grouping import (
    build_feature_group_manifest,
    validate_group_membership,
)
from warranty_analytics_model.feature_selection.planner import build_compute_plan
from warranty_analytics_model.feature_selection.selection import (
    feature_list_sha256,
    feature_set_sha256,
    generate_candidates,
    replacement_decision,
    select_candidate,
    subset_feature_set,
)


def _parent() -> FeatureSetSpec:
    names = tuple(f"f_{index}" for index in range(40))
    return FeatureSetSpec(
        experiment_id="E1",
        feature_names=names,
        numeric_features=names,
        categorical_features=(),
        boolean_features=(),
        text_features=(),
        phase7_core_count=40,
        phase7_extended_count=0,
        phase8_lexical_count=0,
        phase8_text_count=0,
        feature_set_sha256=feature_set_sha256("E1", names),
    )


def _families(
    parent: FeatureSetSpec,
) -> tuple[dict[str, tuple[str, ...]], dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    phase7 = {
        name: {
            "is_model_feature": True,
            "is_control": False,
            "target_dependent": False,
            "feature_type": "numeric",
            "family": "telemetry" if index % 2 else "maintenance",
            "tier": "CORE",
            "value_sources": ["history"],
        }
        for index, name in enumerate(parent.feature_names)
    }
    return {"T1": parent.feature_names}, phase7, {}


def test_contract_config_and_compute_plan() -> None:
    settings = load_feature_selection_settings()
    assert settings.tracks == ("T1", "T3")
    assert settings.maximum_candidate_subsets_per_track == 8
    assert validate_feature_selection_contract()["valid"] is True
    plan = build_compute_plan(settings, logical_processors=22)
    assert plan.worker_count == 2
    assert plan.threads_per_worker == 10
    assert plan.maximum_concurrent_threads <= 22
    small = build_compute_plan(settings, logical_processors=3)
    assert small.worker_count <= 3
    assert small.maximum_concurrent_threads <= 3


def test_grouping_is_complete_and_deterministic() -> None:
    parent = _parent()
    parent_features, phase7, phase8 = _families(parent)
    manifest, membership = build_feature_group_manifest(parent_features, phase7, phase8)
    assert manifest["feature_count"] == 40
    assert manifest["family_count"] == 2
    validate_group_membership(membership, set(parent.feature_names))
    shuffled = membership.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert (
        build_feature_group_manifest(parent_features, phase7, phase8)[0]["membership_sha256"]
        == manifest["membership_sha256"]
    )
    with pytest.raises(ValueError, match="cover"):
        validate_group_membership(shuffled.iloc[:-1], set(parent.feature_names))


def test_subset_candidates_selection_and_replacement() -> None:
    parent = _parent()
    settings = load_feature_selection_settings()
    family_by_feature = {
        name: ("telemetry_history" if index % 2 else "maintenance_history")
        for index, name in enumerate(parent.feature_names)
    }
    stability = [
        {
            "feature": name,
            "top_50_percent_fold_count": 3 if index < 30 else 0,
            "top_25_percent_fold_count": 1 if index < 20 else 0,
        }
        for index, name in enumerate(parent.feature_names)
    ]
    ablation = [
        {"family": "maintenance_history", "delta_ap_vs_parent": 0.0001},
        {"family": "telemetry_history", "delta_ap_vs_parent": 0.01},
    ]
    candidates = generate_candidates(
        "T1", parent, list(parent.feature_names), ablation, stability, family_by_feature, settings
    )
    assert 1 <= len(candidates) <= 8
    assert all(set(item.feature_list).issubset(parent.feature_names) for item in candidates)
    subset = subset_feature_set(parent, candidates[-1].feature_list, candidates[-1].candidate_id)
    assert subset.feature_set_sha256 == candidates[-1].feature_set_sha256
    assert feature_list_sha256(subset.feature_names)
    rows = [
        {
            "candidate_id": "full",
            "feature_count": 40,
            "mean_average_precision": 0.10,
            "min_average_precision": 0.08,
            "std_average_precision": 0.01,
            "mean_roc_auc": 0.70,
            "mean_log_loss": 0.13,
        },
        {
            "candidate_id": "small",
            "feature_count": 20,
            "mean_average_precision": 0.0999,
            "min_average_precision": 0.079,
            "std_average_precision": 0.01,
            "mean_roc_auc": 0.699,
            "mean_log_loss": 0.131,
        },
    ]
    selected, trace = select_candidate(rows, settings)
    assert selected["candidate_id"] == "small"
    assert trace["selected_candidate_id"] == "small"
    assert replacement_decision(
        {"average_precision": 0.10, "roc_auc": 0.70, "log_loss": 0.13, "feature_count": 40},
        {"average_precision": 0.101, "roc_auc": 0.70, "log_loss": 0.13, "feature_count": 20},
        settings,
    )["replace_parent"]


def test_checkpoint_round_trip_and_stale_rejection(tmp_path: Path) -> None:
    payload = {
        "experiment_id": "P11_T1_TOP_070",
        "experiment_spec_sha256": "spec",
        "track": "T1",
        "feature_set_sha256": "features",
        "parameter_sha256": "params",
        "fold_id": 1,
        "metrics": {"mean_average_precision": 0.1},
        "training_seconds": 1.0,
        "completed_at": "2026-08-16T00:00:00Z",
    }
    write_checkpoint(tmp_path, payload)
    assert (
        load_valid_checkpoint(
            tmp_path,
            experiment_id="P11_T1_TOP_070",
            experiment_spec_sha256="spec",
            track="T1",
            feature_set_sha256="features",
            parameter_sha256="params",
            fold_id=1,
        )
        == payload
    )


def test_phase11_fail_closed_guards_and_payloads(tmp_path: Path) -> None:
    settings = load_feature_selection_settings()
    payload = phase11_config.settings_payload(settings)
    assert payload["compute"]["preferred_max_workers"] == 2
    for raw in (True, "not-a-number"):
        with pytest.raises(ValueError):
            phase11_config._number(raw, "test")
    for raw in (False, 0, "1"):
        with pytest.raises(ValueError):
            phase11_config._positive_int(raw, "test")
    with pytest.raises(ValueError):
        build_compute_plan(settings, logical_processors=0)
    with pytest.raises(ValueError):
        build_compute_plan(settings, logical_processors=4, max_workers=0)
    with pytest.raises(ValueError):
        build_compute_plan(settings, logical_processors=4, single_fit_threads=0)
    isolated_root = tmp_path / "isolated_project"
    (isolated_root / "configs").mkdir(parents=True)
    (isolated_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    invalid_contract = validate_feature_selection_contract(isolated_root)
    assert invalid_contract["valid"] is False
    with pytest.raises(ValueError):
        load_feature_selection_contract(isolated_root)

    parent = _parent()
    with pytest.raises(ValueError, match="non-empty"):
        subset_feature_set(parent, [], "empty")
    with pytest.raises(ValueError, match="outside"):
        subset_feature_set(parent, ["not-a-parent"], "added")
    with pytest.raises(ValueError, match="permutation"):
        generate_candidates("T1", parent, list(parent.feature_names[:-1]), [], [], {}, settings)
    with pytest.raises(ValueError, match="empty"):
        select_candidate([], settings)
    fallback = replacement_decision(
        {"average_precision": 0.2, "roc_auc": 0.8, "log_loss": 0.1, "feature_count": 40},
        {"average_precision": 0.1, "roc_auc": 0.5, "log_loss": 0.2, "feature_count": 20},
        settings,
    )
    assert fallback["reason"] == "FALLBACK_PARENT"

    missing = pd.DataFrame({"feature": ["f_0"]})
    with pytest.raises(ValueError, match="schema"):
        validate_group_membership(missing, {"f_0"})
    phase7 = {
        "f_0": {
            "is_model_feature": True,
            "is_control": False,
            "target_dependent": False,
            "feature_type": "numeric",
        }
    }
    with pytest.raises(ValueError, match="family"):
        build_feature_group_manifest({"T1": ("f_0",)}, phase7, {})
    phase8 = {
        "lex": {
            "is_model_feature": True,
            "is_control": False,
            "target_dependent": False,
            "feature_type": "numeric",
            "value_sources": ["prior_failure__failure_description"],
        }
    }
    manifest, membership = build_feature_group_manifest({"T1": ("lex",)}, {}, phase8)
    assert manifest["families"] == {"historical_lexical_features": 1}
    assert membership.iloc[0]["family"] == "historical_lexical_features"
    with pytest.raises(ValueError, match="source"):
        build_feature_group_manifest(
            {"T1": ("lex",)},
            {},
            {"lex": {**phase8["lex"], "value_sources": ["unauthorized"]}},
        )

    incomplete = tmp_path / "checkpoints" / "exp" / "fold_1.json"
    incomplete.parent.mkdir(parents=True)
    incomplete.write_text(json.dumps({"experiment_id": "exp"}), encoding="utf-8")
    with pytest.raises(ValueError, match="[Ss]tale"):
        load_valid_checkpoint(
            tmp_path,
            experiment_id="exp",
            experiment_spec_sha256="s",
            track="T1",
            feature_set_sha256="f",
            parameter_sha256="p",
            fold_id=1,
        )
    path = tmp_path / "checkpoints" / "P11_T1_TOP_070" / "fold_1.json"
    path.parent.mkdir(parents=True)
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt"):
        load_valid_checkpoint(
            tmp_path,
            experiment_id="P11_T1_TOP_070",
            experiment_spec_sha256="spec",
            track="T1",
            feature_set_sha256="features",
            parameter_sha256="params",
            fold_id=1,
        )


def test_phase11_runner_and_validator_helpers(tmp_path: Path) -> None:
    assert len(phase11_runner.phase11_run_id()) == 16
    assert phase11_runner._resolve(tmp_path, None, "child") == (tmp_path / "child").resolve()
    absolute = (tmp_path / "absolute").resolve()
    assert phase11_runner._resolve(tmp_path, absolute, "ignored") == absolute

    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({"ok": True}), encoding="utf-8")
    assert phase11_runner._read_json(valid) == {"ok": True}
    with pytest.raises(ValueError, match="Invalid JSON"):
        phase11_runner._read_json(tmp_path / "missing.json")
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        phase11_runner._read_json(non_object)

    class Inputs:
        development = pd.DataFrame(
            {
                "warranty_claim_key": [2, 1, 3],
                "split": ["TRAIN", "TRAIN", "VALIDATION"],
            }
        )

    assert phase11_runner._train_rows(Inputs()).warranty_claim_key.tolist() == [1, 2]
    aggregate = phase11_runner._aggregate(
        [
            {
                "fold_id": 1,
                "average_precision": 0.1,
                "roc_auc": 0.7,
                "log_loss": 0.2,
                "brier_score": 0.05,
                "training_seconds": 1.5,
            },
            {
                "fold_id": 2,
                "average_precision": 0.2,
                "roc_auc": 0.8,
                "log_loss": 0.1,
                "brier_score": 0.03,
                "training_seconds": 2.5,
            },
        ],
        20,
        0.5,
    )
    assert aggregate["feature_count"] == 20
    assert aggregate["training_seconds"] == 4.0
    errors: list[str] = []
    phase11_validation._error(errors, "expected")
    assert errors == ["expected"]


def test_phase11_configuration_rejects_policy_drift(tmp_path: Path) -> None:
    isolated_root = tmp_path / "config_project"
    config_dir = isolated_root / "configs"
    config_dir.mkdir(parents=True)
    (isolated_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    config_path = config_dir / "feature_selection_ablation.yaml"
    source = yaml.safe_load(
        Path("configs/feature_selection_ablation.yaml").read_text(encoding="utf-8")
    )

    def rejects(mutator) -> None:
        payload = deepcopy(source)
        mutator(payload["phase11_feature_selection"])
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            load_feature_selection_settings(isolated_root)

    rejects(lambda raw: raw.__setitem__("tracks", ["T1"]))
    rejects(lambda raw: raw.__setitem__("primary_metric", "roc_auc"))
    rejects(lambda raw: raw.__setitem__("candidate_fractions", []))
    rejects(lambda raw: raw.__setitem__("candidate_fractions", [1.0, 0.8]))
    rejects(lambda raw: raw.__setitem__("feature_importance", {}))
    rejects(lambda raw: raw.__setitem__("simpler_candidate_selection", None))
    rejects(lambda raw: raw.__setitem__("compute", None))
    rejects(lambda raw: raw.__setitem__("maximum_candidate_subsets_per_track", 9))

    config_path.unlink()
    with pytest.raises(ValueError, match="Could not read"):
        load_feature_selection_settings(isolated_root)

