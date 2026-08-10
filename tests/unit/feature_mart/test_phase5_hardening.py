"""Behavior-focused coverage for Phase 5 configuration, lineage, and validation guards."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from warranty_analytics_model.database.schema_contract import load_schema_contract
from warranty_analytics_model.feature_mart.common import (
    assert_pair_unique,
    assert_unique_key,
    deterministic_sort,
    empty_with_columns,
    history_diagnostics,
    merge_many_to_one,
)
from warranty_analytics_model.feature_mart.component_history import (
    build_component_installation_history,
)
from warranty_analytics_model.feature_mart.config import (
    load_feature_mart_settings,
    resolve_mart_output_root,
    resolve_mart_report_root,
    technical_settings_dict,
)
from warranty_analytics_model.feature_mart.direct_snapshot import build_direct_snapshot
from warranty_analytics_model.feature_mart.extraction_plan import (
    ExtractionPlan,
    explicit_count_sql,
    explicit_select_sql,
    plan_columns,
    quote_identifier,
)
from warranty_analytics_model.feature_mart.lineage import (
    build_group_membership,
    build_safe_scenario_fingerprint,
    canonical_value,
)
from warranty_analytics_model.feature_mart.mart_contract import load_mart_contract
from warranty_analytics_model.feature_mart.models import FeatureMartError, FeatureMartSettings
from warranty_analytics_model.feature_mart.validation import (
    validate_artifact_integrity,
    validate_frames,
)
from warranty_analytics_model.policy.loader import load_phase4_contracts

from .test_phase5_artifacts import _all_frames
from .test_phase5_bridges import _claims, _history_sources
from .test_phase5_snapshot import _direct_frames

ROOT = Path(__file__).resolve().parents[3]


def test_feature_mart_settings_resolve_and_serialize() -> None:
    """Technical settings resolve without changing business policy."""

    settings = load_feature_mart_settings(ROOT)
    assert settings.serialization_format == "parquet"
    assert (
        resolve_mart_output_root(ROOT, settings, None)
        == (ROOT / settings.output_directory).resolve()
    )
    assert (
        resolve_mart_report_root(ROOT, settings, None)
        == (ROOT / settings.report_directory).resolve()
    )
    absolute = (ROOT / "tmp-feature-mart").resolve()
    assert resolve_mart_output_root(ROOT, settings, absolute) == absolute
    payload = technical_settings_dict(settings)
    assert payload["compression"] == "snappy"
    assert payload["validate_after_build"] is True


def test_feature_mart_settings_fail_closed(tmp_path: Path) -> None:
    """Missing, malformed, and invalid technical YAML is rejected."""

    missing = tmp_path / "missing.yaml"
    with pytest.raises(FeatureMartError, match="missing"):
        load_feature_mart_settings(ROOT, path=missing)

    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("feature_mart: [not-a-mapping]", encoding="utf-8")
    with pytest.raises(FeatureMartError, match="mapping"):
        load_feature_mart_settings(ROOT, path=malformed)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("feature_mart:\n  compression: invalid\n", encoding="utf-8")
    with pytest.raises(FeatureMartError, match="Invalid"):
        load_feature_mart_settings(ROOT, path=invalid)


def test_common_cardinality_and_diagnostic_guards() -> None:
    """Shared bridge helpers fail closed and preserve missing-history diagnostics."""

    with pytest.raises(FeatureMartError, match="missing required key"):
        assert_unique_key(pd.DataFrame({"other": [1]}), "key", "dimension")
    with pytest.raises(FeatureMartError, match="null key"):
        assert_unique_key(pd.DataFrame({"key": [None]}), "key", "dimension")
    with pytest.raises(FeatureMartError, match="duplicate key"):
        assert_unique_key(pd.DataFrame({"key": [1, 1]}), "key", "dimension")

    left = pd.DataFrame({"key": [1, 2]})
    right = pd.DataFrame({"key": [1], "value": ["one"]})
    merged, diagnostics = merge_many_to_one(left, right, on="key", label="example")
    assert len(merged) == 2
    assert diagnostics["unmatched_rows"] == 1
    with pytest.raises(FeatureMartError, match="join keys missing"):
        merge_many_to_one(left, right, on="missing", label="example")
    with pytest.raises(FeatureMartError, match="dimension keys"):
        merge_many_to_one(left, pd.DataFrame({"key": [1, 1]}), on="key", label="example")

    with pytest.raises(FeatureMartError, match="bridge key"):
        assert_pair_unique(pd.DataFrame({"claim": [1]}), ["claim", "event"], "bridge")
    with pytest.raises(FeatureMartError, match="duplicate bridge pairs"):
        assert_pair_unique(
            pd.DataFrame({"claim": [1, 1], "event": [2, 2]}),
            ["claim", "event"],
            "bridge",
        )

    eligible = pd.DataFrame({"warranty_claim_key": [1, 2]})
    diagnostics = history_diagnostics(
        eligible,
        pd.DataFrame({"current_warranty_claim_key": [1], "event": [9]}),
    )
    assert diagnostics["claims_with_history"] == 1
    assert diagnostics["claims_without_history"] == 1
    assert empty_with_columns(["a", "a", "b"]).columns.tolist() == ["a", "b"]
    assert deterministic_sort(pd.DataFrame({"value": [2, 1]}), ["missing"]).equals(
        pd.DataFrame({"value": [2, 1]})
    )
    assert deterministic_sort(pd.DataFrame(columns=["value"]), ["value"]).empty


def test_explicit_sql_helpers_fail_closed() -> None:
    """Extraction SQL is schema-qualified, explicit, and strictly quoted."""

    assert quote_identifier("valid_name") == "[valid_name]"
    with pytest.raises(FeatureMartError, match="Unsafe SQL"):
        quote_identifier("invalid-name")
    with pytest.raises(FeatureMartError, match="schema-qualified"):
        explicit_select_sql("fact", ["key"])
    with pytest.raises(FeatureMartError, match="No columns"):
        explicit_select_sql("dbo.fact", [])
    assert explicit_count_sql("dbo.fact") == (
        "SELECT COUNT_BIG(*) AS [exact_row_count] FROM [dbo].[fact]"
    )
    with pytest.raises(FeatureMartError, match="schema-qualified"):
        explicit_count_sql("fact")
    with pytest.raises(FeatureMartError, match="not in"):
        plan_columns(ExtractionPlan({}, ()), "dbo.missing")


def test_lineage_values_fingerprint_and_group_metadata() -> None:
    """Lineage is deterministic, target-free, and explicitly non-model."""

    assert canonical_value(None) == "<NULL>"
    assert canonical_value(pd.NA) == "<NULL>"
    assert canonical_value(float("nan")) == "<NULL>"
    assert canonical_value(pd.Timestamp("2026-01-02")) == "2026-01-02T00:00:00"
    assert canonical_value(datetime(2026, 1, 2)) == "2026-01-02T00:00:00"
    assert canonical_value(date(2026, 1, 2)) == "2026-01-02"
    assert canonical_value(Decimal("1.20")) == "1.20"
    assert canonical_value(True) == "true"

    source = pd.DataFrame({"safe": ["same", "other"]})
    fingerprints = build_safe_scenario_fingerprint(source, ["safe"])
    assert fingerprints.nunique() == 2
    with pytest.raises(FeatureMartError, match="missing"):
        build_safe_scenario_fingerprint(source, ["missing"])
    with pytest.raises(FeatureMartError, match="cannot be empty"):
        build_safe_scenario_fingerprint(source, [])

    contract, _ = load_mart_contract(ROOT)
    direct = build_direct_snapshot(_direct_frames(), contract).snapshot
    component, _, _ = build_component_installation_history(
        _claims(), _history_sources()["installations"], _history_sources()["components"]
    )
    component.loc[0, ["supplier_key", "component_lot_no", "production_batch_id"]] = None
    groups = build_group_membership(direct, component, contract)
    assert "safe_scenario_fingerprint" in set(groups["group_type"])
    assert groups["is_model_feature"].eq(False).all()
    assert {"historical_supplier", "historical_component_lot"} <= set(groups["group_type"])

    bad_rules = dict(contract.safety_rules)
    bad_rules["safe_scenario_fingerprint_input_columns"] = ["target__high_cost_claim_flag"]
    bad_contract = contract.model_copy(update={"safety_rules": bad_rules})
    with pytest.raises(FeatureMartError, match="Target columns"):
        build_group_membership(direct, component, bad_contract)
    with pytest.raises(FeatureMartError, match="warranty_claim_key"):
        build_group_membership(pd.DataFrame({"other": [1]}), component, contract)


def test_validation_blocks_current_service_event_in_memory(tmp_path: Path) -> None:
    """The final validator reports current service contamination even with prior history."""

    pytest.importorskip("pyarrow")
    from warranty_analytics_model.feature_mart.runner import build_feature_mart_from_frames

    frames = _all_frames()
    result = build_feature_mart_from_frames(
        frames=frames,
        source_row_counts={name: len(frame) for name, frame in frames.items()},
        root=ROOT,
        settings=FeatureMartSettings(),
        environment="test",
        source_database="warranty_analytics",
        output_root=tmp_path / "artifacts",
        report_root=None,
        no_report=True,
        run_id="validation-hardening",
    )
    mart_dir = Path(result.run_directory)
    loaded, artifact_frames = validate_artifact_integrity(mart_dir)
    snapshot = artifact_frames["claim_snapshot"]
    current_claim = int(snapshot.loc[0, "warranty_claim_key"])
    current_event = int(snapshot.loc[0, "lineage__current_service_event_key"])
    contaminated = artifact_frames["service_history"].iloc[[0]].copy()
    contaminated["current_warranty_claim_key"] = current_claim
    contaminated["service_event_key"] = current_event
    contaminated["service__service_date"] = snapshot.loc[0, "claim__claim_date"] - pd.Timedelta(
        days=1
    )
    artifact_frames["service_history"] = pd.concat(
        [artifact_frames["service_history"], contaminated],
        ignore_index=True,
    )
    _schema, schema_checksum = load_schema_contract(ROOT)
    phase4 = load_phase4_contracts(ROOT)
    contract, mart_checksum = load_mart_contract(ROOT)
    assert schema_checksum == contract.schema_contract_checksum
    validation = validate_frames(
        frames=artifact_frames,
        eligibility={"eligible_claims": loaded["manifest"]["eligible_claims"]},
        contract=contract,
        phase4_bundle=phase4,
        column_manifest=loaded["column_manifest"],
        history_coverage=loaded["manifest"]["history_coverage"],
        direct_join_validation=loaded["manifest"]["direct_join_validation"],
        source_target_values=pd.Series([0.0, 1.0]),
    )
    assert validation["status"] == "BLOCKED"
    assert validation["temporal"]["current_service_event_violations"] == 1
    assert mart_checksum == loaded["manifest"]["mart_contract_checksum"]

    bad_frames = {name: frame.copy() for name, frame in artifact_frames.items()}
    bad_frames["claim_snapshot"].loc[0, "target__high_cost_claim_flag"] = 2
    bad_frames["claim_snapshot"].loc[1, "warranty_claim_key"] = current_claim
    unknown_telemetry = bad_frames["telemetry_history"].iloc[[0]].copy()
    unknown_telemetry["current_warranty_claim_key"] = 999999
    bad_frames["telemetry_history"] = pd.concat(
        [bad_frames["telemetry_history"], unknown_telemetry],
        ignore_index=True,
    )
    bad_frames["claim_group_membership"].loc[0, "is_model_feature"] = True
    bad_manifest = [
        *loaded["column_manifest"],
        {
            "artifact_name": "component_installation_history",
            "source_table": "dbo.fact_warranty_claim",
            "source_column": "causal_component_key",
            "is_model_feature": False,
        },
    ]
    bad_validation = validate_frames(
        frames=bad_frames,
        eligibility={"eligible_claims": loaded["manifest"]["eligible_claims"]},
        contract=contract,
        phase4_bundle=phase4,
        column_manifest=bad_manifest,
        history_coverage={"telemetry": {"eligible_claims": 0}},
        direct_join_validation={"dimension": {"multiplication_count": 1}},
    )
    assert bad_validation["status"] == "BLOCKED"
    assert any("target" in error.lower() for error in bad_validation["errors"])
    assert any("outside" in error.lower() for error in bad_validation["errors"])
    assert bad_validation["temporal"]["component_history_causal_component_dependency"] == 1
