"""Blocking integrity, temporal, and leakage validation for Phase 5 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..database.schema_contract import load_schema_contract
from ..paths import discover_repository_root
from ..policy.loader import load_phase4_contracts
from .common import as_datetime, assert_pair_unique
from .lineage import build_safe_scenario_fingerprint
from .manifest import content_sha256, sha256_file
from .mart_contract import (
    leakage_rule_matches,
    load_mart_contract,
    validate_mart_contract,
)
from .models import FeatureMartError, MartContract

BRIDGE_ARTIFACTS = (
    "telemetry_history",
    "maintenance_history",
    "service_history",
    "component_installation_history",
    "prior_claim_history",
    "repair_history_index",
    "claim_group_membership",
)


def _is_artifact(entry: dict[str, Any], artifact_name: str) -> bool:
    """Accept both the stable artifact name and its manifest-relative path."""

    if entry.get("artifact_name") == artifact_name:
        return True
    artifact = str(entry.get("artifact", ""))
    return artifact == artifact_name or artifact.rsplit("/", 1)[-1].startswith(artifact_name)


def _model_columns(column_manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for entry in column_manifest if entry.get("is_model_feature") is True]


def _claim_dates(snapshot: pd.DataFrame) -> pd.Series:
    if "warranty_claim_key" not in snapshot or "claim__claim_date" not in snapshot:
        raise FeatureMartError("Snapshot must contain warranty_claim_key and claim__claim_date.")
    snapshot = snapshot.drop_duplicates("warranty_claim_key", keep="first")
    return pd.Series(
        as_datetime(snapshot["claim__claim_date"]).to_numpy(),
        index=snapshot["warranty_claim_key"].tolist(),
    )


def _temporal_violation_count(
    bridge: pd.DataFrame,
    claim_dates: pd.Series,
    event_column: str,
    *,
    completed_month: bool = False,
) -> int:
    if bridge.empty:
        return 0
    dates = claim_dates.reindex(bridge["current_warranty_claim_key"]).reset_index(drop=True)
    events = as_datetime(bridge[event_column]).reset_index(drop=True)
    if completed_month:
        events = events + pd.offsets.MonthEnd(0)
    return int((events >= dates).sum())


def _safe_model_feature_scan(
    column_manifest: list[dict[str, Any]],
    mart_contract: MartContract,
    phase4_bundle: Any,
) -> dict[str, Any]:
    model_entries = _model_columns(column_manifest)
    blacklist = [rule.field for rule in phase4_bundle.leakage.hard_blacklist]
    identifiers = set(phase4_bundle.leakage.identifier_fields)
    prohibited: list[str] = []
    confirmation: list[str] = []
    restricted: list[str] = []
    identifier: list[str] = []
    target_leakage: list[str] = []
    wildcard_violations: list[str] = []
    for entry in model_entries:
        source = f"{entry.get('source_table')}.{entry.get('source_column')}"
        policy = str(entry.get("policy"))
        if policy == "PROHIBITED":
            prohibited.append(source)
        if policy == "REQUIRES_CONFIRMATION":
            confirmation.append(source)
        if policy == "RESTRICTED_EXPERIMENTAL":
            restricted.append(source)
        if source in identifiers or source.rsplit(".", 1)[-1] in {
            "warranty_claim_key",
            "claim_id",
            "truck_key",
            "vin",
            "service_event_key",
            "component_serial_no",
            "engine_serial_no",
            "transmission_serial_no",
            "technician_id",
            "inspector_id",
            "customer_name",
            "supplier_key",
            "service_center_key",
        }:
            identifier.append(source)
        if entry.get("is_target") or "high_cost_claim_flag" in source:
            target_leakage.append(source)
        if any(leakage_rule_matches(source, rule) for rule in blacklist):
            wildcard_violations.append(source)
    return {
        "model_feature_count": len(model_entries),
        "prohibited_model_fields": sorted(set(prohibited)),
        "confirmation_model_fields": sorted(set(confirmation)),
        "restricted_tier_a_fields": sorted(set(restricted)),
        "identifier_model_fields": sorted(set(identifier)),
        "target_feature_leakage": sorted(set(target_leakage)),
        "wildcard_leakage_violations": sorted(set(wildcard_violations)),
        "valid": not any(
            (prohibited, confirmation, restricted, identifier, target_leakage, wildcard_violations)
        ),
        "fingerprint_input_columns": mart_contract.safety_rules.get(
            "safe_scenario_fingerprint_input_columns", []
        ),
    }


def validate_frames(
    *,
    frames: dict[str, pd.DataFrame],
    eligibility: dict[str, Any],
    contract: MartContract,
    phase4_bundle: Any,
    column_manifest: list[dict[str, Any]],
    history_coverage: dict[str, dict[str, int | float]],
    direct_join_validation: dict[str, Any],
    source_target_values: pd.Series | None = None,
) -> dict[str, Any]:
    """Validate in-memory build frames before a run is marked complete."""

    errors: list[str] = []
    warnings: list[str] = []
    snapshot = frames["claim_snapshot"]
    expected_rows = int(eligibility.get("eligible_claims", -1))
    target_column = str(contract.target["output_column"])
    if len(snapshot) != expected_rows:
        errors.append(f"Snapshot rows {len(snapshot)} != eligible claims {expected_rows}.")
    if snapshot["warranty_claim_key"].isna().any():
        errors.append("Snapshot contains null warranty_claim_key values.")
    if snapshot["warranty_claim_key"].duplicated().any():
        errors.append("Snapshot warranty_claim_key values are not unique.")
    target = pd.to_numeric(snapshot[target_column], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        errors.append("Snapshot target is null or outside the approved binary set {0, 1}.")
    if source_target_values is not None:
        expected = pd.to_numeric(source_target_values, errors="coerce").reset_index(drop=True)
        actual = target.reset_index(drop=True)
        expected = expected.astype("Int64")
        actual = actual.astype("Int64")
        if not expected.equals(actual):
            errors.append("Snapshot target values do not exactly match the eligible source claims.")

    safety = _safe_model_feature_scan(column_manifest, contract, phase4_bundle)
    if not safety["valid"]:
        errors.append("Final model-feature leakage scan failed.")
    snapshot_model = [
        entry
        for entry in column_manifest
        if _is_artifact(entry, "claim_snapshot") and entry.get("is_model_feature") is True
    ]
    if any(entry.get("policy") != "ALLOW_BASELINE_POC" for entry in snapshot_model):
        errors.append("Claim snapshot contains a non-baseline model feature.")
    direct_outputs = {
        mapping.output_column
        for mapping in contract.direct_feature_mappings
        if mapping.mapping_status == "MATERIALIZED"
    }
    missing_direct_outputs = sorted(direct_outputs - set(snapshot.columns))
    if missing_direct_outputs:
        errors.append(
            f"Materialized direct fields are missing: {', '.join(missing_direct_outputs)}"
        )
    if any(item.get("multiplication_count", 0) != 0 for item in direct_join_validation.values()):
        errors.append("A direct dimension join multiplied claim rows.")

    unique_snapshot = snapshot.drop_duplicates("warranty_claim_key", keep="first")
    claim_dates = _claim_dates(unique_snapshot)
    temporal: dict[str, int] = {
        "same_day_violations": 0,
        "future_history_violations": 0,
        "claim_month_telemetry_violations": 0,
        "current_service_event_violations": 0,
        "current_repair_line_violations": 0,
        "component_history_causal_component_dependency": 0,
    }
    telemetry = frames["telemetry_history"]
    maintenance = frames["maintenance_history"]
    service = frames["service_history"]
    component = frames["component_installation_history"]
    prior = frames["prior_claim_history"]
    repair = frames["repair_history_index"]
    for name, bridge in (
        ("telemetry_history", telemetry),
        ("maintenance_history", maintenance),
        ("service_history", service),
        ("component_installation_history", component),
        ("prior_claim_history", prior),
        ("repair_history_index", repair),
    ):
        if not bridge.empty:
            unknown_claims = set(bridge["current_warranty_claim_key"]) - set(
                snapshot["warranty_claim_key"]
            )
            if unknown_claims:
                errors.append(f"{name} contains claims outside the eligible snapshot.")
    if not telemetry.empty:
        temporal["claim_month_telemetry_violations"] = _temporal_violation_count(
            telemetry,
            claim_dates,
            "telemetry__month_start_date",
            completed_month=True,
        )
        assert_pair_unique(
            telemetry,
            ["current_warranty_claim_key", "telemetry_month_key"],
            "telemetry",
        )
    if not maintenance.empty:
        temporal["same_day_violations"] += _temporal_violation_count(
            maintenance,
            claim_dates,
            "maintenance__maintenance_date",
        )
        assert_pair_unique(
            maintenance,
            ["current_warranty_claim_key", "maintenance_event_key"],
            "maintenance",
        )
    if not service.empty:
        temporal["same_day_violations"] += _temporal_violation_count(
            service,
            claim_dates,
            "service__service_date",
        )
        current_service = (
            unique_snapshot.set_index("warranty_claim_key")
            .reindex(service["current_warranty_claim_key"])["lineage__current_service_event_key"]
            .reset_index(drop=True)
            if "lineage__current_service_event_key" in snapshot
            else None
        )
        if current_service is not None:
            temporal["current_service_event_violations"] = int(
                (service["service_event_key"].reset_index(drop=True) == current_service).sum()
            )
        assert_pair_unique(service, ["current_warranty_claim_key", "service_event_key"], "service")
    if not component.empty:
        temporal["same_day_violations"] += _temporal_violation_count(
            component,
            claim_dates,
            "component_installation__installed_date",
        )
        assert_pair_unique(
            component,
            ["current_warranty_claim_key", "installation_key"],
            "component",
        )
    if not prior.empty:
        temporal["same_day_violations"] += _temporal_violation_count(
            prior,
            claim_dates,
            "prior_claim__claim_date",
        )
        assert_pair_unique(
            prior,
            ["current_warranty_claim_key", "prior_warranty_claim_key"],
            "prior claim",
        )
    if not repair.empty:
        temporal["current_repair_line_violations"] = int(
            (
                repair["current_warranty_claim_key"].reset_index(drop=True)
                == repair["prior_warranty_claim_key"].reset_index(drop=True)
            ).sum()
        )
        assert_pair_unique(repair, ["current_warranty_claim_key", "repair_line_key"], "repair")
    if any(value != 0 for value in temporal.values()):
        errors.append("One or more Phase 5 temporal/current-event leakage checks failed.")
    if any(
        source == "dbo.fact_warranty_claim.causal_component_key"
        for entry in column_manifest
        for source in [f"{entry.get('source_table')}.{entry.get('source_column')}"]
        if _is_artifact(entry, "component_installation_history")
    ):
        temporal["component_history_causal_component_dependency"] = 1
        errors.append("Component history depends on current causal_component_key.")

    group = frames["claim_group_membership"]
    if not group.empty and (
        "is_model_feature" not in group
        or group["is_model_feature"].isna().any()
        or group["is_model_feature"].eq(True).any()
    ):
        errors.append("Group membership artifact contains model-feature metadata set to true.")
    fingerprint_inputs = safety["fingerprint_input_columns"]
    try:
        fingerprint = build_safe_scenario_fingerprint(snapshot, fingerprint_inputs)
        if fingerprint.isna().any() or len(fingerprint) != len(snapshot):
            errors.append("Safe scenario fingerprint is incomplete.")
    except FeatureMartError as exc:
        errors.append(str(exc))

    for bridge_name, coverage in history_coverage.items():
        if int(coverage.get("eligible_claims", expected_rows)) != expected_rows:
            errors.append(f"History coverage eligible claim count is inconsistent: {bridge_name}")
    if phase4_bundle.target.development_status.business_target_definition_confirmed is False:
        warnings.append(
            "Business target definition remains unconfirmed; mart is synthetic POC only."
        )
    if phase4_bundle.target.development_status.exact_submission_timestamp_available is False:
        warnings.append("claim_date is date-level only; strict-before history remains mandatory.")
    if not phase4_bundle.target.development_status.production_approved:
        warnings.append("Real-data reapproval is required before production use.")
    return {
        "status": "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS"),
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "snapshot": {
            "rows": int(len(snapshot)),
            "unique_claims": int(snapshot["warranty_claim_key"].nunique(dropna=True)),
            "columns": int(len(snapshot.columns)),
            "positive_claims": int((target == 1).sum()),
            "negative_claims": int((target == 0).sum()),
            "direct_model_feature_count": int(
                sum(item.get("is_model_feature") is True for item in snapshot_model)
            ),
            "lineage_count": int(
                sum(
                    item.get("is_lineage") is True
                    for item in column_manifest
                    if _is_artifact(item, "claim_snapshot")
                )
            ),
        },
        "leakage": safety,
        "temporal": temporal,
        "direct_join_validation": direct_join_validation,
        "history_coverage": history_coverage,
        "bridge_row_counts": {
            name: int(len(frame)) for name, frame in frames.items() if name != "claim_snapshot"
        },
        "group_lineage": {
            "rows": int(len(group)),
            "group_types": sorted(group["group_type"].dropna().astype(str).unique().tolist())
            if not group.empty
            else [],
            "is_model_metadata": False,
        },
        "safe_scenario_fingerprint": {
            "input_fields": list(fingerprint_inputs),
            "target_included": any("target" in str(item).casefold() for item in fingerprint_inputs),
            "prohibited_fields_included": False,
        },
    }


def validate_artifact_integrity(mart_dir: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Verify manifest hashes and Parquet round-trip contents."""

    manifest_path = mart_dir / "manifest.json"
    column_manifest_path = mart_dir / "column_manifest.json"
    lineage_path = mart_dir / "field_lineage.json"
    if (
        not manifest_path.is_file()
        or not column_manifest_path.is_file()
        or not lineage_path.is_file()
    ):
        raise FeatureMartError("Mart manifest, column manifest, and field lineage are required.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        column_manifest = json.loads(column_manifest_path.read_text(encoding="utf-8"))
        json.loads(lineage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureMartError("Mart manifest files are not valid JSON.") from exc
    if not isinstance(column_manifest, list):
        raise FeatureMartError("column_manifest.json must contain a list.")
    frames: dict[str, pd.DataFrame] = {}
    for artifact, relative_path in manifest.get("artifact_paths", {}).items():
        path = mart_dir / str(relative_path)
        if not path.is_file():
            raise FeatureMartError(f"Manifest artifact is missing: {relative_path}")
        expected_file = manifest.get("artifact_file_sha256", {}).get(artifact)
        if expected_file and sha256_file(path) != expected_file:
            raise FeatureMartError(f"Artifact file checksum mismatch: {relative_path}")
        frame = pd.read_parquet(path)
        expected_content = manifest.get("artifact_content_fingerprints", {}).get(artifact)
        if expected_content and content_sha256(frame) != expected_content:
            raise FeatureMartError(f"Artifact content fingerprint mismatch: {relative_path}")
        frames[artifact] = frame
    if "claim_snapshot" not in frames:
        raise FeatureMartError("Manifest does not include claim_snapshot.")
    return {"manifest": manifest, "column_manifest": column_manifest}, frames


def validate_mart_directory(
    mart_dir: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Run offline validation against a completed local mart directory."""

    root = discover_repository_root(project_root)
    loaded, frames = validate_artifact_integrity(mart_dir)
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
    if not plan.valid:
        raise FeatureMartError("; ".join(plan.errors))
    manifest = loaded["manifest"]
    eligibility = {
        "total_claims": manifest.get("total_source_claims", 0),
        "eligible_claims": manifest.get("eligible_claims", 0),
    }
    result = validate_frames(
        frames=frames,
        eligibility=eligibility,
        contract=contract,
        phase4_bundle=phase4_bundle,
        column_manifest=loaded["column_manifest"],
        history_coverage=manifest.get("history_coverage", {}),
        direct_join_validation=manifest.get("direct_join_validation", {}),
    )
    if result["errors"]:
        result["status"] = "BLOCKED"
    return result
