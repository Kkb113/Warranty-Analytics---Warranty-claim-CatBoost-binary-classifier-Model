"""Authoritative Phase 4/5 source-policy resolution for Phase 7 lineage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..feature_mart.mart_contract import iter_contract_mappings, load_mart_contract
from ..paths import discover_repository_root
from ..policy.loader import load_phase4_contracts

DISALLOWED_VALUE_POLICIES = {
    "TARGET_ONLY",
    "PROHIBITED",
    "REQUIRES_CONFIRMATION",
    "RESTRICTED_EXPERIMENTAL",
}
RAW_IDENTIFIER_TOKENS = (
    "warranty_claim_key",
    "truck_key",
    "service_event_key",
    "maintenance_event_key",
    "installation_key",
    "telemetry_month_key",
    "component_key",
    "supplier_key",
    "component_lot_no",
    "production_batch_id",
    "vin",
    "engine_serial",
    "transmission_serial",
    "technician",
    "inspector",
)


@dataclass(frozen=True, slots=True)
class ResolvedSource:
    """Policy classification for a Phase 7 lineage source alias."""

    alias: str
    policy: str
    source_table: str | None = None
    source_column: str | None = None
    known: bool = True


def is_raw_identifier(source: str) -> bool:
    """Return whether a source alias represents a raw identifier value."""

    lowered = source.casefold()
    return any(token in lowered for token in RAW_IDENTIFIER_TOKENS)


class SourcePolicyResolver:
    """Resolve Phase 7 aliases through Phase 5 mappings and Phase 4 policy."""

    def __init__(self, project_root: Path | None = None) -> None:
        root = discover_repository_root(project_root)
        mart_contract, _ = load_mart_contract(root)
        phase4 = load_phase4_contracts(root)
        policy_by_source = {
            f"{item.table}.{item.column}": item.policy
            for item in phase4.feature_policy.field_policies
        }
        self._aliases: dict[str, ResolvedSource] = {}
        for mapping in iter_contract_mappings(mart_contract):
            source = f"{mapping.source_table}.{mapping.source_column}"
            self._aliases[mapping.output_column] = ResolvedSource(
                alias=mapping.output_column,
                policy=policy_by_source.get(source, mapping.policy),
                source_table=mapping.source_table,
                source_column=mapping.source_column,
            )
        by_column: dict[str, list[ResolvedSource]] = {}
        for source, policy in policy_by_source.items():
            table, column = source.rsplit(".", 1)
            by_column.setdefault(column, []).append(
                ResolvedSource(
                    alias=column,
                    policy=policy,
                    source_table=table,
                    source_column=column,
                )
            )
        for alias, candidates in by_column.items():
            policies = {candidate.policy for candidate in candidates}
            if len(policies) == 1 and alias not in self._aliases:
                self._aliases[alias] = candidates[0]
        for alias in ("split",):
            self._aliases.setdefault(alias, ResolvedSource(alias=alias, policy="CONTROL_ONLY"))

    def resolve(self, source: str) -> ResolvedSource:
        """Resolve one output alias or raw Phase 4 field name fail-closed."""

        return self._aliases.get(
            source,
            ResolvedSource(alias=source, policy="UNKNOWN", known=False),
        )


def validate_lineage_sources(
    lineage: dict[str, dict[str, Any]], *, project_root: Path | None = None
) -> dict[str, Any]:
    """Validate value/control roles and authoritative Phase 4 source policy."""

    resolver = SourcePolicyResolver(project_root)
    errors: list[str] = []
    policy_counts: dict[str, int] = {}
    for feature_name, item in lineage.items():
        if item.get("is_model_feature") is not True:
            continue
        value_sources = item.get("value_sources", [])
        control_sources = item.get("control_sources", [])
        raw_source_columns = item.get("source_columns", [])
        if (
            not isinstance(value_sources, list)
            or not isinstance(control_sources, list)
            or not isinstance(raw_source_columns, list)
        ):
            errors.append(f"Feature {feature_name} has invalid value/control source metadata.")
            continue
        values = set(str(source) for source in value_sources)
        controls = set(str(source) for source in control_sources)
        source_columns = set(str(source) for source in raw_source_columns)
        if values & controls:
            errors.append(f"Feature {feature_name} assigns a source as both value and control.")
        if not source_columns.issubset(values | controls):
            missing = sorted(source_columns - values - controls)
            errors.append(
                f"Feature {feature_name} has unclassified source_columns: {', '.join(missing)}."
            )
        for source in sorted(values):
            resolved = resolver.resolve(source)
            policy_counts[resolved.policy] = policy_counts.get(resolved.policy, 0) + 1
            if not resolved.known:
                errors.append(f"Feature {feature_name} has unknown value source: {source}.")
            elif resolved.policy in DISALLOWED_VALUE_POLICIES:
                errors.append(
                    f"Feature {feature_name} has disallowed {resolved.policy} value source: {source}."
                )
            elif is_raw_identifier(source):
                errors.append(
                    f"Feature {feature_name} treats raw identifier as predictive value: {source}."
                )
        for source in sorted(controls):
            resolved = resolver.resolve(source)
            if not resolved.known:
                errors.append(f"Feature {feature_name} has unknown control source: {source}.")
            elif resolved.policy in {"TARGET_ONLY", "PROHIBITED"}:
                errors.append(f"Feature {feature_name} has prohibited control source: {source}.")
    return {
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "policy_counts": policy_counts,
    }
