"""Contract-driven, explicit-column source extraction for Phase 5."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..database.models import SchemaContract
from ..policy.models import Phase4ContractBundle
from .mart_contract import iter_contract_mappings
from .models import FeatureMartError, MartContract

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class ExtractionColumn:
    """One explicitly requested source column and its bounded purpose."""

    table: str
    column: str
    purpose: str
    persist_as_feature: bool
    control_gate: bool


@dataclass(frozen=True, slots=True)
class ExtractionPlan:
    """Deterministic source table/column plan derived from the contracts."""

    columns_by_table: dict[str, tuple[ExtractionColumn, ...]]
    excluded_tables: tuple[str, ...]

    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(self.columns_by_table)


def quote_identifier(identifier: str) -> str:
    """Quote a contract identifier after strict validation."""

    if not _IDENTIFIER_RE.fullmatch(identifier):
        raise FeatureMartError(f"Unsafe SQL identifier: {identifier!r}")
    return f"[{identifier}]"


def explicit_select_sql(table_name: str, columns: list[str] | tuple[str, ...]) -> str:
    """Build a read-only explicit-column SELECT for a contract table."""

    if "." not in table_name:
        raise FeatureMartError(f"Expected schema-qualified table name: {table_name}")
    schema, table = table_name.split(".", 1)
    if not columns:
        raise FeatureMartError(f"No columns requested for extraction: {table_name}")
    selected = ", ".join(quote_identifier(column) for column in columns)
    return f"SELECT {selected} FROM {quote_identifier(schema)}.{quote_identifier(table)}"


def explicit_count_sql(table_name: str) -> str:
    """Build a read-only exact count query for one approved table."""

    if "." not in table_name:
        raise FeatureMartError(f"Expected schema-qualified table name: {table_name}")
    schema, table = table_name.split(".", 1)
    return (
        f"SELECT COUNT_BIG(*) AS [exact_row_count] FROM "
        f"{quote_identifier(schema)}.{quote_identifier(table)}"
    )


def _mapping_purpose(mapping: Any) -> tuple[str, bool]:
    if mapping.is_target:
        return "TARGET_LABEL", False
    if mapping.is_model_feature:
        if mapping.transform_type == "history_bridge":
            return "HISTORICAL_FEATURE", False
        return "DIRECT_FEATURE", False
    if mapping.is_control or mapping.is_lineage:
        return "CONTROL_GATE", True
    return "LINEAGE", False


def build_extraction_plan(
    schema_contract: SchemaContract,
    phase4_bundle: Phase4ContractBundle,
    mart_contract: MartContract,
) -> ExtractionPlan:
    """Create the explicit source plan from versioned contract mappings."""

    schema_tables = schema_contract.table_map
    excluded = set(schema_contract.excluded_tables)
    if set(phase4_bundle.feature_policy.excluded_tables) != excluded:
        raise FeatureMartError("Phase 5 extraction scope disagrees with the Phase 4 exclusions.")
    requested: dict[tuple[str, str], ExtractionColumn] = {}

    def add_requested(
        table_name: str,
        column_name: str,
        *,
        purpose: str,
        persist_as_feature: bool,
        control_gate: bool,
    ) -> None:
        if table_name == "derived":
            return
        if table_name in excluded:
            raise FeatureMartError(f"Excluded ML table requested by mart: {table_name}")
        table = schema_tables.get(table_name)
        if table is None:
            raise FeatureMartError(f"Mart mapping references unknown source table: {table_name}")
        if column_name not in table.column_map:
            raise FeatureMartError(
                f"Mart mapping references unknown source column: {table_name}.{column_name}"
            )
        key = (table_name, column_name)
        existing = requested.get(key)
        requested[key] = ExtractionColumn(
            table=table_name,
            column=column_name,
            purpose=existing.purpose if existing else purpose,
            persist_as_feature=bool(existing and existing.persist_as_feature) or persist_as_feature,
            control_gate=bool(existing and existing.control_gate) or control_gate,
        )

    for mapping in iter_contract_mappings(mart_contract):
        purpose, gate = _mapping_purpose(mapping)
        add_requested(
            mapping.source_table,
            mapping.source_column,
            purpose=purpose,
            persist_as_feature=mapping.is_model_feature,
            control_gate=gate,
        )
        for join_field in mapping.join_path:
            if join_field.count(".") < 2:
                continue
            table_name, column_name = join_field.rsplit(".", 1)
            if table_name in schema_tables:
                add_requested(
                    table_name,
                    column_name,
                    purpose="CONTROL_GATE",
                    persist_as_feature=False,
                    control_gate=True,
                )

    def add_join_paths(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                add_join_paths(item)
        elif isinstance(value, list):
            for item in value:
                add_join_paths(item)
        elif isinstance(value, str) and value.count(".") >= 2:
            table_name, column_name = value.rsplit(".", 1)
            if table_name in schema_tables:
                add_requested(
                    table_name,
                    column_name,
                    purpose="CONTROL_GATE",
                    persist_as_feature=False,
                    control_gate=True,
                )

    add_join_paths(mart_contract.source_join_paths)

    if not requested:
        raise FeatureMartError("The Phase 5 extraction plan is empty.")
    by_table: dict[str, list[ExtractionColumn]] = {}
    for item in requested.values():
        by_table.setdefault(item.table, []).append(item)
    ordered: dict[str, tuple[ExtractionColumn, ...]] = {}
    for table_name, items in by_table.items():
        table = schema_tables[table_name]
        order = {column.name: column.ordinal for column in table.columns}
        ordered[table_name] = tuple(sorted(items, key=lambda item: order[item.column]))
    return ExtractionPlan(columns_by_table=ordered, excluded_tables=tuple(sorted(excluded)))


def plan_columns(plan: ExtractionPlan, table_name: str) -> tuple[str, ...]:
    """Return ordered source column names for one planned table."""

    try:
        return tuple(item.column for item in plan.columns_by_table[table_name])
    except KeyError as exc:
        raise FeatureMartError(
            f"Table is not in the Phase 5 extraction plan: {table_name}"
        ) from exc
