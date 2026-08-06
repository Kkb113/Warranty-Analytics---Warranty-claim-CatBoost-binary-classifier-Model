"""Typed contract, catalog metadata, diff, and report result models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Severity = Literal["ERROR", "WARNING", "INFO"]
ValidationStatus = Literal["passed", "failed"]


class SQLTypeSpec(BaseModel):
    """Normalized SQL Server type metadata in logical length units."""

    model_config = ConfigDict(extra="forbid")

    base_type: str
    max_length: int | None = Field(default=None, ge=0)
    precision: int | None = Field(default=None, ge=1)
    scale: int | None = Field(default=None, ge=0)
    unicode: bool = False
    length_unit: Literal["characters", "bytes"] | None = None
    is_max: bool = False

    @field_validator("base_type")
    @classmethod
    def normalize_base_type(cls, value: str) -> str:
        return value.strip().casefold()


class ColumnSpec(BaseModel):
    """Contract or live catalog representation of one column."""

    model_config = ConfigDict(extra="forbid")

    ordinal: int = Field(ge=1)
    name: str
    sql_type: SQLTypeSpec
    nullable: bool
    identity: bool = False
    computed: bool = False
    default: str | None = None
    collation: str | None = None


class PrimaryKeySpec(BaseModel):
    """Ordered primary-key metadata."""

    model_config = ConfigDict(extra="forbid")

    name: str
    columns: list[str] = Field(min_length=1)
    system_named: bool | None = None


class ForeignKeySpec(BaseModel):
    """Ordered foreign-key mapping metadata."""

    model_config = ConfigDict(extra="forbid")

    name: str
    referenced_table: str
    parent_columns: list[str] = Field(min_length=1)
    referenced_columns: list[str] = Field(min_length=1)
    on_delete: str
    on_update: str
    trusted: bool | None = None


class IndexSpec(BaseModel):
    """Index metadata used for non-blocking drift reporting."""

    model_config = ConfigDict(extra="forbid")

    name: str
    index_type: str
    unique: bool
    key_columns: list[str] = Field(min_length=1)
    included_columns: list[str] = Field(default_factory=list)
    disabled: bool | None = None
    filter_definition: str | None = None


class TableProperties(BaseModel):
    """Documented table storage properties."""

    model_config = ConfigDict(extra="forbid")

    storage_classification: str | None = None
    temporal: bool | None = None
    memory_optimized: bool | None = None
    file_table: bool | None = None


class TableSpec(BaseModel):
    """One included table and its schema metadata."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    schema_name: str = Field(alias="schema")
    table: str
    estimated_rows: int | None = Field(default=None, ge=0)
    column_count: int = Field(ge=0)
    primary_key: PrimaryKeySpec | None = None
    foreign_keys: list[ForeignKeySpec] = Field(default_factory=list)
    indexes: list[IndexSpec] = Field(default_factory=list)
    properties: TableProperties = Field(default_factory=TableProperties)
    columns: list[ColumnSpec] = Field(default_factory=list)

    @property
    def column_map(self) -> dict[str, ColumnSpec]:
        return {column.name: column for column in self.columns}

    @property
    def schema(self) -> str:  # type: ignore[override]
        """Return the schema using the contract's public vocabulary."""

        return self.schema_name


class ContractDatabase(BaseModel):
    """Database identity recorded by the source document."""

    model_config = ConfigDict(extra="forbid")

    name: str
    platform: Literal["sql_server"]


class ContractSource(BaseModel):
    """Provenance for the checked-in contract."""

    model_config = ConfigDict(extra="forbid")

    document: str
    sha256: str
    extraction_date: str
    views_documented: int = Field(ge=0)
    sequences_documented: int = Field(ge=0)


class ContractSummary(BaseModel):
    """Reconciled document totals."""

    model_config = ConfigDict(extra="forbid")

    included_tables: int = Field(ge=0)
    included_columns: int = Field(ge=0)
    documented_foreign_keys: int = Field(ge=0)
    estimated_rows: int = Field(ge=0)


class SchemaContract(BaseModel):
    """Fully parsed version-controlled schema contract."""

    model_config = ConfigDict(extra="forbid")

    contract_version: str
    database: ContractDatabase
    source: ContractSource
    summary: ContractSummary
    excluded_tables: list[str]
    tables: list[TableSpec]

    @property
    def table_map(self) -> dict[str, TableSpec]:
        return {table.name: table for table in self.tables}

    @property
    def included_column_count(self) -> int:
        return sum(len(table.columns) for table in self.tables)

    @property
    def foreign_key_count(self) -> int:
        return sum(len(table.foreign_keys) for table in self.tables)


class LiveSchemaMetadata(BaseModel):
    """Read-only SQL Server catalog snapshot for included objects."""

    model_config = ConfigDict(extra="forbid")

    database_name: str
    server_name: str | None = None
    sql_version: str | None = None
    tables: list[TableSpec] = Field(default_factory=list)
    all_table_names: list[str] = Field(default_factory=list)
    schemas: list[str] = Field(default_factory=list)
    views: list[str] = Field(default_factory=list)
    sequences: list[str] = Field(default_factory=list)
    excluded_objects: list[str] = Field(default_factory=list)
    catalog_readable: bool = True

    @property
    def table_map(self) -> dict[str, TableSpec]:
        return {table.name: table for table in self.tables}

    @property
    def included_column_count(self) -> int:
        return sum(len(table.columns) for table in self.tables)

    @property
    def foreign_key_count(self) -> int:
        return sum(len(table.foreign_keys) for table in self.tables)


class SchemaIssue(BaseModel):
    """One actionable contract-versus-live metadata finding."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    severity: Severity
    code: str
    object_type: str
    schema_name: str | None = Field(default=None, alias="schema")
    table: str | None = None
    column: str | None = None
    constraint: str | None = None
    expected: Any = None
    actual: Any = None
    message: str
    remediation: str

    @property
    def schema(self) -> str | None:  # type: ignore[override]
        """Return the optional schema using the report vocabulary."""

        return self.schema_name


class SchemaDiffResult(BaseModel):
    """Structured result of a contract-versus-catalog comparison."""

    model_config = ConfigDict(extra="forbid")

    issues: list[SchemaIssue] = Field(default_factory=list)
    row_count_differences: int = Field(default=0, ge=0)


class ValidationResult(BaseModel):
    """Stable, secret-free schema validation result used by reports and CLI."""

    model_config = ConfigDict(extra="forbid")

    validation_id: str
    contract_version: str
    contract_checksum: str
    execution_timestamp: datetime
    environment: str
    server: str | None = None
    database: str
    sql_version: str | None = None
    status: ValidationStatus
    included_table_count: int = Field(ge=0)
    included_column_count: int = Field(ge=0)
    included_foreign_key_count: int = Field(ge=0)
    actual_table_count: int = Field(ge=0)
    actual_column_count: int = Field(ge=0)
    actual_foreign_key_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    info_count: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    issues: list[SchemaIssue] = Field(default_factory=list)
    excluded_objects: list[str] = Field(default_factory=list)
    row_count_differences: int = Field(default=0, ge=0)


class ConnectivityResult(BaseModel):
    """Safe result for the lightweight ``db-check`` command."""

    model_config = ConfigDict(extra="forbid")

    checked_at: datetime
    server: str | None = None
    port: int
    expected_database: str
    actual_database: str | None = None
    sql_version: str | None = None
    catalog_readable: bool
    duration_seconds: float = Field(ge=0)
