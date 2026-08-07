"""Secure read-only data access and schema validation for Phase 2."""

from .config import load_database_settings
from .connection import (
    DatabaseConnection,
    available_odbc_drivers,
    build_connection_url,
    check_database_connection,
    safe_connection_display,
    validate_driver,
)
from .metadata import collect_schema_metadata
from .models import ConnectivityResult, LiveSchemaMetadata, SchemaContract, ValidationResult
from .reporting import write_validation_reports
from .schema_contract import contract_checksum, load_schema_contract, validate_contract
from .schema_diff import diff_schema
from .schema_validator import validate_schema

__all__ = [
    "ConnectivityResult",
    "DatabaseConnection",
    "LiveSchemaMetadata",
    "SchemaContract",
    "ValidationResult",
    "available_odbc_drivers",
    "build_connection_url",
    "check_database_connection",
    "collect_schema_metadata",
    "contract_checksum",
    "diff_schema",
    "load_schema_contract",
    "load_database_settings",
    "safe_connection_display",
    "validate_contract",
    "validate_driver",
    "validate_schema",
    "write_validation_reports",
]
