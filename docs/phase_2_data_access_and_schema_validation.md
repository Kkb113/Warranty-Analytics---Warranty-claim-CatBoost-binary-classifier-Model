# Phase 2 — Data access and schema validation

## Implemented outcome

Phase 2 adds a version-controlled schema contract, development-only DOCX
extraction, secure typed SQL Server settings, lazy SQLAlchemy connectivity, and
read-only catalog validation. The contract source is
`warranty_analytics_schema_document.docx` and records the approved
`warranty_analytics` scope:

- 16 included tables
- 209 included columns
- 22 documented foreign keys
- 392,352 estimated rows
- zero documented views and zero documented sequences

The excluded scope is exactly:

- `dbo.ml_region_terrain_warranty_risk_dataset`
- `dbo.ml_truck_failure_risk_dataset`
- `dbo.ml_truck_region_terrain_failure_risk_dataset`

## Validation behavior

The loader self-validates contract version, source checksum format, totals,
duplicates, ordinals, SQL type metadata, primary keys, foreign-key mappings,
included/excluded overlap, row estimates, and the approved totals. Live
validation compares tables, columns, normalized SQL types, Unicode and logical
string lengths, decimal precision/scale, nullability, identity/computed flags,
primary keys, foreign keys, indexes, properties, and partition row estimates.

Missing or incompatible schema metadata is blocking. Extra objects, collation,
default, index, foreign-key trust, widened string lengths, and row-estimate
differences are warnings by default; strict mode promotes warnings to errors.
Excluded objects are reported by name only. No contract is automatically
updated from live metadata.

## Deliberate non-scope

No warranty rows, sample claims, full counts, feature mart, missingness profile,
target construction, leakage analysis, feature engineering, model training,
predictions, API, monitoring, or AI agents were added. Phase 0 open questions
remain open until a later approved phase resolves them from evidence.

## Quality and live status

Offline commands and tests do not require SQL Server credentials. Live checks
are opt-in through `WARRANTY_RUN_DB_TESTS=true` and are not run in ordinary CI.
The live database status must be reported from an actual local run; this record
does not claim live connectivity by itself.
