# Phase 2 data access

## Scope

Phase 2 provides secure, read-only SQL Server connectivity and schema validation
for the 16 tables documented in
`warranty_analytics_schema_document.docx`. It validates the checked-in contract
against catalog metadata; it does not extract warranty records or build any
feature, target, or modeling dataset.

## Configuration

Settings extend the existing typed configuration layer. Live commands use:

- `WARRANTY_DB_SERVER` (required only for live commands)
- `WARRANTY_DB_PORT` (default `1433`)
- `WARRANTY_DB_DATABASE` (default `warranty_analytics`)
- `WARRANTY_DB_DRIVER` (default `ODBC Driver 18 for SQL Server`)
- `WARRANTY_DB_AUTH_MODE` (`trusted` or `sql_password`)
- `WARRANTY_DB_USERNAME` and `WARRANTY_DB_PASSWORD` for SQL authentication
- `WARRANTY_DB_ENCRYPT=true`, `WARRANTY_DB_TRUST_SERVER_CERTIFICATE=false`
- `WARRANTY_DB_APPLICATION_INTENT=ReadOnly`
- connection timeout 15 seconds and query timeout 30 seconds by default

Passwords are `SecretStr` values, never YAML settings, and never included in
diagnostics, exception text, reports, or logs. Trusted Windows authentication
does not require a username or password.

## Read-only policy

The connection layer permits only the bounded operations required for `db-check`
and `schema-validate`: `SELECT 1`, server/database identity, catalog metadata,
object names, and `sys.partitions` row estimates. There is no arbitrary SQL
execution method. No DML, DDL, stored procedure, temporary object, validation
table, export, business-table scan, or record-level sample is implemented.

Excluded ML tables may be detected by exact object name for reporting. Their
columns, indexes, keys, row counts, and definitions are never queried.

## Commands and exit policy

    warranty-model schema-contract-check
    warranty-model db-check
    warranty-model schema-validate [--strict] [--no-report]

Exit codes are 0 for a valid contract or non-blocking validation, 1 for an
invalid contract or blocking schema mismatch, 2 for configuration errors, 3
for missing drivers or connection errors, and 4 for unexpected failures.
Warnings pass by default. `--strict` promotes warnings to blocking errors.

Reports are written only when `schema-validate` is explicitly run. The default
directory is `reports/schema_validation/`, with timestamped JSON and Markdown
files. Generated contents are ignored by Git.
