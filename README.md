# Warranty Analytics Model

This repository is the reproducible foundation for a truck-warranty model that
will eventually predict `dbo.fact_warranty_claim.high_cost_claim_flag` at the
initial warranty-claim submission. The Phase 0 contract remains authoritative;
unresolved business and availability questions are not inferred by code.

## Current status

Phase 1 scaffolding and Phase 2 data access/schema validation are implemented.
The project can validate the version-controlled SQL Server schema contract and,
when explicitly configured, compare it with live SQL Server catalog metadata.
No warranty records are extracted, no feature mart or target is constructed,
and no model, prediction API, or monitoring capability exists.

## Local setup

    python -m pip install -e ".[dev,database]"

The optional `.env` file is local-only. Copy `.env.example` to `.env` and set
live database values only through a secure local environment. Never commit
credentials or database exports.

## Commands

    warranty-model doctor
    warranty-model show-config
    warranty-model version
    warranty-model schema-contract-check
    warranty-model db-check
    warranty-model schema-validate

`schema-contract-check` is offline and validates the checked-in YAML. `db-check`
and `schema-validate` require live settings such as `WARRANTY_DB_SERVER` and an
installed Microsoft ODBC Driver 18. `schema-validate` writes timestamped JSON
and Markdown reports under `reports/schema_validation/`; use `--no-report` to
disable output or `--strict` to treat warnings as blocking.

## Repository structure

- `contracts/`: version-controlled schema contract and provenance notes.
- `configs/`: non-secret layered settings.
- `docs/`: model contract, setup, architecture, and phase records.
- `src/warranty_analytics_model/`: installable package, including read-only
  database access and catalog validation.
- `sql/source_validation/`: SQL validation resource documentation; packaged
  reviewed catalog queries live with the installable database module.
- `tests/`: unit, integration, and optional live database checks.
- `data/`, `artifacts/`, `reports/`, `logs/`: controlled or generated areas.

## Quality checks

    python -m ruff check .
    python -m ruff format --check .
    python -m mypy src
    python -m pytest
    warranty-model schema-contract-check

CI runs these checks without credentials or SQL Server access. Live tests are
opt-in only when `WARRANTY_RUN_DB_TESTS=true` and valid local settings exist.

## Boundaries and safety

Phase 2 reads SQL Server catalog views and partition row estimates only. It does
not run `SELECT *`, full business-table scans, full `COUNT(*)` scans, DML/DDL,
stored procedures, exports, profiling, target construction, feature engineering,
training, prediction, or monitoring. The three approved ML dataset tables are
excluded and may be detected by exact object name only; their columns are never
inspected or used.

## Documentation

- [Model contract](docs/model_contract.md)
- [Phase 0 open questions](docs/phase_0_open_questions.md)
- [Architecture](docs/architecture.md)
- [Development setup](docs/development_setup.md)
- [Data access](docs/data_access.md)
- [Phase 2 implementation record](docs/phase_2_data_access_and_schema_validation.md)
- [Schema contract notes](contracts/README.md)
- [Phase 1 scaffolding record](docs/phase_1_scaffolding.md)
- [Contributing guide](CONTRIBUTING.md)
