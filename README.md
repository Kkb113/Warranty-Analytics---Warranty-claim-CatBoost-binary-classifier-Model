# Warranty Analytics Model

This repository is the reproducible foundation for a truck-warranty model that
will eventually predict `dbo.fact_warranty_claim.high_cost_claim_flag` at the
initial warranty-claim submission. The Phase 0 contract remains authoritative;
unresolved business and availability questions are not inferred by code.

## Current status

Current phase: **Phase 5 — Claim-Level Feature Mart Construction**.

Phase 0 model-contract, Phase 1 scaffolding, Phase 2 read-only data access /
schema validation, and the Phase 3 live profiling run are complete. The live
run finished with **READY WITH WARNINGS**: all 16 approved tables and 392,352
rows were profiled, with 0 errors and 8 warnings. This corrected run supersedes
the earlier 9-warning Phase 3 diagnostic. The timestamped aggregate reports are
under `reports/data_profiling/20260810T041849Z/`.

The corrected baseline includes explicit component/supplier context joins,
as-of-safe component-installation matching, and monthly telemetry semantics
that do not treat `engine_hours_month` as cumulative. It remains diagnostic;
Phase 4 target approval is still provisional and its warnings remain active.

Phase 4 contract validation is complete offline and is the required gate before
the live read-only target/policy audit. The Phase 5 feature mart is built and
validated with **PASS WITH WARNINGS**: 8,500 source claims, 8,500 eligible
claims, 8,500 unique snapshot rows, 259 positives, 8,241 negatives, 41/41
direct fields, and 43/43 historical fields. The run is under
`artifacts/feature_mart/20260810T102230Z/`; aggregate reports are under
`reports/phase5_feature_mart/20260810T102230Z/`.

Feature mart built. Train/validation/test splits have not yet been created.
No predictive model has been trained.
The final live Phase 4 audit is READY WITH WARNINGS: 8,500 of 8,500 claims are
eligible, positive prevalence is 3.047059%, policy coverage is 209/209, and
there are 0 blocking errors. The aggregate report is under
reports/phase4_validation/20260810T101935Z/.

## Local setup

    python -m pip install -e ".[dev,database,profiling,mart]"

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
    warranty-model data-profile
    warranty-model synthetic-audit
    warranty-model data-quality-check
    warranty-model phase3-run --no-charts
    warranty-model phase4-contract-check
    warranty-model phase4-validate
    warranty-model phase5-plan-check
    warranty-model phase5-build
    warranty-model phase5-validate --mart-dir artifacts/feature_mart/<run_id>

Phase 3 commands share the read-only extractor but select distinct task groups:
`data-profile` runs profiling and target/category/missingness diagnostics,
`synthetic-audit` runs synthetic/leakage/group diagnostics,
`data-quality-check` runs relational/temporal/operational checks, and
`phase3-run` runs the complete workflow. CI installs the same profiling group
with `.[dev,database,profiling,mart]` and remains SQL Server-independent.

`schema-contract-check` is offline and validates the checked-in YAML. `db-check`
and `schema-validate` require live settings such as `WARRANTY_DB_SERVER` and an
installed Microsoft ODBC Driver 18. `schema-validate` writes timestamped JSON
and Markdown reports under `reports/schema_validation/`; use `--no-report` to
disable output or `--strict` to treat warnings as blocking. `phase3-run` writes
timestamped aggregate JSON/Markdown reports under `reports/data_profiling/`.

The Phase 4 contract check is offline and validates the target, feature-policy,
and leakage contracts across all 209 schema columns. The Phase 4 validation
command is live and read-only: it runs schema validation, target eligibility,
target prevalence, target-generation regression evidence, source-policy checks,
and secret-safe reports under reports/phase4_validation/. Use --no-report,
--format, --output-dir, or --strict as needed.

The recorded live run used a process-only
`WARRANTY_DB_TRUST_SERVER_CERTIFICATE=true` override because the local SQL
Server certificate was not trusted; `.env` was not changed. Prefer a trusted
server certificate for persistent use. If the local override is approved for
development, set it only in the local environment and never commit `.env`.

Phase 5 uses the Phase 4 contracts to build a local Parquet mart bundle. The
offline `phase5-plan-check` validates all 41 direct and 43 historical mappings.
`phase5-build` performs the required live read-only gates, writes an atomic run
under `artifacts/feature_mart/<run_id>/`, and generates aggregate-only reports
under `reports/phase5_feature_mart/<run_id>/`. `phase5-validate` rechecks an
existing bundle without database access. Generated artifacts and reports are
ignored and must not be committed.

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

Phase 2 reads SQL Server catalog views and partition row estimates. Phase 3
reads approved business-table columns in read-only, explicit-column chunks and
uses exact counts for profiling. It does not run `SELECT *`, DML/DDL, stored
procedures, exports, target construction, production feature engineering,
training, prediction, or monitoring. The three excluded ML tables are detected
by exact object name only; their contents are never read.

## Documentation

- [Model contract](docs/model_contract.md)
- [Phase 0 open questions](docs/phase_0_open_questions.md)
- [Architecture](docs/architecture.md)
- [Development setup](docs/development_setup.md)
- [Data access](docs/data_access.md)
- [Phase 2 implementation record](docs/phase_2_data_access_and_schema_validation.md)
- [Data profiling methodology](docs/data_profiling.md)
- [Phase 3 implementation record](docs/phase_3_data_profiling_and_synthetic_audit.md)
- [Phase 4 target and leakage policy](docs/phase_4_target_and_leakage_policy.md)
- [Phase 5 claim feature mart](docs/phase_5_claim_feature_mart.md)
- [Schema contract notes](contracts/README.md)
- [Phase 1 scaffolding record](docs/phase_1_scaffolding.md)
- [Contributing guide](CONTRIBUTING.md)
