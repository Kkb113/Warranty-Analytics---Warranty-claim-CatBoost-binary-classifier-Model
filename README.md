# Warranty Analytics Model

This repository is the reproducible foundation for a truck-warranty model that
will eventually predict `dbo.fact_warranty_claim.high_cost_claim_flag` at the
initial warranty-claim submission. The Phase 0 contract remains authoritative;
unresolved business and availability questions are not inferred by code.

## Current status

Current phase: **Phase 10 — CatBoost Optimization**.

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

Feature mart built. The Phase 6 split contract and deterministic chronological
split implementation are now in place. The final split is built only from the
verified Phase 5 mart; no SQL Server query is used to create assignments. No
predictive model has been trained.
The final live Phase 4 audit is READY WITH WARNINGS: 8,500 of 8,500 claims are
eligible, positive prevalence is 3.047059%, policy coverage is 209/209, and
there are 0 blocking errors. The aggregate report is under
reports/phase4_validation/20260810T101935Z/.

## Local setup

    python -m pip install -e ".[dev,database,profiling,mart,modeling,optimization]"

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
    warranty-model phase6-contract-check
    warranty-model phase6-plan-check --mart-dir artifacts/feature_mart/<run_id>
    warranty-model phase6-build --mart-dir artifacts/feature_mart/<run_id>
    warranty-model phase6-validate --split-dir artifacts/splits/<run_id>
    warranty-model phase7-contract-check
    warranty-model phase7-plan-check --mart-dir artifacts/feature_mart/<run_id> --split-dir artifacts/splits/<run_id>
    warranty-model phase7-build --mart-dir artifacts/feature_mart/<run_id> --split-dir artifacts/splits/<run_id>
    warranty-model phase7-validate --feature-dir artifacts/structured_features/<run_id>
    warranty-model phase8-contract-check
    warranty-model phase8-plan-check --mart-dir artifacts/feature_mart/<run_id> --split-dir artifacts/splits/<run_id> --structured-dir artifacts/structured_features/<run_id>
    warranty-model phase8-build --mart-dir artifacts/feature_mart/<run_id> --split-dir artifacts/splits/<run_id> --structured-dir artifacts/structured_features/<run_id>
    warranty-model phase8-validate --text-dir artifacts/text_features/<run_id>
    warranty-model phase9-contract-check
    warranty-model phase9-plan-check --mart-dir artifacts/feature_mart/<run_id> --split-dir artifacts/splits/<run_id> --structured-dir artifacts/structured_features/<run_id> --text-dir artifacts/text_features/<run_id>
    warranty-model phase9-train --mart-dir artifacts/feature_mart/<run_id> --split-dir artifacts/splits/<run_id> --structured-dir artifacts/structured_features/<run_id> --text-dir artifacts/text_features/<run_id>
    warranty-model phase9-validate --model-dir artifacts/baseline_models/<run_id>
    warranty-model phase10-contract-check
    warranty-model phase10-plan-check --phase9-dir artifacts/baseline_models/20260811T_PHASE9_FINAL
    warranty-model phase10-optimize --phase9-dir artifacts/baseline_models/20260811T_PHASE9_FINAL --run-id 20260811T_PHASE10
    warranty-model phase10-validate --optimization-dir artifacts/catboost_optimization/20260811T_PHASE10

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

Phase 6 consumes an already validated Phase 5 mart and freezes one
date-preserving chronological TRAIN/VALIDATION/TEST partition. The boundary
algorithm uses only claim dates, date-level row counts, and the configured
70/15/15 fractions; target values are used only for post-assignment
diagnostics. Phase 6 also writes hash-only test-lock metadata, group-exposure
diagnostics, and claim-level evaluation cohorts. Generated split artifacts and
reports remain ignored. The test target is reserved for first final evaluation
in Phase 15.

Phase 7 structured feature engineering is complete and hardened. The validated
offline run provides 507 leakage-safe structured candidates, with no predictive
model trained. Phase 8 historical text feature development is complete and
hardened with 33 target-independent text candidates across 8,500 claims. It
uses only prior failure descriptions from the locked Phase 5 history and keeps
the exact Phase 6 membership and TEST lock. The Phase 8 run is under
`artifacts/text_features/20260810T_PHASE8/`; aggregate reports are under
`reports/phase8_text_features/20260810T_PHASE8/`.

Phase 9 baseline training and its corrective hardening are complete with
**PASS WITH WARNINGS**. E0–E4 are evaluated on the frozen VALIDATION split
only, E3 remains the development champion, and saved-model reload validation
passes at the locked probability tolerance. The immutable original run is
retained as `LEGACY_VALID`; the hardened reproduction records exact target,
feature, prediction, runtime, inventory, and policy provenance. No TEST target,
prediction, or metric is accessed or generated. Phase 10 may start only after
the hardened run/comparison passes; this remains a synthetic POC and is not
production approval.

Phase 10 optimizes only immutable Phase 9 E1 CORE (T1) and E3
structured-plus-lexical (T3) tracks with three TRAIN-only chronological inner
folds and 50 sequential Optuna trials per track. Outer VALIDATION opens only
after `study_freeze.json`; TEST targets, predictions, metrics, and hashes remain
sealed until Phase 15. Phase 11 owns feature selection. See
`docs/phase_10_catboost_optimization.md` for the contract and run sequence.

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
    warranty-model phase4-contract-check
    warranty-model phase5-plan-check
    warranty-model phase6-contract-check
    warranty-model phase7-contract-check
    warranty-model phase7-plan-check --mart-dir artifacts/feature_mart/<run_id> --split-dir artifacts/splits/<run_id>
    warranty-model phase8-contract-check
    warranty-model phase8-plan-check --mart-dir artifacts/feature_mart/<run_id> --split-dir artifacts/splits/<run_id> --structured-dir artifacts/structured_features/<run_id>
    warranty-model phase9-contract-check

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
- [Phase 6 train/validation/test split design](docs/phase_6_train_validation_test_split.md)
- [Phase 7 structured feature engineering](docs/phase_7_structured_feature_engineering.md)
- [Phase 8 text feature development](docs/phase_8_text_feature_development.md)
- [Phase 9 baseline model training](docs/phase_9_baseline_model_training.md)
- [Phase 10 CatBoost optimization](docs/phase_10_catboost_optimization.md)
- [Schema contract notes](contracts/README.md)
- [Phase 1 scaffolding record](docs/phase_1_scaffolding.md)
- [Contributing guide](CONTRIBUTING.md)
