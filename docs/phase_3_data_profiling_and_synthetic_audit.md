# Phase 3 — Data Profiling and Synthetic Data Audit

## Objective

Determine whether the approved warranty source is sufficiently understood for
Phase 4 target and leakage enforcement. The work is a reproducible diagnostic
pipeline, not a modeling result.

## State before Phase 3

Phase 0 documented the prediction contract and unresolved timing, target,
missingness, and group questions. Phase 1 supplied typed project scaffolding.
Phase 2 supplied the SQL Server read-only connection, the 16-table/209-column/
22-FK contract, and name-only exclusion of three `ml_*` tables. Phase 2 did
not read warranty records.

## Implementation

Created:

- `src/warranty_analytics_model/profiling/` with typed settings, explicit
  contract-scoped extraction, table/column/target profiling, relational,
  temporal, telemetry, maintenance, service/repair, component/supplier,
  association, category sparsity, duplicate, identifier, text, synthetic
  audit, findings, reporting, and runner modules;
- `configs/profiling.yaml`;
- database-independent fictional unit tests under
  `tests/unit/profiling/` and an opt-in live-test placeholder under
  `tests/integration/profiling/`;
- this record and `docs/data_profiling.md`.

Modified:

- `pyproject.toml` with the optional `profiling` dependency set
  (`numpy`, `pandas`, `scipy`, `matplotlib`) and a pandas mypy override;
- `src/warranty_analytics_model/cli.py` with `data-profile`,
  `synthetic-audit`, `data-quality-check`, and `phase3-run`.

Generated report directories remain ignored and no warranty snapshot is
committed.

## Architecture and safety

The runner reuses Phase 2 settings and `DatabaseConnection`. It validates the
contract before live extraction, issues explicit-column `SELECT` statements
only for the approved 16 tables, uses exact `COUNT_BIG(*)` row counts and
bounded row reads, and holds no persistent raw cache. Report objects contain
aggregate values, counts, percentages, distributions, and diagnostic hashes;
they do not contain raw VINs, customer names, identifiers, notes, comments, or
records.

## Findings and live status

The Phase 3 implementation and fictional offline tests completed successfully.
The local live database status must be taken from an actual operator run of
`db-check`, `schema-validate`, and `phase3-run`; this record does not infer
live row counts or target prevalence from the Phase 2 estimates. Any live
summary must document exact counts for all 16 tables and the claim target.

Offline tests specifically demonstrate threshold-separation evidence,
post-outcome leakage diagnostics, identifier/group purity, duplicate and text
fingerprints, FK orphan counts, date rules, telemetry gaps/decreases,
maintenance conflicts, service/repair arithmetic, category sparsity, and
secret-safe JSON/Markdown output. These fictional fixtures are not evidence
about the live warranty population.

## Known limitations and Phase 4 recommendations

- The schema documents dates rather than an exact claim-submission timestamp;
  same-day as-of order remains unresolved.
- Phase 3 does not infer business thresholds, eligibility rules, imputation, or
  group holdout sizes.
- Post-outcome fields are quantified as leakage evidence but remain prohibited
  from prediction-time features until availability is confirmed.
- Phase 4 should resolve Phase 0 questions, define the claim-time snapshot,
  exclude target/outcome/identifier shortcuts, and preserve group/fingerprint
  lineage for later split design.

## Definition of done

Implementation scope is complete for the reproducible offline pipeline and
quality gates. Full Phase 3 data-availability completion remains conditional on
the explicit live run documenting all 16 exact counts and audit results. No
database writes, excluded-table reads, production features, train/test split,
or predictive model training occurred.
