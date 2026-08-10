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
The live run completed on 2026-08-10 UTC with `db-check` and
`schema-validate` passing, followed by `phase3-run --no-charts --format both
--fail-on-error`. The run status is **READY WITH WARNINGS**: 0 errors, 9
warnings, and 0 informational findings. The report is aggregate-only and is
stored in the ignored directory
`reports/data_profiling/20260810T032748Z/`.

The live run profiled these exact approved-table row counts:

| Table | Rows |
|---|---:|
| `dbo.dim_component` | 350 |
| `dbo.dim_customer` | 500 |
| `dbo.dim_date` | 1,827 |
| `dbo.dim_failure_code` | 120 |
| `dbo.dim_location` | 75 |
| `dbo.dim_service_center` | 100 |
| `dbo.dim_supplier` | 100 |
| `dbo.dim_truck` | 5,000 |
| `dbo.dim_truck_model` | 20 |
| `dbo.dim_warranty_policy` | 10 |
| `dbo.fact_component_installation` | 90,000 |
| `dbo.fact_maintenance_event` | 42,000 |
| `dbo.fact_repair_line` | 29,750 |
| `dbo.fact_service_event` | 34,000 |
| `dbo.fact_telemetry_monthly` | 180,000 |
| `dbo.fact_warranty_claim` | 8,500 |
| **Total** | **392,352** |

The target audit found 8,500 usable claims, 259 positive/high-cost claims
(3.047059%), 8,241 negative claims (96.952941%), and no null or invalid target
values. The target is imbalanced. `total_claim_cost` separated the target with
an empirical candidate threshold of 9,999.525, maximum negative 9,998.14,
minimum positive 10,000.91, zero exceptions, and no distribution overlap;
this is evidence of a likely synthetic generation rule, not an approved
business definition.

The nine live warnings were: high missingness in
`dim_warranty_policy.effective_end_date` (60.0%) and
`fact_repair_line.part_no` (71.428571%); target imbalance; 44,193 telemetry
missing-month gaps; 87,526 telemetry engine-hours decreases; synthetic
identifier leakage; 293 supported target-pure groups; duplicate scenario
families in repair-line and repair-pattern fingerprints; and one sparse
category field. No temporal rule violations, maintenance logical conflicts,
duplicate event records, or foreign-key orphans were observed. Repair-line
part numbers were missing in 21,250 rows; service/repair and component/supplier
audits otherwise completed with aggregate diagnostics.

Fourteen suspected post-outcome leakage fields were quantified:
`total_claim_cost`, `labor_cost`, `parts_cost`, `other_cost`,
`diagnostic_cost`, `towing_cost`, `approved_amount`, `rejected_amount`,
`customer_paid_amount`, `days_to_repair`, `root_cause_category`,
`claim_status`, `repeat_claim_flag`, and `potential_recall_flag`.

The database connection required a process-only
`WARRANTY_DB_TRUST_SERVER_CERTIFICATE=true` override because the local SQL
Server certificate was not trusted. The repository `.env` was not changed and
no credentials were included in the reports. A trusted server certificate is
preferred for persistent use.

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

Implementation scope and the live Phase 3 data-availability audit are complete
with documented warnings. No database writes, excluded-table reads, production
features, train/test split, or predictive model training occurred.
