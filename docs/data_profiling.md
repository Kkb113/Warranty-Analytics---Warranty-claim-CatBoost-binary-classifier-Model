# Phase 3 data profiling

## Purpose and boundary

Phase 3 profiles the 16 approved business tables in the checked-in schema
contract and audits the synthetic warranty data for quality, leakage,
duplication, temporal consistency, and suitability for Phase 4. It is
diagnostic work only: it does not clean records, impute values, engineer a
production feature mart, create a train/test split, or train a model.

The three excluded objects remain excluded by exact name:

- `dbo.ml_region_terrain_warranty_risk_dataset`
- `dbo.ml_truck_failure_risk_dataset`
- `dbo.ml_truck_region_terrain_failure_risk_dataset`

Their names may be reported by the Phase 2 catalog check; their contents are
never read.

## Configuration and commands

Install the optional analytical dependencies with:

    python -m pip install -e ".[dev,database,profiling]"

Profiling settings are typed from `configs/profiling.yaml`. The settings cover
chunk size, percentile display, category summaries, optional text/identifier/
temporal audits, and the ignored report root. They are not model-acceptance
thresholds.

All live Phase 3 commands require the same explicit database configuration as
Phase 2 and use the read-only `ApplicationIntent=ReadOnly` connection:

    warranty-model data-profile
    warranty-model synthetic-audit
    warranty-model data-quality-check
    warranty-model phase3-run --output-dir reports/data_profiling

Common options are `--output-dir`, `--no-charts`, `--format json|markdown|both`,
and `--fail-on-error`. The commands print counts and a status only; they never
print passwords, connection strings, VINs, names, notes, comments, or raw rows.

## Extraction and calculations

`profiling/extractor.py` reuses the Phase 2 `DatabaseConnection`, validates
identifiers against the schema contract, selects explicit approved columns, and
reads row-level diagnostics in bounded chunks. Exact row counts are obtained
with `COUNT_BIG(*)`. No DML, DDL, temporary objects, exports, persistent raw
cache, or excluded-table query is used.

The pipeline produces:

- table and column counts, null rates, date ranges, key uniqueness, duplicate
  rows, numeric percentiles, categorical support, text lengths, and size hints;
- actual foreign-key orphan counts for every declared relationship;
- claim target prevalence by time and joined descriptive groups;
- cost/target separability and candidate threshold intervals as empirical
  evidence only;
- descriptive point-biserial and Cramer's V associations, missingness by
  target, identifier pattern checks, group purity, duplicate fingerprints,
  text-template hashes, temporal rules, telemetry sequences, maintenance,
  service, repair, component, supplier, customer, location, and category
  sparsity diagnostics.

Post-outcome cost, amount, status, repair, finalized diagnosis, and target
relationships are explicitly reported as leakage evidence. A deterministic
cost separation is worded exactly as **“Empirical synthetic target-generation
rule suspected”**; it is not described as a confirmed business rule.

## Reports and severities

Reports are written under:

    reports/data_profiling/<run_timestamp>/

Required files are `phase_3_summary.md`, `phase_3_summary.json`,
`table_profiles.json`, `target_profile.json`, `data_quality_findings.json`,
`synthetic_data_audit.json`, and `leakage_diagnostics.json`. Generated reports
are ignored by Git. Sensitive values are replaced with aggregate counts or
short diagnostic hashes; raw text and identifiers are not reported.

Findings use:

- `ERROR` for structural or logically certain defects such as duplicate
  primary keys, foreign-key orphans, invalid required targets, and strong
  temporal contradictions;
- `WARNING` for suspected leakage, duplicate families, sparse categories,
  high missingness, sequence gaps, and provisional business-process conflicts;
- `INFO` for descriptive distributions and normal inventory facts.

`BLOCKED` means an unresolved structural error prevents the next phase. The
presence of expected post-outcome leakage in a synthetic audit alone does not
automatically block Phase 4; the legitimate prediction-time snapshot must be
defined and the fields excluded.

## Corrective hardening baseline

The corrective hardening pass superseded the earlier live diagnostic because the
original component-installation join used the claim key name on both sides and
the telemetry audit incorrectly treated `engine_hours_month` as cumulative. The
corrected engine now uses explicit left/right dimension keys, including
`causal_component_key` to `dim_component.component_key`, and validates every
dimension join as many-to-one. The component-to-supplier chain is retained in
claim diagnostic context without multiplying claim rows.

Installation group-purity diagnostics use the deterministic as-of rule
`failure_date` when available, otherwise `claim_date`, and require
`installed_date <= diagnostic_as_of_date`. The latest eligible installation is
selected. Same-date conflicting rows are marked ambiguous and excluded from
purity groups; identical grouping values may be collapsed deterministically.
The output reports matched, unmatched, ambiguous, future-excluded, and
multiple-historical-installation counts without exposing row-level identifiers.

`total_odometer_miles` remains cumulative and is checked for decreases.
`engine_hours_month` is monthly usage, so month-to-month decreases are valid;
negative engine or idle hours remain invalid, and `idle_hours_month` greater
than `engine_hours_month` is a conservative logical warning.

The four CLI commands select shared task groups: `data-profile` runs table,
column, target, category, and missingness profiling; `synthetic-audit` runs
target-generation, leakage, identifier, duplicate, text, and group-purity
audits; `data-quality-check` runs referential, temporal, telemetry,
maintenance, service/repair, and component/supplier checks; and `phase3-run`
runs all groups. CI installs the `profiling` optional dependency group while
remaining independent of SQL Server credentials.

The corrected live baseline completed with 16/16 approved tables, 392,352
rows, 8,500 claims, 0 errors, and 8 warnings. It attached component and
supplier context to all 8,500 claim rows without multiplication, matched 8,499
as-of installations, marked 11 ambiguous matches, found 230 claims with
multiple historical installations, excluded 0 future installation rows, and
reported 456 supported target-pure groups. The corrected aggregate reports are
under `reports/data_profiling/20260810T041849Z/`.

## Interpreting Phase 4 recommendations

Before feature design, the data owner and business owner still need to confirm
the target-generation rule, field availability at initial claim submission,
claim-time timestamps, missingness meaning, valid keys, date policies, and
group-aware evaluation requirements. Phase 3 records evidence; it does not
close Phase 0 questions by assumption.
