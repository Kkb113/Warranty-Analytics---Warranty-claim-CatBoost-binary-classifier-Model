# Phase 5 — Claim-Level Feature Mart Construction

Phase 5 turns the validated Phase 4 target, availability, and leakage contracts
into a local, read-only claim-level mart bundle. It preserves the eligible claim
snapshot and the complete safe history needed by later feature engineering. It
does not create splits, train a model, calculate model metrics, or write to SQL
Server.

## Objective and boundary

The prediction grain is one eligible `dbo.fact_warranty_claim` row. The target
is copied from the stored `high_cost_claim_flag`; it is never recreated from
cost and is never used to choose history. The provisional prediction reference
is `claim_date`. Because the source has date-level, not submission-timestamp,
precision, all date-only history uses strict-before semantics.

Phase 5 is intentionally a mart bundle rather than a flattened feature table:

```text
eligible claim
  ├── claim_snapshot.parquet                 one row per eligible claim
  ├── history/telemetry_history.parquet      zero-to-many completed months
  ├── history/maintenance_history.parquet    zero-to-many prior events
  ├── history/service_history.parquet         zero-to-many prior events
  ├── history/component_installation_history.parquet
  ├── history/prior_claim_history.parquet    zero-to-many earlier claims
  ├── history/repair_history_index.parquet   prior-repair control/index only
  └── lineage/claim_group_membership.parquet group and scenario metadata
```

No history is aggregated, imputed, encoded, scaled, embedded, or target
conditioned in this phase. Missing history remains missing and never removes an
otherwise eligible claim.

## Authoritative inputs and checksums

The checked-in mart contract is
`contracts/claim_feature_mart_v1.yaml`. `phase5-plan-check` recomputes and
validates the following exact inputs before live extraction:

| Contract | Version | SHA-256 |
| --- | --- | --- |
| Schema contract | 1.0.0 | `00f86ec9db8ea84d88ea53d3a95d5df4e9a455d9555013d089026a399660cb06` |
| `high_cost_target_v1.yaml` | 1.0.0 | `17f39d6abf71861bd9c9d7c514c1847cef08049a312988b1c4dec3b4b204c25b` |
| `claim_time_feature_policy_v1.yaml` | 1.0.0 | `1b42aef48ddaedddaae9288cec2494fcce528788c6038f14668e767403e7c3c9` |
| `leakage_policy_v1.yaml` | 1.0.0 | `fda75d50c0e6fd06c59e04a0d7cfc3dad5080b1257870021c76aa6040140ec3e` |
| `claim_feature_mart_v1.yaml` | 1.0.0 | `6e2650270c788dadb8ab46ddaf9e38f2940474d85cf959a96f168ab80c45855e` |

The Phase 5 manifest records all five values, the package version, build time,
source database name, source row counts, and the Git commit SHA. Credentials,
connection strings, raw rows, and generated artifacts are never written to the
repository.

## Direct claim snapshot

The snapshot applies the exact Phase 4 eligibility mask: non-null unique claim
key, non-null date, binary stored target, and a resolvable truck link. It then
performs explicit, validated many-to-one joins:

```text
claim → truck → truck_model
      → effective warranty_policy
claim_date → dim_date
claim.service_center_key → service_center → location
```

The service-location route is deliberately restricted to the current claim’s
`service_center_key`. Customer-derived location paths are not used. Policy
versions are valid when `effective_start_date <= claim_date` and the end date is
null or `effective_end_date >= claim_date`; a null end date is treated as
open-ended. Missing policy values are reported as an applicability diagnostic,
not imputed and not used to delete an eligible claim.

The 41 direct Tier A fields are all materialized and namespaced in the snapshot:

| Source | Fields |
| --- | --- |
| Claim (6) | `claim_date`, `odometer_miles_at_failure`, `engine_hours_at_failure`, `months_in_service`, `warranty_coverage_status`, `claim_type` |
| Truck (8) | `manufacturing_plant`, `assembly_line`, `build_date`, `delivery_date`, `in_service_date`, `axle_configuration`, `fuel_type`, `emission_standard` |
| Truck model (8) | `brand`, `model_name`, `model_year`, `segment`, `application_type`, `cab_type`, `engine_platform`, `gvwr_class` |
| Warranty policy (5) | `coverage_months`, `coverage_miles`, `coverage_engine_hours`, `deductible_amount`, `coverage_type` |
| Claim calendar (10) | `day_number`, `day_name`, `week_number`, `month_number`, `month_name`, `quarter_number`, `year_number`, `fiscal_month`, `fiscal_quarter`, `fiscal_year` |
| Service location (4) | `country`, `region`, `climate_zone`, `terrain_type` |

The stored target is `target__high_cost_claim_flag`. Claim and lineage keys are
control metadata, not model features.

## Historical bridges

All bridges preserve every eligible pre-claim record for which the Phase 4
as-of rule is satisfied. Each bridge validates the pair
`current_warranty_claim_key + source_record_key` and allows zero-to-many rows.

| Artifact | As-of and current-record rule | Approved source fields |
| --- | --- | --- |
| Telemetry | `end_of_month(month_start_date) < claim_date`; claim month and future months excluded | 16 monthly telemetry fields |
| Maintenance | `maintenance_date < claim_date` | 7 maintenance fields |
| Service | `service_date < claim_date` and source event key differs from current claim event key | 3 service fields |
| Component installation | `installed_date < claim_date`; truck-level history; never selected by `causal_component_key` | 6 component-dimension fields plus 4 installation fields |
| Prior claim | `prior_claim.claim_date < claim_date`; current claim cannot join to itself | 7 prior failure-taxonomy fields |
| Repair index | Prior claim `repair_end_date < claim_date`; current-claim lines excluded | No repair values; only prior eligibility/index controls |

The 43 historical Tier A fields are mapped as follows:

- `dim_component`: `component_system`, `component_category`,
  `standard_life_miles`, `standard_life_months`, `is_safety_critical`,
  `unit_cost` (6).
- `dim_failure_code`: `failure_code`, `failure_description`,
  `failure_system`, `failure_category`, `severity_level`,
  `safety_related_flag`, `recall_related_flag` (7).
- `fact_component_installation`: `quality_check_status`, `rework_flag`,
  `torque_value`, `inspection_score` (4).
- `fact_maintenance_event`: `odometer_miles`, `engine_hours`,
  `maintenance_type`, `scheduled_flag`, `completed_on_time_flag`,
  `overdue_days`, `maintenance_cost` (7).
- `fact_service_event`: `odometer_miles`, `engine_hours`, `service_type` (3).
- `fact_telemetry_monthly`: `mileage_month`, `total_odometer_miles`,
  `engine_hours_month`, `idle_hours_month`, `avg_engine_temp`,
  `max_engine_temp`, `avg_oil_pressure`, `low_oil_pressure_events`,
  `brake_air_pressure_alerts`, `battery_voltage_alerts`, `fault_code_count`,
  `harsh_braking_events`, `avg_payload_weight`, `fuel_efficiency_mpg`,
  `route_severity_score`, `maintenance_compliance_score` (16).

Current repair-line values remain excluded by the Phase 4 wildcard
`dbo.fact_repair_line.*`. `repair_history_index.parquet` exists only to show
which earlier repair lines satisfy the completion gate. `repair_end_date` is a
control gate and is not a model feature. `failure_code_key` and
`causal_component_key` remain confirmation-only in current-claim context.

## Lineage and scenario fingerprint

`claim_group_membership.parquet` is non-model metadata. It records deterministic
group membership for truck, truck model, manufacturing plant, assembly line,
production batch, service center, historical supplier, historical component lot,
historical component batch, and the safe scenario fingerprint. Group rows carry
`is_model_feature: false`.

The Phase 5 `leakage_safe_scenario_fingerprint` is a SHA-256 fingerprint of
approved snapshot values plus split-control lineage keys only. It excludes the
target, costs, repair outcomes, current diagnostics, and other post-outcome
fields. It is a future split-control diagnostic, not a model input.

## Extraction and atomicity

The extraction planner derives explicit source columns from the schema contract,
Phase 4 policy, and mart contract. It rejects unknown columns, excluded ML
tables, and broad `SELECT *` plans. SQL Server access uses the existing Phase 2
read-only connection with `ApplicationIntent=ReadOnly`; no DML, DDL, stored
procedures, or SQL feature tables are created. `total_claim_cost` is not read by
Phase 5 construction.

The runner extracts once in chunks, constructs the snapshot and bridges with
vectorized DataFrame joins, writes to a temporary run directory, validates every
artifact, and renames it to the final run directory only after validation passes.
An existing completed run is not overwritten unless `--overwrite` is supplied.

## Artifact layout and manifests

Each run is written beneath:

```text
artifacts/feature_mart/<run_id>/
  claim_snapshot.parquet
  history/{telemetry,maintenance,service,component_installation,prior_claim}_history.parquet
  history/repair_history_index.parquet
  lineage/claim_group_membership.parquet
  manifest.json
  column_manifest.json
  field_lineage.json
  validation.json
```

`manifest.json` contains aggregate row counts, positive/negative counts, bridge
counts, source counts, all contract checksums, artifact paths, file SHA-256
hashes, canonical content SHA-256 fingerprints, deferred fields, and validation
status. `column_manifest.json` and `field_lineage.json` provide one auditable
entry per materialized column, including source, policy, join path, as-of rule,
transform type, and model/target/lineage/control flags. Parquet is written
without a DataFrame index, with deterministic columns and row order, preserving
SQL null semantics.

Reports are aggregate-only and are written beneath
`reports/phase5_feature_mart/<run_id>/`:

```text
phase_5_summary.md
phase_5_summary.json
mart_validation.json
history_coverage.json
direct_feature_coverage.json
historical_field_coverage.json
```

## Validation and acceptance

`phase5-plan-check` is database-independent and fails closed on contract drift,
missing mappings, unsafe policies, customer-location paths, wildcard leakage,
identifier model inputs, excluded tables, or unsafe history rules. The current
offline plan validates 41/41 direct and 43/43 historical mappings.

`phase5-validate --mart-dir ...` verifies artifact file/content checksums,
manifest coverage, snapshot grain and target integrity, direct join
cardinality, bridge pair uniqueness, strict temporal rules, current service and
repair contamination, component-history independence from current causal
component, group metadata safety, and scenario-fingerprint safety. Blocking
conditions produce `BLOCKED`; structurally valid synthetic-PoC output is
`PASS WITH WARNINGS` while Phase 4 business/timestamp/production warnings remain
visible.

The required operator sequence is:

```text
python -m warranty_analytics_model db-check
python -m warranty_analytics_model schema-validate
python -m warranty_analytics_model phase4-validate
python -m warranty_analytics_model phase5-build
python -m warranty_analytics_model phase5-validate --mart-dir artifacts/feature_mart/<run_id>
```

## Source counts and live-run record

The checked-in contracts preserve the previous synthetic baseline of
approximately 8,500 claims, 259 positives, and 8,241 negatives. Live values
must be recorded from the current read-only source and must not be forced to
match that baseline. If counts drift while Phase 4 remains valid, the drift is
reported; if Phase 4 or schema validation fails, the mart is blocked.

The verified live run for this checkout is:

| Metric | Result |
| --- | ---: |
| Status | PASS WITH WARNINGS |
| Source claims / eligible claims / snapshot rows | 8,500 / 8,500 / 8,500 |
| Unique snapshot claims | 8,500 |
| Positive / negative target counts | 259 / 8,241 |
| Direct / historical Tier A coverage | 41/41 / 43/43 |
| Lineage columns / group-membership rows | 11 / 520,525 |
| Direct join multiplication | 0 |
| Telemetry / maintenance / service rows | 164,551 / 39,323 / 32,778 |
| Component / prior-claim / repair-index rows | 153,675 / 9,887 / 34,022 |
| Same-day / future / claim-month telemetry violations | 0 / 0 / 0 |
| Current service / current repair contamination | 0 / 0 |
| Prohibited / confirmation / restricted / identifier model fields | 0 / 0 / 0 / 0 |
| Source drift from 8,500 / 259 / 8,241 baseline | 0 / 0 / 0 |

The run directory is `artifacts/feature_mart/20260810T102230Z/` and the
aggregate-only report directory is
`reports/phase5_feature_mart/20260810T102230Z/`. The final
`phase5-validate --mart-dir ...` command passed. This document intentionally
does not copy raw rows or credentials.

## Limitations and Phase 6 readiness

The following warnings remain authoritative and are intentionally carried
forward:

- development data is synthetic only;
- the business target definition and generator source are unconfirmed;
- `claim_date` is date-level and no precise submission timestamp is available;
- customer fields, current failure/causal-component fields, and text fields are
  unresolved;
- supplier and service-center descriptive identifiers remain control/lineage;
- real-data reapproval is required before production use;
- no predictive model performance is claimed.

When the live gates and artifact validation pass, this bundle is ready for Phase
6 split design and Phase 7 structured feature engineering. Phase 6 must consume
the recorded group and scenario lineage, and Phase 7 must define any missingness
representation or historical aggregates under a new reviewed contract. No
Phase 6 split or model artifact is created by this Phase 5 implementation.
