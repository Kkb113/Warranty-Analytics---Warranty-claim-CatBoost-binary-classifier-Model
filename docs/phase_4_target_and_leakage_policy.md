# Phase 4 — Target, claim-time availability, and leakage policy

Phase 4 converts the approved Phase 0–3 evidence into versioned, machine-readable
contracts for the future claim-level feature mart. It is an enforcement phase:
it does not build a feature mart, calculate rolling or historical aggregates,
create train/validation/test assignments, train a model, or modify SQL Server.

## Machine-readable contracts

The required contracts are version-controlled under contracts/:

- high_cost_target_v1.yaml
- claim_time_feature_policy_v1.yaml
- leakage_policy_v1.yaml

Each contract records its version, creation date, source schema version and
checksum, source documents, and synthetic-development status. The offline
phase4-contract-check command validates all three contracts and fails closed
when a schema column is missing, unknown, duplicated, contradictory, or
implicitly allowed.

The current source schema contract is version 1.0.0, with 16 included tables
and 209 columns. The Phase 4 policy classifies all 209 columns exactly once.
The policy-file checksums are printed by the offline command and recorded in
Phase 4 validation reports.

## Target contract

The supervised target remains the stored binary field:

dbo.fact_warranty_claim.high_cost_claim_flag

- Positive class: 1
- Negative class: 0
- Grain: one warranty claim
- Prediction point: initial claim submission
- Prediction reference: claim_date
- Reference status: provisional date-level reference, not a precise submission timestamp

The stored target must be used as-is. total_claim_cost and all other cost or
outcome fields are prohibited from target derivation and feature use.

Phase 3 empirically found exact separation in the synthetic population:
maximum negative cost 9,998.14, minimum positive cost 10,000.91, candidate
separator 9,999.525, and zero exceptions. This is recorded as synthetic
target-generation evidence only. It is not an approved business definition,
and Phase 4 does not implement a total_claim_cost >= 10000 rule.

The target is technically valid for synthetic development, but:

- the business target definition is unconfirmed;
- the synthetic generator source was not found or approved;
- production approval is false;
- real-data reapproval is required.

## Claim eligibility

Eligibility is evaluated at the claim row level and every row receives a
category; rows are never silently dropped. The minimum rules are:

1. warranty_claim_key is non-null and unique.
2. claim_date is non-null.
3. high_cost_claim_flag is in {0, 1}.
4. truck_key resolves to dbo.dim_truck.
5. One row represents one warranty claim.

The validator reports ELIGIBLE,
INELIGIBLE_INVALID_CLAIM_KEY, INELIGIBLE_NULL_TARGET,
INELIGIBLE_INVALID_TARGET, INELIGIBLE_MISSING_CLAIM_DATE, and
INELIGIBLE_MISSING_TRUCK_LINK. Complete telemetry or maintenance history,
repair completion, part number, or post-outcome cost information are not
eligibility requirements.

## Prediction-time and same-day policy

Because the schema has dates but no precise submission/event timestamps,
historical event data uses the conservative strict_before_date_policy:

event_date < claim_date

Same-day records are excluded by default. The rule applies to maintenance,
service, component installation, prior claims, and any other date-only event.
The policy cannot be relaxed in Phase 5 without a versioned contract revision
based on an approved precise-timestamp process.

Monthly telemetry is stricter than a month-start comparison:

end_of_month(month_start_date) < claim_date

For example, a June monthly aggregate is not eligible for a June 15 claim
because it may include information from after June 15. May is eligible.

## Historical source rules

| Source | Phase 5 eligibility rule |
| --- | --- |
| dbo.fact_telemetry_monthly | Completed month: end_of_month(month_start_date) < claim_date |
| dbo.fact_maintenance_event | maintenance_date < claim_date |
| dbo.fact_service_event | service_date < claim_date and service_event_key != current_claim.service_event_key |
| dbo.fact_repair_line | Current lines prohibited; later history requires a different prior claim and prior_claim.repair_end_date < claim_date |
| Prior warranty claims | prior_claim.claim_date < claim_date; prior outcomes additionally require completion before the current claim |
| dbo.fact_component_installation | installed_date < claim_date; do not require the current claim's causal component |

Phase 4 defines these gates only. It does not calculate histories or
aggregates. Missing historical rows do not make a claim ineligible, and no
imputation is performed.

## Field policy and tiers

The feature policy uses exactly these seven values:

- TARGET_ONLY
- CONTROL_ONLY
- ALLOW_BASELINE_POC
- ALLOW_HISTORICAL_POC
- RESTRICTED_EXPERIMENTAL
- REQUIRES_CONFIRMATION
- PROHIBITED

Current contract counts are:

| Policy | Count |
| --- | ---: |
| TARGET_ONLY | 1 |
| CONTROL_ONLY | 57 |
| ALLOW_BASELINE_POC | 41 |
| ALLOW_HISTORICAL_POC | 43 |
| RESTRICTED_EXPERIMENTAL | 26 |
| REQUIRES_CONFIRMATION | 25 |
| PROHIBITED | 16 |

Tier A is generated only from explicit baseline and historical entries with
is_model_feature: true; historical entries require their source as-of rule.
Tier B contains restricted experimental entries and is isolated for later
controlled evaluation. Control, target, confirmation, and prohibited entries
are not model inputs.

The safe baseline includes stable claim/truck/model/policy descriptors and
historical telemetry, maintenance, prior-service, component-installation, and
prior-claim taxonomy only under their declared rules. It does not include
current diagnostic fields.

## Leakage enforcement

The hard blacklist covers the stored target, claim costs, adjudication amounts,
repair completion/duration, claim status, repeat/potential-recall indicators,
root cause, current diagnostic information, and current-claim repair-line data.
Leakage is classified as direct, temporal, identifier, group, or
synthetic-generator leakage.

Current-claim repair lines are prohibited. Current labor hours, parts,
repair actions, repair notes, and other repair-line values cannot enter the
baseline. Historical repair details remain restricted until prior completion
can be proven.

failure_code_key, causal_component_key, failure_date, repair_start_date,
causal_part_no, and failure_mode remain REQUIRES_CONFIRMATION.
root_cause_category remains PROHIBITED. Phase 3 diagnostic joins do not
prove that these fields were available at claim submission.

Current complaint_description and diagnostic_summary remain
REQUIRES_CONFIRMATION; no text embeddings are created in Phase 4.

Raw IDs, VINs, serial numbers, technician/inspector IDs, customer names, and
supplier/service-center identifiers are control or lineage metadata only.
They are never normal predictive inputs.

## Group and lineage policy

Phase 3 found 456 supported target-pure groups. This does not mean 456
confirmed leakage groups: with approximately 3.05% positive prevalence, many
small groups naturally contain no positive claims. The finding is treated as a
memorization and split-design risk, not automatic feature prohibition.

Production batches, component lots, supplier keys, and service-center keys are
restricted or control lineage fields. They are kept for later group-aware
evaluation, including unseen-truck, batch, supplier, lot, dealer, and duplicate
scenario holdouts.

Future datasets must carry explicit metadata such as:

- claim key and claim date;
- truck and truck-model keys;
- manufacturing plant, assembly line, production batch;
- service-center and safely derived supplier/component-lot lineage;
- duplicate/scenario fingerprint;
- target-contract and feature-policy versions.

Lineage presence does not imply model-input status. Future feature-mart rows
must carry an explicit is_model_feature flag.

## Missingness and unresolved decisions

Phase 4 does not impute. Missing telemetry, maintenance, repair part number,
or other history does not automatically exclude a claim. A null
effective_end_date remains potentially open-ended and is not replaced with
an arbitrary date. Future Phase 5 may preserve history-coverage metadata, but
must not fill missingness with future or outcome information.

The following remain open business/data-owner questions: the exact target
definition and generator logic, precise submission timestamps, claim-field
population timing, diagnostic/text timing, change history, history refresh
guarantees, lookback windows, group-holdout design, customer/location/supplier
governance, missingness representation, and real-data validation.

## CLI and readiness

Offline:

    warranty-model phase4-contract-check

Live, read-only:

    warranty-model phase4-validate
    warranty-model phase4-validate --strict

Live reports are written beneath
reports/phase4_validation/<timestamp>/. They contain policy and aggregate
validation metadata only; no VINs, raw claim text, customer records,
technician notes, passwords, connection strings, or generated data are included.

Phase 4 is READY or READY WITH WARNINGS only when all schema columns are
classified, known leakage is blocked, identifiers are excluded from Tier A,
historical rules are enforceable, and the live target is structurally valid.
Business confirmation warnings remain visible. Phase 5 must consume these
contracts and record the exact policy versions/checksums used; it must not make
new availability decisions by guessing.

## Final live validation

The final read-only run completed with READY WITH WARNINGS:

- 8,500 total claims and 8,500 eligible claims;
- 259 positive and 8,241 negative claims;
- 3.047059% positive prevalence;
- 209 of 209 schema columns classified;
- 0 blocking errors and 3 documented warnings.

The warnings are the expected synthetic-development-only status, date-level
claim reference, and unconfirmed business target definition. The aggregate
reports are under reports/phase4_validation/20260810T052019Z/.
