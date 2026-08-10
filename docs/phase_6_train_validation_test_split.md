# Phase 6 — Train / Validation / Test Split Design

## Objective

Phase 6 freezes the evaluation design before structured feature engineering,
feature selection, model fitting, hyperparameter tuning, imbalance handling,
threshold tuning, calibration, or ensembling. It answers which eligible claims
belong to TRAIN, VALIDATION, and TEST, records exact date boundaries and
membership hashes, and seals TEST for first final target-based evaluation in
Phase 15.

Phase 6 does not engineer features, transform labels, resample rows, train a
model, or calculate model-performance metrics.

## Authoritative input and preconditions

The input is an already completed Phase 5 mart directory such as:

    artifacts/feature_mart/<run_id>/

Phase 6 reads `claim_snapshot.parquet`,
`lineage/claim_group_membership.parquet`, `manifest.json`, and the Phase 5
manifests. It does not reconstruct the mart or query SQL Server.

Before assigning a split, the implementation verifies:

- Phase 5 status is `PASS` or `PASS WITH WARNINGS` and its offline validation has no errors.
- Required snapshot, group-membership, and manifest files exist.
- Snapshot grain is one row per eligible claim, claim keys are unique, claim dates are valid, and the stored target is binary `{0, 1}`.
- Phase 5 Parquet file/content fingerprints still match its manifest.
- The Phase 5 mart contract and all Phase 4 schema, target, feature-policy, and leakage checksums match the current versioned contracts.

Any failed integrity or compatibility check blocks Phase 6.

## Split contract and configuration

`contracts/claim_split_v1.yaml` is the versioned machine-readable contract.
`configs/splits.yaml` contains technical evaluation-design settings. The
current contract carries forward the Phase 5 synthetic-POC development status:
production approval is false, real-data reapproval is required, the business
target definition is unconfirmed, and precise submission timestamps are not
available.

The requested fractions are:

| Split | Requested fraction |
| --- | ---: |
| TRAIN | 70% |
| VALIDATION | 15% |
| TEST | 15% |

Fractions are row-count targets. Date grouping may move actual percentages;
deviations above three percentage points are warnings, and any split below ten
percent is blocking.

## Chronological boundary algorithm

The primary partition is chronological. Claims are grouped by the unique
date-level prediction reference `claim__claim_date`. Dates are sorted
ascending, date-level claim counts are calculated, and cumulative claim counts
are compared with:

    total_claims * 0.70
    total_claims * (0.70 + 0.15)

The closest unique date is selected for each boundary. If two dates are equally
close, the earlier date wins. The target column is not passed into, read by, or
used by boundary selection.

Every claim on one date is assigned to the same split:

    TRAIN       claim_date <= train_end_date
    VALIDATION  train_end_date < claim_date <= validation_end_date
    TEST        claim_date > validation_end_date

Therefore no same-date claims are arbitrarily ordered across partitions. The
validator requires strict date ordering, exact claim coverage, no overlap, and
no duplicate assignment.

## Target diagnostics and sufficiency gates

After assignments exist, reports record row counts, actual percentages, date
ranges, positive/negative counts, positive prevalence, and overall prevalence.
The split is never changed in response to target diagnostics.

Blocking conditions are:

- a split contains only one target class;
- validation has fewer than 10 positives;
- test has fewer than 10 positives;
- any split has fewer than 10% of claims;
- assignment, date, checksum, or test-lock integrity fails.

Warnings are emitted for validation/test positive counts from 10 through 19,
fewer than 100 training positives, and date-grouping fraction drift.
No stratification, resampling, SMOTE, target weighting, or label mutation is
performed.

## Group exposure and evaluation cohorts

The Phase 5 `safe_scenario_fingerprint` is consumed as-is; it is not
recalculated or changed. Phase 6 also consumes available truck, model, plant,
assembly-line, production-batch, service-center, historical supplier,
historical component-lot, and historical component-batch memberships.

`group_exposure.parquet` is normalized to one claim × one hashed group
membership. It records first-seen split, seen-in-TRAIN, seen-in-VALIDATION,
seen-in-development, and corresponding unseen flags. Raw group values are not
copied into this artifact and all rows are marked `is_model_feature: false`.

The primary split is not group-forced. A known truck, batch, service center,
supplier, lot, or repeated scenario remains in the chronological partition;
overlap is reported as a generalization dimension rather than automatically
treated as leakage.

`evaluation_cohorts.parquet` contains one metadata row per claim. For
TRAIN is the reference population and is not considered unseen: its available
direct groups are reference-known, and its historical-group unseen counts are
zero. For validation, unseen means unseen in TRAIN. For test, unseen means
unseen in TRAIN + VALIDATION (development). Flags include fingerprint, truck,
production-batch, and service-center cohorts plus any/all unseen historical
supplier, component-lot, and component-batch exposure. These flags are not
model features and do not create additional splits.

Aggregate group-overlap claim counts are scoped independently by `group_type`.
A claim can therefore be seen for one dimension and unseen for another. For
one-to-many historical groups, a claim with a mixture of reference-known and
unseen values may count in both the seen and unseen claim totals; the cohort
artifact separately records `any_unseen` and `all_unseen`. The corrected Phase 6
run supersedes the earlier aggregate overlap report while preserving the exact
chronological assignments and TEST lock.

Fingerprint overlap is a warning. A fingerprint-clean validation claim has a
fingerprint never seen in TRAIN. A fingerprint-clean test claim has a
fingerprint never seen in TRAIN or VALIDATION. Overlapping claims remain in the
primary test set so the full chronological evaluation is preserved.

## Artifacts, hashes, and test lock

Each build writes an immutable generated directory:

    artifacts/splits/<run_id>/
        split_assignments.parquet
        group_exposure.parquet
        evaluation_cohorts.parquet
        split_manifest.json
        split_validation.json
        test_lock.json

`split_assignments.parquet` contains only `warranty_claim_key`, `claim_date`,
and `split`; it does not copy model features or the target. Parquet file hashes
and deterministic content hashes are recorded for every artifact. Membership
hashes use canonical `claim_date` ascending / `warranty_claim_key` ascending
ordering.

`test_lock.json` contains no raw claim keys. It records the split-contract
checksum, exact Phase 5 input fingerprint, test dates/count, ordered and
unordered test-key hashes, test-assignment content hash, `locked: true`, and
`allowed_first_target_evaluation_phase: 15`. Re-running the same Phase 5 mart
with the same Phase 6 contract must reproduce all membership and content
fingerprints. A changed Phase 5 mart creates a new split run; completed runs
are never silently replaced.

## Test access policy

- TRAIN may be used for Phase 7 target-independent feature engineering and Phase 9+ model fitting.
- VALIDATION may be used for Phase 7 target-independent feature engineering and Phase 9–14 model selection/tuning/development evaluation.
- TEST may receive only target-independent feature construction in Phase 7. Test target-based performance must not guide Phases 9–14. Phase 15 is the first final target-based evaluation.

Integrity checks may still verify test target counts and membership hashes.

## Commands and reports

Run the full required sequence against the verified Phase 5 mart:

    warranty-model phase5-validate --mart-dir artifacts/feature_mart/<run_id>
    warranty-model phase6-plan-check --mart-dir artifacts/feature_mart/<run_id>
    warranty-model phase6-build --mart-dir artifacts/feature_mart/<run_id>
    warranty-model phase6-validate --split-dir artifacts/splits/<phase6_run_id>

The offline contract gate is:

    warranty-model phase6-contract-check

Aggregate reports are written under
`reports/phase6_splits/<run_id>/`:

- `phase_6_summary.md` and `phase_6_summary.json`
- `split_distribution.json`
- `group_overlap.json`
- `evaluation_cohorts.json`
- `split_validation.json`

Reports contain aggregate diagnostics only. Generated artifacts and reports
are Git-ignored and must not be committed to the public repository.

## Readiness

Phase 6 is safe to hand off to Phase 7 only when Phase 5 validation, contract
compatibility, assignment coverage, date ordering, target reconciliation,
group/cohort consistency, artifact checksums, and the test lock all pass. A
`PASS WITH WARNINGS` result is expected for the current synthetic POC because
the target/business definition and production approval remain provisional.
