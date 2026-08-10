# Phase 7 — Structured feature engineering

Phase 7 creates one deterministic, target-independent structured feature row
per eligible warranty claim. It consumes the completed Phase 5 claim mart and
the locked Phase 6 chronological assignments entirely offline.

The versioned policy is `contracts/structured_feature_contract_v1.yaml` and
technical settings are in `configs/structured_features.yaml`. The builder
preserves raw date fields only as controls, carries safe non-date direct values
as CORE candidates, and leaves categorical values as strings for later CatBoost
handling. It creates lifecycle, usage, warranty, telemetry, maintenance,
service, component, prior-claim, and history-coverage features.

No target, target statistic, current repair/diagnostic outcome, text, restricted
identifier, evaluation cohort flag, global fitted transformation, imputation,
scaling, encoding, feature selection, or model metric is produced in this
phase. `repair_history_index` remains control-only and
`prior_failure__failure_description` is deferred to Phase 8.

## Offline commands

```text
warranty-model phase7-contract-check
warranty-model phase7-plan-check --mart-dir <phase5-run> --split-dir <phase6-run>
warranty-model phase7-build --mart-dir <phase5-run> --split-dir <phase6-run>
warranty-model phase7-validate --feature-dir artifacts/structured_features/<run_id>
```

The build verifies Phase 5 and Phase 6 validation, all frozen membership and
TEST-lock hashes, source compatibility, as-of windows, target absence,
deterministic ordering, feature lineage, numeric safety, and atomic publication.
Generated Parquet, manifests, and reports are ignored by the public repository.

## Hardened feature semantics

The model matrix contains deterministic direct, lifecycle, usage, warranty,
telemetry, maintenance, service, component, prior-claim, and history-coverage
families. Historical windows are `3m`, `6m`, `12m`, `24m`, and `all`, with
CORE/EXTENDED metadata preserved for every model candidate. Safe categorical
values remain strings; Phase 7 performs no encoding, imputation, scaling,
feature selection, or model training.

Telemetry history uses strict completed-month semantics. The telemetry month
containing the claim is excluded from both the observed-month numerator and the
expected-month denominator. For a June claim, May is the latest eligible
completed month regardless of whether the claim occurs on June 1, June 15, or
June 30. The denominator also respects the requested lookback and the vehicle
in-service month; invalid or empty ranges remain NULL, and coverage ratios are
not clipped.

Feature lineage distinguishes `value_sources` from `control_sources`. A value
source contributes a measurement, category, or numeric date-derived quantity.
Keys and dates used only for joins, filtering, counting, ordering, tie-breaking,
or identity are controls. The authoritative Phase 4 policy and Phase 5 mart
contract reject target-only, prohibited, confirmation-required, restricted, and
raw-identifier values. `repair_history_index` produces no model features and
`prior_failure__failure_description` remains deferred to Phase 8.

Target independence is enforced by building without the target column; changing
or removing target values cannot change the structured matrix. The exact
corrected Phase 6 assignments and TEST lock are revalidated before every build.
The current validated run is `artifacts/structured_features/20260810T_PHASE7_HARDENED/`.
It consumes `artifacts/feature_mart/20260810T102230Z/` and
`artifacts/splits/20260810T_PHASE6_CORRECTED/`; its expected warnings are constant
TRAIN features, synthetic-POC status, date-level prediction reference, and
real-data reapproval requirements.

Phase 7 is hardened and was the locked structured input for Phase 8. Phase 8
owns historical text feature development; Phase 9 owns baseline model training.
