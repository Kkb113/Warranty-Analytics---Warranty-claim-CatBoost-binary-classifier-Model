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
