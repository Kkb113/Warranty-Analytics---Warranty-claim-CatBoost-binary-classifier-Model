# Phase 12 — Imbalance & Threshold Optimization

Phase 12 evaluates only the locked Phase 11 effective T1/T3 parents. It tests
the eight versioned weighting strategies (`S0_NONE` through
`S7_AUTO_BALANCED`) on frozen chronological TRAIN inner folds and selects a
raw-score technical threshold from out-of-fold predictions using MCC and
deterministic tie breaks.

All TRAIN fits, OOF predictions, strategy selection, and threshold evidence are
completed before `phase12_freeze.json` is written. Only then may outer
VALIDATION labels be read. TEST remains sealed until Phase 15. Feature changes,
hyperparameter search, resampling, calibration, and ensembling are prohibited.

Use the offline gates before training:

```text
warranty-model phase12-contract-check
warranty-model phase12-plan-check --phase11-dir artifacts/feature_selection/<accepted-run>
```

The default bounded plan is four workers × five CatBoost threads (20 active
logical threads with two reserved); a finalist fit may use up to 16 threads.
Execution-only overrides are available through `--max-workers`,
`--threads-per-fit`, and `--single-fit-threads`.

```text
warranty-model phase12-optimize --phase11-dir artifacts/feature_selection/<accepted-run>
warranty-model phase12-optimize --phase11-dir artifacts/feature_selection/<accepted-run> --resume
warranty-model phase12-validate --phase12-dir artifacts/imbalance_threshold/<run-id>
```

Completed bundles are published atomically. The persisted threshold is a
`RAW_UNCALIBRATED_PROBABILITY` technical development threshold; business
approval and calibration are deferred to Phase 13.
