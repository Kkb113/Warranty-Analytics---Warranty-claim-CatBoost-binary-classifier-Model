# Phase 13 — Probability calibration and controlled ensembling

Phase 13 consumes only an independently accepted Phase 12 run. It does not
retrain CatBoost, change features, retune hyperparameters, or rerun imbalance
strategies. The explicit Phase 12 path is required so the parent and effective
strategy are resolved from immutable artifacts rather than a hard-coded run ID.

## Offline gates

```text
warranty-model phase13-contract-check
warranty-model phase13-plan-check \
  --phase12-dir artifacts/imbalance_threshold/<accepted_phase12_run>
```

The plan reports the detected logical CPU count, reserved capacity, bounded
calibration workers, one thread per calibration worker, and the optional
CatBoost replay-thread override. These are execution controls only and cannot
change selection semantics. The complete calibration, tolerance, threshold,
compute, and checkpoint payload is fail-closed; `phase13-contract-check`
rejects any configuration drift.

## Run and resume

```text
warranty-model phase13-calibrate \
  --phase12-dir artifacts/imbalance_threshold/<accepted_phase12_run> \
  --run-id <phase13_run_id> \
  --max-workers 4 \
  --catboost-replay-threads 16

warranty-model phase13-calibrate \
  --phase12-dir artifacts/imbalance_threshold/<accepted_phase12_run> \
  --run-id <phase13_run_id> --resume
```

Stage A uses TRAIN OOF rows only. C1 fits source fold 1 and evaluates source
fold 2; C2 fits source folds 1–2 and evaluates source fold 3. The available
calibrators are exactly NONE, unweighted sigmoid, and guarded isotonic. The
selected calibrators are frozen before any outer validation target is loaded.
Each independent fit runs in the bounded worker pool under one native numerical
thread. Valid per-fold checkpoints include the serialized calibrator and are
reused by `--resume`; stale, corrupt, or mismatched checkpoints are refit.

The ensemble evaluates exactly the eleven convex T1 weights 0.0 through 1.0
by 0.1. Thresholds are selected on the 0.001–0.999 grid by MCC with the locked
F2/recall/precision/lower-threshold tie break. A rejected validation
calibration falls back only the competing single-track candidate to its raw
Phase 12 score policy. The frozen ensemble keeps its TRAIN-defined calibrated
components; if a component is rejected, the ensemble is rejected rather than
silently becoming a RAW/CALIBRATED hybrid. No post-validation search is
performed.

## Independent validation

```text
warranty-model phase13-validate \
  --phase13-dir artifacts/calibration_ensemble/<phase13_run_id> --json
```

Publication is atomic. The validator independently regenerates fold
assignments, calibration metrics, final calibrators, ensemble summaries,
thresholds, outer acceptance/fallback decisions, effective score spaces,
effective manifest, and the development champion. It verifies both the
canonical freeze-content hash and the freeze-file hash. TEST target rows,
predictions, and metrics remain sealed until Phase 15.

Generated artifacts live under `artifacts/calibration_ensemble/<run_id>/` and
aggregate, claim-free reports under
`reports/phase13_calibration_ensemble/<run_id>/`. They are intentionally not
committed to source control.
