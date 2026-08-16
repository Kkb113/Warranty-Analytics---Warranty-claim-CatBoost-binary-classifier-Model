# Phase 11 — Feature Selection & Ablation

Phase 11 is a controlled, TRAIN-only feature-selection experiment over the
locked Phase 10 T1/E1 and T3/E3 parents. It removes features only; CatBoost
statistical parameters, class-imbalance policy, threshold, calibration, and
ensemble policy remain frozen.

The workflow first verifies the Phase 9 and Phase 10 hardening seals and the
Phase 10 acceptance overlay, then reuses the exact three expanding inner folds.
Lineage-backed families, LossFunctionChange and mean-absolute SHAP importance,
leave-one-family-out ablations, and nested stable-ranking subsets are evaluated
using TRAIN labels only. `selection_freeze.json` is written with
`outer_validation_accessed: false` before the one selected candidate per track
is fitted on outer TRAIN and scored once on outer VALIDATION. TEST remains
sealed until Phase 15.

## Commands

```text
warranty-model phase11-contract-check
warranty-model phase11-plan-check --phase10-dir artifacts/catboost_optimization/20260811T_PHASE10
warranty-model phase11-select --phase10-dir artifacts/catboost_optimization/20260811T_PHASE10 --run-id <run-id>
warranty-model phase11-select --phase10-dir artifacts/catboost_optimization/20260811T_PHASE10 --run-id <run-id> --resume
warranty-model phase11-validate --selection-dir artifacts/feature_selection/<run-id>
```

The default local planner detects logical processors, reserves two, and uses
two bounded workers with ten CatBoost threads each on a 22-thread machine.
`--max-workers`, `--threads-per-fit`, and `--single-fit-threads` safely
downscale or override that execution plan without changing experiment
semantics. Every experiment/fold writes an atomic checkpoint under the hidden
`.phase11_<run-id>.work` directory; stale or corrupt checkpoints are rejected.

Generated artifacts and aggregate reports are intentionally ignored by Git.
Phase 11 is a synthetic proof-of-concept and does not establish production
readiness or confirm the final business target definition.

