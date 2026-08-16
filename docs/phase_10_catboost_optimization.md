# Phase 10 — CatBoost optimization

Phase 10 is an acceptance-gated hyperparameter optimization pass over the
immutable Phase 9 model inputs. It is a development experiment, not production
approval. Phase 11 must not begin until the Phase 10 contract, artifact
validator, and CI workflow are green.

## Scope and lock

Only the following Phase 9 experiments are optimized:

| Track | Phase 9 experiment | Feature set |
| --- | --- | --- |
| T1 | E1 | 301-feature CORE |
| T3 | E3 | 536-feature structured-plus-lexical |

The required upstream run is `20260811T_PHASE9_FINAL` with
`HARDENED_PASS` status. Its TRAIN and VALIDATION target hashes and the E1/E3
feature-set hashes are fixed in
[`contracts/catboost_optimization.yaml`](../contracts/catboost_optimization.yaml).

Each track has exactly 50 sequential Optuna TPE trials (`n_jobs=1`) with no
pruning. The objective is mean average precision over three expanding,
chronological, same-date-grouped inner folds built from outer TRAIN only.
Every successful trial must persist exactly one row for each fold ID 1, 2,
and 3 in `trial_fold_metrics.parquet`; those rows must aggregate back to the
corresponding `trial_history.parquet` row.

The fixed CatBoost policy is CPU execution with 10 threads, deterministic
seed `20260810`, `Logloss`, Bayesian bootstrap, no class weighting, no
resampling, no early stopping, no calibration, no threshold tuning, and no
feature selection. Feature selection belongs to Phase 11.

Outer VALIDATION target access is blocked until `study_freeze.json` has been
written. Only the two frozen optimized finalists are fit on full outer TRAIN
and scored on outer VALIDATION. TEST targets, predictions, metrics, and hashes
remain sealed until Phase 15.

## Commands

Install the optimization extra before running quality checks:

```text
python -m pip install -e ".[dev,database,profiling,mart,modeling,optimization]"
```

Run the offline policy and input checks:

```text
warranty-model phase10-contract-check
warranty-model phase10-plan-check --phase9-dir artifacts/baseline_models/20260811T_PHASE9_FINAL
```

Run optimization only when a new Phase 10 run is explicitly authorized:

```text
warranty-model phase10-optimize \
  --phase9-dir artifacts/baseline_models/20260811T_PHASE9_FINAL \
  --run-id 20260811T_PHASE10
```

Validate an existing run without rerunning the 100-trial studies:

```text
warranty-model phase10-validate \
  --optimization-dir artifacts/catboost_optimization/20260811T_PHASE10
```

The standalone validator reloads Phase 9, verifies all persisted hashes and
policies, validates every trial's three fold rows and aggregate metrics, and
re-trains the selected T1 and T3 trials across the three frozen folds (six
models total). This replay is an acceptance check: it does not start Optuna or
rewrite the core optimization evidence. It refreshes `validation.json` with the
latest acceptance result and, for an existing run, writes a one-time
`phase10_acceptance_overlay.json` beside the bundle.

The overlay is deliberately separate from the optimization manifest. It records
the current manifest hash, contract checksum, validator revision, finalist model
hashes, TEST seal, and whether a pre-v2 manifest copy was preserved. If the
pre-v2 manifest is unavailable, the overlay records that fact and leaves its
original hash unset; a post-run v2 manifest is never presented as the original
legacy evidence.

## Acceptance evidence

`optimization_manifest.json` records the Phase 10 contract version/checksum,
the contract policy snapshot, the locked Phase 9 run and hashes, settings,
objective, dependency compatibility, and artifact hashes.

`validation.json` records the final hardening status plus fold-evidence and
winning-trial replay results. A `PASS WITH WARNINGS` result can retain
statistical warnings such as `INNER_CV_INSTABILITY`, but any missing or
inconsistent evidence is `BLOCKED`.

The generated reports remain local under
`reports/phase10_catboost_optimization/`; they are intentionally excluded from
Git because they contain run-specific model artifacts and metrics.
