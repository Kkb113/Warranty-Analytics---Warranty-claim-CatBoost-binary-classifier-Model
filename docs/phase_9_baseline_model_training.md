# Phase 9 — Baseline Model Training

Phase 9 is the first model-training phase. It consumes only the locked Phase 5
mart, corrected Phase 6 split, hardened Phase 7 structured features, and Phase
8 historical-text companion artifact. CatBoost models fit on TRAIN only;
selection metrics use VALIDATION only. TEST labels remain sealed until Phase 15,
and Phase 9 creates no TEST predictions or metrics.

## Fixed experiments

- E0: constant TRAIN prevalence, excluded from champion selection.
- E1: 301 Phase 7 CORE features.
- E2: all 507 Phase 7 features.
- E3: E2 plus 29 Phase 8 numeric/boolean lexical features.
- E4: E3 plus four raw historical documents through CatBoost native text.

All CatBoost experiments use Logloss, 500 iterations, learning rate 0.05,
depth 6, seed 20260810, CPU, one thread, Bayesian bootstrap, and no early
stopping, class weighting, resampling, threshold optimization, calibration,
feature selection, or ensembling. Numeric missing values are preserved;
categorical and text missing values use fixed model-only sentinels.

## Validation and selection

Average precision is primary. The run also records trapezoidal PR AUC, ROC AUC,
log loss, Brier score, and descriptive confusion metrics at the fixed 0.5
threshold. Ties use ROC AUC, then lower log loss, then the simpler experiment.
Saved models are reloaded and their VALIDATION probabilities and metrics are
recomputed at a tolerance of `1e-12` before atomic publication.

The completed local run `20260810T_PHASE9` selected E3 on VALIDATION average
precision. This remains a synthetic POC result, is not production approval,
and must not be interpreted as final TEST performance.

## Commands

    warranty-model phase9-contract-check
    warranty-model phase9-plan-check --mart-dir <phase5> --split-dir <phase6> --structured-dir <phase7> --text-dir <phase8>
    warranty-model phase9-train --mart-dir <phase5> --split-dir <phase6> --structured-dir <phase7> --text-dir <phase8>
    warranty-model phase9-validate --model-dir artifacts/baseline_models/<run_id>

Generated model artifacts and aggregate-only reports remain ignored under
`artifacts/baseline_models/` and `reports/phase9_baseline_models/`.
