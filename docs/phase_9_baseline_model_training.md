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

## Corrective hardening

The corrective hardening pass preserves the Phase 9 experiment semantics and
adds fail-closed artifact acceptance controls:

- target labels are validated as exact binary values before integer casting;
- standalone validation freshly reloads development targets and reconciles all
  persisted TRAIN/VALIDATION target hashes while rejecting recursive TEST hashes;
- E0–E4 statuses, model files, per-experiment VALIDATION membership, duplicate
  rows, E4-unavailable behavior, champion selection, AP lift, and feature-set
  lineage hashes are independently recomputed;
- persisted CatBoost policy is checked for the locked parameters, disabled
  weighting, no evaluation-set/early-stopping decisions, and reload probability
  equality at `atol=1e-12`, `rtol=0`;
- runtime/library provenance is recorded without usernames, home paths,
  secrets, or environment dumps; and
- the TEST seal requires zero labels, predictions, and metrics, with no TEST
  artifact files and Phase 15 as the first allowed TEST-target phase.

The original `20260810T_PHASE9` run is immutable and validates as
`LEGACY_VALID`. A corrective run should use a new immutable ID, for example
`20260811T_PHASE9_HARDENED`; its report includes the before/after comparison.
If the locked inputs and runtime are reproduced, the expected champion remains
E3. Phase 10 may begin only after the hardened run and comparison pass.

## Commands

    warranty-model phase9-contract-check
    warranty-model phase9-plan-check --mart-dir <phase5> --split-dir <phase6> --structured-dir <phase7> --text-dir <phase8>
    warranty-model phase9-train --mart-dir <phase5> --split-dir <phase6> --structured-dir <phase7> --text-dir <phase8>
    warranty-model phase9-validate --model-dir artifacts/baseline_models/<run_id>

Generated model artifacts and aggregate-only reports remain ignored under
`artifacts/baseline_models/` and `reports/phase9_baseline_models/`.
