# Architecture

## Purpose and current boundary

The repository is a configuration-driven Python package for future truck-warranty
claim modeling. Phase 1 implements package, configuration, path, logging,
reproducibility, CLI, and quality infrastructure. Phase 2 adds read-only SQL
Server catalog access and schema validation. Phase 3 adds contract-scoped,
read-only data profiling and synthetic-data auditing. Phase 4 versions target,
availability, and leakage policy. Phase 5 builds a local, contract-driven,
read-only claim snapshot and as-of history bundle. No phase trains a model or
serves inference.

## Package structure

- common/: shared utilities that are not specific to a future modeling stage.
- database/: Phase 2 typed SQL Server connection, catalog metadata, contract,
  diff, and reporting modules.
- policy/: Phase 4 versioned target, claim-time feature-availability, leakage,
  eligibility, allowlist, lineage, and report enforcement.
- profiling/: Phase 3 explicit-column extraction, table/column/target profiles,
  data-quality checks, temporal/telemetry audits, synthetic-data audits,
  as-of installation matching, findings, and report generation.
- ingestion/: later boundary for approved production source extraction; the
  Phase 3 profiling extractor is diagnostic and contract-scoped, not an
  ingestion or feature pipeline.
- validation/: Phase 2 schema comparison and later logical data-quality boundary.
- feature_mart/: Phase 5 claim snapshot, as-of history bridges, manifests, and validation.
- splits/: Phase 6 chronological assignments, test-locking, group exposure,
  evaluation cohorts, manifests, and offline validation.
- structured_features/: Phase 7 deterministic structured feature construction,
  source-policy lineage, manifests, quality diagnostics, and offline validation.
- text_features/: Phase 8 historical text documents and deterministic lexical
  aggregates, source-policy lineage, manifests, quality diagnostics, and
  offline validation.
- baseline_model/: Phase 9 fixed experiment definitions, adapters, CatBoost
  training, validation-only metrics, manifests, reporting, and model reload
  validation.
- catboost_optimization/: Phase 10 immutable E1/E3 track optimization,
  TRAIN-only chronological inner folds, sequential Optuna studies, finalist
  fitting, manifests, reports, and standalone validation.
- feature_selection/: Phase 11 family ablation, fold-stable importance,
  deterministic candidate subsets, checkpointed TRAIN-only selection, and
  post-freeze outer-VALIDATION replacement evaluation.
- models/: Phase 5 and later boundary for training and model artifacts.
- evaluation/: Phase 5 and later boundary for metrics and calibration.
- inference/: Phase 5 and later boundary for inference; no inference exists in
  Phase 1.
- config.py: typed layered settings and safe redaction.
- paths.py: repository-relative path resolution.
- logging_config.py: standard-library console and optional file logging.
- reproducibility.py: deterministic Python random seeding.
- cli.py: infrastructure and schema commands plus explicit live Phase 3
  profiling/audit, Phase 4 policy validation, Phase 5 mart commands, and
  offline Phase 6 split, Phase 7 structured-feature, and Phase 8 text-feature
  commands.

## Corrected Phase 3 diagnostic flow

Claim diagnostic context remains at one row per warranty claim. Dimension joins
use explicit left/right keys and many-to-one validation. In particular:

    claim.causal_component_key
        -> dim_component.component_key
        -> component attributes and supplier_key
        -> dim_supplier.supplier_key
        -> supplier attributes

Component-installation purity diagnostics use a separate one-to-zero/one
as-of match on `truck_key` plus component key. `failure_date` is preferred and
`claim_date` is the fallback; only `installed_date` values on or before that
diagnostic date are eligible, and the latest eligible installation is selected.
Same-date conflicting rows are counted as ambiguous and excluded from purity
groups unless their audited grouping values are identical. The matching result
is aggregate-reported and does not expose claim identifiers.

Telemetry treats `total_odometer_miles` as cumulative, while
`engine_hours_month` and `idle_hours_month` are monthly quantities. Monthly
engine-hour decreases are not sequence defects. Negative measurements remain
invalid, and idle hours greater than engine hours are reported as a conservative
logical diagnostic.

The Phase 3 CLI shares one extraction and profiling engine with selectable task
groups. `data-profile` runs profiling, target distribution, category, and
missingness work; `synthetic-audit` runs synthetic and leakage diagnostics;
`data-quality-check` runs relational, temporal, telemetry, maintenance,
service/repair, and component/supplier checks; `phase3-run` runs all groups.

The package uses the src layout so imports resolve from the installed package
rather than from an accidental repository-root module.

## Configuration flow

Settings use this precedence:

    Typed defaults < configs/base.yaml < configs/<environment>.yaml
    < optional local .env < operating-system environment variables

WARRANTY_MODEL_ENV selects development or test before the environment-specific
YAML file is loaded. The local .env file is optional. Operating-system
environment variables win over values from .env. Database credentials are
accepted only from local environment variables and live connectivity is
explicitly read only.

## Logging flow

Modules obtain named standard-library loggers through logging_config.get_logger.
The CLI and future entry points call configure_logging explicitly. Console
logging is enabled by default, handlers are marked and reused on repeated setup,
and file logging is opt-in. Generic infrastructure logs must not contain secrets
or record-level values such as VINs, customer names, notes, or complaint text.

## Path handling

paths.py discovers the repository from the current working directory or an
explicit root and resolves configured relative directories with pathlib. Importing
the package does not create directories. Output directories are created only by
an explicit ensure_output_directories call or explicit file-logging setup.

## Testing structure

- tests/unit/: isolated checks for configuration, paths, logging,
  reproducibility, and CLI behavior.
- tests/integration/: checks that multiple Phase 1 components work together.
- tests/conftest.py: shared test isolation and environment cleanup.

Tests use temporary directories and fictional DataFrames for profiling behavior.
Normal CI tests do not connect to a database, require network access, or use
warranty records. Optional live tests remain gated by
`WARRANTY_RUN_DB_TESTS=true`.
Optional live tests are gated by `WARRANTY_RUN_DB_TESTS=true` and valid local
settings.

## Phase boundaries

- Phase 1: project scaffolding and infrastructure quality gates.
- Phase 2: SQL Server connectivity, schema validation, and source-table checks.
- Phase 3: data profiling, synthetic-data audit, and logical data-quality work.
- Phase 4: stored-target validation, claim eligibility, prediction-time
  availability policy, historical as-of rules, leakage enforcement, and
  versioned feature/lineage allowlists.
- Phase 5: claim-level snapshot and safe history-mart construction from the
  Phase 4 contracts; no flattening, imputation, splits, or model training.
- Phase 6: deterministic chronological split design, frozen test membership,
  group-exposure diagnostics, and evaluation cohorts. No feature engineering,
  target transformation, resampling, model training, or model metrics.
- Phase 7: deterministic structured feature engineering from the locked Phase 5
  mart and Phase 6 split; no model training or metrics.
- Phase 8: historical text feature development from prior failure descriptions;
  current-claim narratives, identifiers, costs, targets, and text models remain
  prohibited. The output is a companion artifact to Phase 7.
- Phase 9: fixed CatBoost baseline training on TRAIN and champion selection on
  VALIDATION. TEST labels, predictions, and metrics remain sealed for Phase 15.
- Phase 10: CatBoost hyperparameter optimization for only Phase 9 E1 and E3.
  Three expanding same-date-grouped inner folds use outer TRAIN only; outer
  VALIDATION is opened only after the study freeze and only for two finalists.
  Phase 9 remains permanently locked, TEST remains sealed until Phase 15, and
  Phase 11 owns feature selection.
- Phase 11: feature-family ablation and deterministic feature-subset selection
  on the locked Phase 10 parents. Importance uses labeled inner-validation
  pools, candidates are evaluated on the frozen Phase 10 inner folds, and the
  selected subset is frozen before at most two outer-VALIDATION models run.
  Feature addition, hyperparameter retuning, class weighting, resampling,
  early stopping, calibration, threshold tuning, ensembling, and TEST access
  are prohibited. Every fold is checkpointed atomically and can be resumed
  only when its experiment, feature, and parameter hashes still match.
- Later phases: evaluation, calibration, inference, and monitoring.

These boundaries preserve the Phase 0 contract and prevent infrastructure work
from implying that later capabilities already exist.

## Reusable code, notebooks, and generated artifacts

Reusable behavior belongs under src/warranty_analytics_model/. Notebooks are
allowed for exploration and temporary investigation but are not the only
implementation of reusable workflows. Source code is separate from generated
data, model artifacts, reports, and logs; generated contents are ignored by
default while README files document their purpose.

Phase 11 generated artifacts live under
`artifacts/feature_selection/<run_id>/` and reports under
`reports/phase11_feature_selection/<run_id>/`. The committed contract and
configuration are the source of truth; generated outputs are local evidence
and are intentionally excluded from version control.

