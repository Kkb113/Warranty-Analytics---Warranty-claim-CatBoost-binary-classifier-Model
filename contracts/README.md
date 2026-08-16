# Schema contract

`warranty_analytics_schema_v1.yaml` is the version-controlled Phase 2 contract
for the SQL Server database `warranty_analytics`. It was generated from
`warranty_analytics_schema_document.docx`, whose SHA-256 provenance is recorded
in the YAML. The contract reconciles 16 included tables, 209 columns, 22
documented foreign keys, and 392,352 estimated rows.

Runtime code loads this YAML only; it does not parse the DOCX. The extraction
utility is development-only and refuses to overwrite an existing contract
without an explicit flag:

    python scripts/build_schema_contract.py
    python scripts/build_schema_contract.py --check
    python scripts/build_schema_contract.py --force

Contract changes require an explicit review, a source-document reconciliation,
and a version increment. The three excluded ML dataset tables are retained as
name-only exclusions and must never be inspected, validated, or used as model
inputs.

Phase 4 adds three separate versioned contracts:

    high_cost_target_v1.yaml
    claim_time_feature_policy_v1.yaml
    leakage_policy_v1.yaml

They reference the schema contract version and checksum, classify all 209
included columns, and are validated by:

    warranty-model phase4-contract-check

Phase 5 must consume these contracts and record their exact policy checksums.

The Phase 5 mart contract is `claim_feature_mart_v1.yaml`. It is validated
offline with:

    warranty-model phase5-plan-check

The contract maps all 41 `ALLOW_BASELINE_POC` direct fields and all 43
`ALLOW_HISTORICAL_POC` history fields, defines the six safe history/index
bridges, and forbids excluded ML tables, customer-derived location paths,
current-claim repair values, and current causal-component selection. Its exact
SHA-256 is recorded in every generated Phase 5 manifest. The technical settings
for local Parquet output are in `configs/feature_mart.yaml`.

Phase 6 adds the versioned evaluation-design contract
`claim_split_v1.yaml` and technical settings in `configs/splits.yaml`. The
split contract references the exact Phase 4 policy checksums and Phase 5 mart
contract checksum. Its primary split is chronological and date-preserving;
group exposure and fingerprint-clean cohorts are diagnostics, not alternate
partitions. Validate it offline with:

    warranty-model phase6-contract-check

Phase 6 consumes only an already completed Phase 5 mart. Generated split
assignments, group-exposure artifacts, evaluation cohorts, manifests, test
locks, and reports are local ignored outputs and must not be committed to this
public repository.

Phase 7 adds `structured_feature_contract_v1.yaml` and technical settings in
`configs/structured_features.yaml`. The structured-feature package consumes
the exact Phase 5 mart and corrected Phase 6 split offline, verifies all
membership/test-lock hashes, and publishes one canonical feature matrix with
per-column lineage. Run the database-independent contract check with:

    warranty-model phase7-contract-check

Phase 7 generated feature matrices, manifests, diagnostics, and reports remain
local ignored outputs. The target is sealed for Phase 15; text and repair
details remain deferred. The Phase 7 telemetry coverage denominator counts
only completed calendar months strictly before the claim month; the claim month
is excluded from both the safe-history numerator and the expected-month
denominator. Feature lineage separately records `value_sources` and
`control_sources`; keys and temporal filters used only for alignment, counting,
ordering, or as-of safety are never treated as raw predictive values.

Phase 8 adds `text_feature_contract_v1.yaml` and technical settings in
`configs/text_features.yaml`. It consumes the exact Phase 5 mart, corrected
Phase 6 split, and hardened Phase 7 artifact. The only approved text value
source is `prior_failure__failure_description` from `prior_claim_history` under
the historical POC policy. Current-claim narratives, identifiers, target
fields, costs, vectorizers, embeddings, and fitted text models are prohibited.
The contract check is offline:

    warranty-model phase8-contract-check

Phase 8 uses strict prior-date as-of rules, deterministic document ordering,
NFKC/whitespace/casefold normalization, and 6m/12m/24m/all windows. Its
companion artifacts and aggregate-only reports are generated under ignored
`artifacts/text_features/` and `reports/phase8_text_features/` directories.

Phase 9 adds `baseline_model_contract_v1.yaml` and fixed technical settings in
`configs/baseline_model.yaml`. The contract freezes experiments E0–E4,
CatBoost parameters, model adapters, validation metrics, champion tie-breaks,
and the Phase 15 TEST-target seal. Validate it offline with:

    warranty-model phase9-contract-check

Generated models, validation-only predictions, manifests, and aggregate reports
remain local ignored outputs under `artifacts/baseline_models/` and
`reports/phase9_baseline_models/`.

Phase 10 adds `catboost_optimization.yaml` and
`configs/catboost_optimization.yaml`. It freezes optimization to T1/E1 and
T3/E3, uses three chronological same-date-grouped inner folds from outer TRAIN
only, and requires sequential Optuna studies with no class weighting,
resampling, early stopping, calibration, feature selection, or ensembling.
Outer VALIDATION is opened only after `study_freeze.json` and only two finalists
are evaluated. Phase 9 remains immutable; TEST target access and TEST hashes
remain forbidden until Phase 15, while Phase 11 owns feature selection. Validate
the policy offline with:

    warranty-model phase10-contract-check

Phase 10 artifacts and aggregate reports remain local ignored outputs under
`artifacts/catboost_optimization/` and
`reports/phase10_catboost_optimization/`.
