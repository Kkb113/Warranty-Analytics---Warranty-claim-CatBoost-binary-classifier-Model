# Phase 8 — Text Feature Development

Phase 8 creates deterministic, target-independent historical text candidates
for one row per warranty claim. It is an offline companion artifact to the
locked Phase 7 structured-feature artifact and does not train a text model.

## Locked inputs

The validated implementation uses these exact runs:

```text
artifacts/feature_mart/20260810T102230Z/
artifacts/splits/20260810T_PHASE6_CORRECTED/
artifacts/structured_features/20260810T_PHASE7_HARDENED/
```

Phase 8 rechecks the Phase 5 mart, Phase 6 membership and TEST lock, and Phase
7 content hash before publishing. The Phase 8 contract is
`contracts/text_feature_contract_v1.yaml` and the technical settings are in
`configs/text_features.yaml`.

## Source and temporal policy

The only approved text value source is
`prior_failure__failure_description` from the Phase 5 `prior_claim_history`
artifact under `ALLOW_HISTORICAL_POC`. Current-claim narratives, current
failure descriptions, identifiers, costs, target fields, and any unapproved
text source are rejected. Phase 7 structured taxonomy fields remain structured
inputs and are not copied into this artifact.

For each current claim, prior records satisfy
`prior_claim_date < current_claim_date`. The 6m, 12m, and 24m lower boundaries
are inclusive; `all` has no fixed lower bound. Same-day, future, current-record,
and unknown-current-key rows are blocked. Documents are ordered by prior date
and then `prior_warranty_claim_key`, with the key used only as a control. The
separator is exactly `" [SEP] "`; duplicates are preserved.

Text normalization is deterministic: Unicode NFKC, whitespace collapse,
trimming, and casefolding. Punctuation and numbers are preserved. Null or
blank descriptions are excluded, and a claim with no usable history receives a
NULL document. No stemming, lemmatization, spelling correction, stop-word
removal, vectorizer, embedding, LLM, or target-derived vocabulary is used.

## Output contract

The run is published at:

```text
artifacts/text_features/<run_id>/
reports/phase8_text_features/<run_id>/
```

The artifact contains controls `warranty_claim_key`, `split`, and
`claim__claim_date`, four raw historical document columns for 6m/12m/24m/all,
and 29 aggregate lexical/boolean candidates. The four document columns are
text features; the lexical candidates include description, character, token,
unique-value, average-length, and prior-description-presence features. The
artifact has no target and does not duplicate Phase 7 model columns.

Required aggregate metadata includes the feature manifest, lineage, text
quality, run manifest, and validation files. Reports contain only aggregate
counts, coverage, distributions, source policy, validation, hashes, and
warnings; raw text and claim identifiers are not written to reports.

## Commands

```text
warranty-model phase8-contract-check
warranty-model phase8-plan-check --mart-dir <phase5-run> --split-dir <phase6-run> --structured-dir <phase7-run>
warranty-model phase8-build --mart-dir <phase5-run> --split-dir <phase6-run> --structured-dir <phase7-run>
warranty-model phase8-validate --text-dir artifacts/text_features/<run_id>
```

The build is atomic and deterministic. `--overwrite` is required to replace an
existing run ID. `--no-report` suppresses report publication when appropriate.
The Phase 8 contract check is included in CI and does not require SQL Server.

## Current handoff

The validated run is `artifacts/text_features/20260810T_PHASE8/`: 8,500 rows,
33 text model candidates, and 29 lexical candidates. Validation is
**PASS WITH WARNINGS** with zero blocking errors. The warnings are
`HIGH_TEXT_TAXONOMY_REPETITION` and the expected unversioned
failure-description dimension, which requires real-data reapproval before
production use. Phase 8 is safe to hand off to Phase 9 baseline model design
once this expected reapproval is tracked.
