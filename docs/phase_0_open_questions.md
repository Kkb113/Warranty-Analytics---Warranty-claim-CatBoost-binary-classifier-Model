# Phase 0 Open Questions

## Document control

| Item | Value |
| --- | --- |
| Project | Truck-Warranty High-Cost Claim Prediction |
| Phase | Phase 0 — Context and Model Contract |
| Status | Draft — requires business, data-owner, and machine-learning review |
| Version | 0.1 |
| Last updated | 2026-08-05 |
| Companion document | docs/model_contract.md |
| Source schema document | `warranty_analytics_schema_document.docx` |

This register records unresolved questions that must be answered without guessing.
The current answers distinguish what is documented in the schema from what requires
business-process, synthetic-generator, or data-quality confirmation.

Priority meanings:

- Blocking: must be resolved before the stated future phase can begin safely.
- High: materially affects target validity, leakage control, business value, or
  evaluation design.
- Medium: important for robust implementation but can follow the blocking contract
  decisions.
- Low: useful for later operational refinement and documentation.

## Blocking questions

| ID | Question | Reason it matters | Proposed owner | Required before which future phase | Current answer, if available |
| --- | --- | --- | --- | --- | --- |
| B-01 | What exact monetary threshold or logic creates dbo.fact_warranty_claim.high_cost_claim_flag? | The target cannot be interpreted, audited, or aligned to business cost without knowing how the positive class was generated. | Synthetic-data owner and data owner | Phase 4 — target definition and leakage enforcement | Not documented in the schema document. The target column exists, but its synthetic generation formula and threshold are not provided. |
| B-02 | Is the high-cost threshold fixed or dependent on vehicle, component, policy, geography, or claim type? | A varying rule changes label comparability, segment evaluation, calibration, and the meaning of a probability. | Warranty business owner, synthetic-data owner, and data owner | Phase 4 — target definition and leakage enforcement | Not documented. The contract must not assume a fixed or conditional threshold. |
| B-03 | At what exact point should the model score a claim? | Every feature and historical aggregation depends on a precise as-of point; a vague timing rule creates temporal leakage. | Warranty operations lead and business owner | Phase 4 — target definition and leakage enforcement | The contract proposes initial claim submission. The schema documents claim_date but no precise claim-submission timestamp. |
| B-04 | Which claim fields exist at the agreed scoring point? | Database presence does not prove operational availability. The answer controls the allowed feature set and prevents use of fields populated later. | Warranty operations lead and data owner | Phase 4 — target definition and leakage enforcement | The schema documents fields in dbo.fact_warranty_claim, but it does not document the workflow time at which each field is populated. |
| B-05 | Is complaint_description available at submission? | Complaint text may be a useful early signal, but using it is valid only if it is captured before or at the prediction point. | Warranty operations lead and service-process owner | Phase 4 — target definition and leakage enforcement | complaint_description exists in dbo.fact_service_event. Its capture timing for the current claim is not documented. |
| B-06 | Is diagnostic_summary available at submission or only after diagnosis? | Diagnostic text can directly encode failure severity or outcome if it is written after inspection or repair begins. | Warranty operations lead and service-process owner | Phase 4 — target definition and leakage enforcement | diagnostic_summary exists in dbo.fact_service_event. The schema does not establish whether it is available at submission; it remains uncertain. |
| B-07 | When are failure_code_key, causal_component_key, failure_mode, and root_cause_category populated? | These fields can be useful taxonomy or component context, but they may be assigned after diagnosis and therefore leak the outcome. | Warranty operations lead, data owner, and warranty adjudication owner | Phase 4 — target definition and leakage enforcement | All four fields are documented in or linked to the claim schema, but their business-process timing is not documented. They require confirmation individually. |
| B-08 | Are repair records connected to the current claim available before the high-cost outcome is known? | Current-claim repair lines can contain costs, labor hours, parts, repair notes, or final actions that reveal the outcome. | Warranty operations lead and data owner | Phase 4 — target definition and leakage enforcement | dbo.fact_repair_line contains warranty_claim_key and repair details. The schema does not document availability timing; current-claim repair data is prohibited by the contract. |
| B-09 | Can the synthetic generator source code or SQL be inspected? | Generator logic may define the target, create deterministic shortcuts, or reveal fields that must be excluded from validation. | Synthetic-data owner and data owner | Phase 4 — target definition and leakage enforcement | No generator source is included in the schema document. It must be located or explicitly declared unavailable. |
| B-10 | Do synthetic scenario IDs, seeds, or templates exist for leakage-safe splitting? | Chronological or row-only splits may look strong if the same generated scenario appears in train and test. Group separation may require generator metadata. | Synthetic-data owner and machine-learning owner | Phase 6 — train/validation/test split design | The schema document does not document scenario IDs, seeds, or templates. Their existence and meaning are unknown. |

## High-priority questions

| ID | Question | Reason it matters | Proposed owner | Required before which future phase | Current answer, if available |
| --- | --- | --- | --- | --- | --- |
| H-01 | What business cost is associated with a false negative? | Missing a high-cost claim may cause avoidable warranty expense or delayed intervention; this cost affects recall and review design. | Warranty finance owner and business owner | Phase 12 — class imbalance and threshold policy | Not provided. No cost value or decision rule is documented. |
| H-02 | What business cost is associated with a false positive? | Excessive review of lower-cost claims consumes warranty capacity and may disrupt service operations; this cost affects precision and threshold selection. | Warranty operations lead and business owner | Phase 12 — class imbalance and threshold policy | Not provided. No cost value or decision rule is documented. |
| H-03 | How many claims can the warranty team review? | Review capacity is needed to translate probabilities into an operational workload and to evaluate precision at a realistic review volume. | Warranty operations lead | Phase 12 — class imbalance and threshold policy | Not provided. A risk category or decision threshold is intentionally not defined in Phase 0. |
| H-04 | What minimum recall or precision will the business eventually require? | Numeric acceptance criteria must reflect business trade-offs and must not be invented from synthetic data alone. | Business owner and warranty operations lead | Phase 12 — class imbalance and threshold policy | Not provided. The contract requires metrics but leaves numeric success thresholds pending. |
| H-05 | Will real warranty data become available for final validation? | Synthetic validation cannot establish production accuracy or real-world calibration. A real-data plan is needed for credible deployment decisions. | Data owner and business owner | Phase 19 — real-data validation | Not confirmed. The current source is synthetic, and no real-data delivery commitment is documented. |
| H-06 | Are the three excluded ML tables intended as training outputs, feature marts, or test datasets? | Their role determines whether they duplicate labels, encode generated outcomes, or are safe to inspect later; they remain excluded until assessed. | Data owner and synthetic-data owner | Phase 4 — target definition and leakage enforcement | The schema explicitly excludes dbo.ml_region_terrain_warranty_risk_dataset, dbo.ml_truck_failure_risk_dataset, and dbo.ml_truck_region_terrain_failure_risk_dataset from its documented sections. Their intended role and columns are not documented. |
| H-07 | Does the source system provide a precise claim-submission timestamp or only claim_date? | Date-only sequencing cannot prove whether same-day telemetry, diagnosis, repair, or status updates preceded scoring. | Data owner and warranty operations lead | Phase 4 — target definition and leakage enforcement | The inspected schema documents claim_date as date and does not document a claim-submission timestamp. |
| H-08 | Which fields are revised after initial submission, and are historical versions retained? | A value that appears in the current row may be a final value rather than the value known at submission. Revision history is necessary for leakage-safe reconstruction. | Data owner and warranty process owner | Phase 4 — target definition and leakage enforcement | No change history or field-versioning rule is documented in the schema. |

## Medium-priority questions

| ID | Question | Reason it matters | Proposed owner | Required before which future phase | Current answer, if available |
| --- | --- | --- | --- | --- | --- |
| M-01 | What constitutes a valid warranty claim key, and how should duplicate or invalid claim records be handled? | The unit of observation is one claim; duplicate or invalid keys can duplicate labels, distort prevalence, and break traceability. | Data owner and warranty business owner | Phase 4 — target definition and leakage enforcement | warranty_claim_key is documented as the primary key of dbo.fact_warranty_claim and is non-null in the schema. Data-quality validity and duplicate handling still require profiling and approval. |
| M-02 | How should claims with invalid, contradictory, or incomplete dates be handled? | Date quality affects eligibility, chronological splits, historical lookbacks, and the claim-time snapshot. | Data owner and warranty business owner | Phase 4 — target definition and leakage enforcement | The schema defines date columns but does not provide a handling policy for invalid or contradictory dates. |
| M-03 | Are historical telemetry, maintenance, service, repair, component-installation, and prior-claim records loaded before each scoring event? | A source table can exist while its latest records are delayed; delayed loading changes whether a history feature is available at scoring time. | Data owner and platform owner | Phase 4 — target definition and leakage enforcement | The schema documents the tables and date fields but not data-refresh timing or availability guarantees. |
| M-04 | What lookback windows and minimum history requirements are acceptable for each historical source? | Aggregations need consistent as-of windows and a documented policy for trucks with little or no history. | Machine-learning owner and warranty business owner | Phase 5 — claim-level feature mart construction | No lookback or minimum-history rule is approved. The contract requires later feature definitions to state both. |
| M-05 | Which production batches, component lots, suppliers, service centers, and trucks must be held out for group evaluation? | High-cardinality groups can lead to memorization or leakage; group-level performance is needed to judge generalization. | Machine-learning owner and data owner | Phase 6 — train/validation/test split design | The contract requires unseen-group evaluation, but the exact group assignment and minimum group sizes are not yet defined. |
| M-06 | Are customer, location, supplier, technician, and inspector attributes permitted for the intended business use? | Some fields may be useful but can create governance, proxy, privacy, or fairness concerns and must be approved before feature use. | Business owner, data owner, and governance owner | Phase 4 — target definition and leakage enforcement | No field-governance decision is documented. Identifier policy excludes identifiers by default but does not settle the use of all descriptive attributes. |
| M-07 | How should missing, null, or not-yet-available candidate fields be represented in the claim-time snapshot? | Missingness may reflect process timing rather than truck risk; imputation or missing indicators can introduce leakage if not designed carefully. | Data owner and machine-learning owner | Phase 4 — target definition and leakage enforcement | No missingness or imputation policy is approved in Phase 0. |

## Low-priority questions

| ID | Question | Reason it matters | Proposed owner | Required before which future phase | Current answer, if available |
| --- | --- | --- | --- | --- | --- |
| L-01 | Should the probability output be recalibrated for a future real-data population? | Calibration can drift when synthetic and real prevalence or cost patterns differ; this affects how users interpret the probability. | Machine-learning owner and business owner | Phase 19 — real-data validation | No calibration-recalibration decision exists. Brier score and calibration curves are required future metrics. |
| L-02 | What metadata retention and lineage are required for warranty_claim_key, model_version, feature_version, and prediction timestamps? | Traceable predictions are needed for review, audit, reproducibility, and later error analysis. | Data owner and platform owner | Phase 4 — target definition and leakage enforcement | The output contract lists possible metadata, but retention and lineage requirements are not approved. |
| L-03 | Will the business later require Low, Medium, or High risk categories? | Categories can simplify workflow but require approved thresholds and should not be confused with the learned probability. | Business owner and warranty operations lead | Phase 12 — class imbalance and threshold policy | A risk category is optional downstream metadata. No category thresholds are defined in Phase 0. |
| L-04 | Which explanation fields should accompany a probability, and how should they be communicated to reviewers? | Explanation outputs can be mistaken for causes or certainty if their scope and language are not controlled. | Machine-learning owner and business owner | Phase 16 — explainability and model card | The output contract permits explanation fields but does not select an explanation method or make causal claims. |

## Resolution protocol

Each question should be closed with one of the following outcomes:

- Confirmed and incorporated into the model contract.
- Confirmed as unavailable, with the affected feature or use case removed.
- Deferred with an explicit owner, rationale, and risk acceptance.
- Rejected because the information is post-outcome, directly leaky, or otherwise
  outside the intended use.

No question in this register should be marked resolved solely because a similarly
named column exists in the database. The data owner and the relevant business owner
must confirm operational timing and meaning.
