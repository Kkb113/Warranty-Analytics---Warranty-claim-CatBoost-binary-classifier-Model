# Truck-Warranty High-Cost Claim Prediction — Model Contract

## 1. Document control

| Item | Value |
| --- | --- |
| Project name | Truck-Warranty High-Cost Claim Prediction |
| Phase name | Phase 0 — Context and Model Contract |
| Document status | Draft — requires business and data-owner approval |
| Version | 0.1 |
| Creation date | 2026-08-05 |
| Source schema document | `warranty_analytics_schema_document.docx` |
| Documented database | warranty_analytics |
| Intended reviewers | Warranty business owner, warranty data owner, synthetic-data/schema owner, machine-learning owner, and warranty operations lead |
| Approval status | Not approved |

This contract defines the initial supervised-learning objective and its data, timing,
evaluation, and use constraints. It is the decision record for later implementation
phases; it is not an implementation plan or a model-training result.

## Phase 0 scope boundary

Phase 0 does not train a model, create an ingestion or preprocessing pipeline, perform
feature engineering, implement prediction code, develop an API, deploy a service, or
introduce AI-agent or agent-orchestration requirements. Those activities require an
approved contract and belong to later phases.

## 2. Business problem

Warranty teams need earlier visibility into claims that are likely to become expensive.
The intended model will estimate that risk while a claim can still be prioritized for
review and cost visibility can still improve.

The model is intended to support:

- Early identification of potentially high-cost claims.
- Prioritization of claims for review.
- Earlier warranty-cost visibility.
- Analysis of cost drivers.
- More consistent risk assessment.

The model is decision support. It must not be described or implemented as an automatic
claim-approval or claim-rejection mechanism.

## 3. Prediction question

The core prediction question is:

> Using only information available at initial claim submission, what is the probability
> that this warranty claim will become a high-cost claim?

The prediction is a probability estimate for one claim. Explanation is a separate
question: which available factors are associated with the estimate, and are those
associations stable and valid? Business action is a third question: what review or
workflow, if any, should follow a probability? A prediction does not by itself provide
causal explanation, approve or reject a claim, charge a customer, penalize a supplier,
or determine an employee action.

## 4. Target definition

| Contract item | Definition |
| --- | --- |
| Target column | dbo.fact_warranty_claim.high_cost_claim_flag |
| Documented SQL type | bit |
| Documented nullability | No |
| Target type | Binary classification |
| Positive class | high_cost_claim_flag = 1 |
| Negative class | high_cost_claim_flag = 0 |
| Prediction unit | One warranty claim |

The schema document does not explain how the synthetic data generator defines the
high-cost threshold. No monetary threshold is invented in this contract.

The following confirmations are required before the target is treated as production
ready:

- Determine whether high_cost_claim_flag was calculated from total_claim_cost.
- Determine the monetary threshold or generation rule.
- Determine whether the definition varies by claim type, component, country, policy,
  or vehicle class.
- Confirm whether the target is suitable for production use or only for synthetic
  proof-of-concept work.

The target is an outcome label and is therefore unavailable as an input feature at
scoring time. Records with a null target cannot be used for supervised training or
evaluation until an explicit data policy is approved.

## 5. Prediction population and eligibility

The initial population is warranty-claim records from dbo.fact_warranty_claim.

An eligible record should initially require:

- A valid warranty claim key.
- A valid claim date.
- A non-null target.
- A linked truck record.
- Sufficient information to create a claim-time snapshot.

These are initial eligibility requirements, not silently resolved data-cleaning rules.
The contract does not yet decide how to handle duplicate keys, invalid dates, missing
truck links, null values, contradictory dates, claims with incomplete history, or
claims whose target is unavailable. Those rules must be confirmed by the data owner
and warranty business owner before future pipeline work.

Eligibility must be evaluated using information available to the future training
snapshot process. Excluding a record because of information that would only be known
after the prediction point would itself create a selection or temporal-leakage risk.

## 6. Unit of observation

One dataset row must represent one warranty claim.

One-to-many tables such as telemetry, maintenance events, repair lines, component
installations, prior service events, and prior claims must eventually be aggregated
before being joined to a claim-level dataset. Direct one-to-many joins must not
duplicate claim labels, inflate row counts, or make a claim appear to have multiple
independent outcomes.

The claim key must remain traceable through any history construction and aggregation.
Any later feature specification must state its as-of date, lookback window, and
aggregation grain.

## 7. Prediction timestamp

The intended scoring point is:

> Initial claim submission

The current schema documents dates rather than precise claim-submission timestamps.
claim_date is therefore the provisional date-level prediction reference.

This is a limitation and not a claim that claim_date records the exact submission
instant:

- The schema does not document a claim-submission timestamp.
- Date-level fields may not establish the exact sequence of events occurring on the
  same day.
- A precise timestamp should be added for production-grade as-of processing.

Later data work must establish an as-of rule for every candidate field and history
record. If exact event order cannot be established, the field must remain uncertain
or be excluded until the business accepts the resulting limitation.

## 8. Candidate source tables

The following are approved as candidate source tables for inspection and contract
design. Candidate-source approval does not approve every column in a table as a
model feature. A field may be used only after its meaning, timing, quality, and
leakage risk are confirmed.

| Candidate source table | Purpose | Relationship to the prediction |
| --- | --- | --- |
| dbo.fact_warranty_claim | Target, claim-time values, mileage, engine hours, months in service, and claim attributes. | The claim-level base table and label source. Current-claim outcome and post-outcome values remain prohibited. |
| dbo.dim_date | Calendar attributes such as full date, month, quarter, and year. | May support date-level joins for claim_date and other confirmed date fields; it must not create future-looking information. |
| dbo.dim_truck | Manufacturing plant, assembly line, production batch, build date, delivery date, in-service date, fuel type, warranty policy, and truck attributes. | Provides truck attributes linked to the claim truck, subject to availability at claim submission and identifier/group controls. |
| dbo.dim_truck_model | Brand, model, model year, application type, engine platform, segment, cab type, and GVWR class. | Provides model and vehicle-class context for the truck associated with the claim. |
| dbo.fact_telemetry_monthly | Historical mileage, engine usage, idle hours, temperatures, oil-pressure events, brake alerts, battery alerts, fault counts, payload, fuel efficiency, route severity, and maintenance compliance. | Historical telemetry may contribute only through records available before the claim prediction reference date. |
| dbo.fact_component_installation | Installed components, supplier, component lot, installation station, inspection results, rework status, torque, and production genealogy. | Provides pre-claim component and production history; current-claim or post-outcome changes require exclusion. |
| dbo.dim_component | Component system, category, expected life, safety criticality, supplier, and unit cost. | Provides component context when the component relationship and availability at submission are confirmed. |
| dbo.dim_supplier | Supplier geography, tier, quality rating, and preferred-supplier status. | Provides supplier context for pre-claim installed components or other confirmed relationships; high-cardinality use requires controlled evaluation. |
| dbo.fact_maintenance_event | Historical maintenance timing, mileage, engine hours, completion compliance, overdue days, costs, and available notes. | Historical maintenance may be aggregated only from events available before the prediction point. Current-claim or post-outcome activity is prohibited. |
| dbo.fact_service_event | Historical service activity, complaint descriptions, diagnostic summaries, downtime, roadside assistance, and repeat visits. | Prior service history may be used when time and capture status are established. Current-claim diagnostics and later service activity are not provisionally available. |
| dbo.fact_repair_line | Historical repair actions, parts, labor, costs, technicians, and repair notes. | Historical, completed records before the current claim may be inspected; current-claim repair lines must not be available at initial submission. |
| dbo.dim_service_center | Dealer group, certification, labor rate, capacity, and location. | Provides service-center context if the center is known at submission and the relationship is confirmed. |
| dbo.dim_customer | Customer type, industry, fleet size, contract type, and account priority. | Provides customer context when the attribute is available at submission and permitted for the intended use. |
| dbo.dim_location | Region, country, state, climate zone, and terrain. | Provides location context when the relevant location is known at submission. |
| dbo.dim_warranty_policy | Coverage months, mileage, engine hours, deductible, and coverage type. | Provides policy and coverage context available at submission, subject to confirmation of policy version and effective date. |
| dbo.dim_failure_code | Failure taxonomy and safety attributes. | A candidate source only if the failure-code fields are populated at claim submission; availability requires confirmation. |

The source document explicitly excludes these tables from approved training sources:

- dbo.ml_region_terrain_warranty_risk_dataset
- dbo.ml_truck_failure_risk_dataset
- dbo.ml_truck_region_terrain_failure_risk_dataset

They remain excluded pending separate inspection of their columns, label-generation
logic, and leakage risk. They must not be used as candidate training sources under
this contract.

## 9. Feature-availability classification

Availability is defined relative to initial claim submission, not relative to the
eventual claim close date. The categories below are contract classifications, not a
substitute for field-level profiling.

### A. Provisionally available at claim submission

The following may be used as candidates if the data owner confirms they are captured
and stable at the scoring point:

- claim_date.
- truck_model_key joined to dbo.dim_truck_model.
- manufacturing_plant.
- assembly_line.
- production_batch_id.
- build_date.
- delivery_date.
- in_service_date.
- odometer_miles_at_failure.
- engine_hours_at_failure.
- months_in_service.
- warranty_coverage_status.
- claim_type.
- service_center_key joined to dbo.dim_service_center.
- Historical telemetry before the claim.
- Historical maintenance before the claim.
- Historical claims before the current claim.
- Component installations completed before the claim.
- Customer and location attributes.
- complaint_description, when captured at submission.

These are provisionally available examples, not approved feature columns. Each later
feature must have an explicit source, as-of rule, lookback rule, missingness rule, and
leakage review.

### B. Requires business confirmation

The following fields or concepts may be useful, but their availability can depend on
the warranty process, who enters the value, and when the value is finalized:

- failure_code_key
- causal_component_key
- failure_mode
- root_cause_category
- diagnostic_summary
- repair_start_date
- causal_part_no
- claim_status

Confirmation must establish whether each field is present at initial submission,
whether it is later revised, whether the value is derived from diagnosis or repair,
and whether its use would encode the eventual outcome. A field that is present in the
database but populated after submission remains unavailable for this model.

### C. Prohibited because it is post-outcome or direct leakage

The following are prohibited for the initial high-cost claim model:

- high_cost_claim_flag.
- total_claim_cost.
- labor_cost.
- parts_cost.
- diagnostic_cost.
- towing_cost.
- other_cost.
- approved_amount.
- rejected_amount.
- customer_paid_amount.
- repair_end_date.
- days_to_repair.
- Current-claim repair-line costs.
- Current-claim line_cost values from dbo.fact_repair_line.
- Current-claim labor_hours from dbo.fact_repair_line.
- Current-claim repair_notes from dbo.fact_repair_line.
- Final diagnostic results.

The suspicious fields listed in the leakage blacklist below are also excluded unless
the data owner and machine-learning owner document evidence that they are known before
the prediction point and do not encode the outcome.

## 10. Leakage blacklist

The following fields and data classes are blacklisted for the initial high-cost claim
model. Blacklisted means they must not be used as model inputs unless this contract is
formally revised after a documented timing and semantic review.

### Explicitly prohibited fields and outcomes

- high_cost_claim_flag
- total_claim_cost
- labor_cost
- parts_cost
- diagnostic_cost
- towing_cost
- other_cost
- approved_amount
- rejected_amount
- customer_paid_amount
- repair_end_date
- days_to_repair

### Post-outcome or suspicious unless proven otherwise

- claim_status
- root_cause_category
- repeat_claim_flag
- potential_recall_flag
- Current-claim repair-line costs.
- Current-claim labor hours.
- Current-claim repair notes.
- Final diagnostic results.
- Any current-claim repair or diagnosis record created or changed after initial
  submission.

The current claim's repair lines are not historical features, even if they are stored
in dbo.fact_repair_line. Only prior records that can be proven to precede the
prediction reference date may be considered.

### Leakage types

1. Direct leakage: a field mathematically defines or directly reveals the target.
   Costs, approved amounts, and the target flag itself are examples.
2. Temporal leakage: a field became available only after the prediction time, even
   if it is not mathematically related to the target. Final diagnosis and repair
   completion are examples.
3. Group leakage: related records from the same truck, batch, lot, or synthetic
   scenario appear across training and test data in a way that lets the model
   memorize shared structure rather than generalize.
4. Synthetic-generator leakage: IDs, templates, dates, deterministic generation
   rules, scenario markers, or other artifacts reveal how the target was generated.

The leakage review must cover not only selected features but also joins, aggregation
windows, imputation rules, split construction, and any generated dataset or feature
mart used in later phases.

## 11. Identifier policy

Identifiers may be used for joins, grouping, history construction, and data splitting,
but should not automatically be used as model features. Their presence in a source
table is not evidence that they carry valid predictive meaning.

The following identifiers are join, grouping, or control fields by default and require
explicit review before feature use:

- warranty_claim_key
- claim_id
- truck_key
- vin
- service_event_key
- component_serial_no
- engine_serial_no
- transmission_serial_no
- technician_id
- inspector_id

High-cardinality fields such as production_batch_id, component_lot_no, supplier_key,
component_key, and service_center_key require controlled evaluation. They may contain
useful group information, but they may also cause memorization, proxy discrimination,
unstable performance, or leakage across train and test data.

The warranty claim key may be returned as output metadata and used to preserve row
traceability while remaining excluded from learned features by default.

## 12. Model output contract

The primary output is:

    high_cost_probability: float between 0 and 1

Illustrative response shape:

    {
      "warranty_claim_key": 12345,
      "high_cost_probability": 0.82,
      "model_version": "future-version"
    }

The example does not establish a threshold, model version, or expected probability
for any real claim.

Possible metadata fields include:

- warranty_claim_key
- model_version
- feature_version
- prediction_timestamp
- prediction_reference_date
- decision_threshold
- risk_category
- explanation fields

decision_threshold and risk_category are downstream decisions. They are not part of
the learned target, and thresholds must not be defined in Phase 0 unless a confirmed
business rule already exists. Explanation fields must be documented separately from
the probability and must not be presented as causal findings without appropriate
analysis.

## 13. Evaluation strategy

Overall accuracy must not be the only evaluation metric. Required future metrics are:

- PR-AUC.
- ROC-AUC.
- Precision.
- Recall.
- F1 score.
- Balanced accuracy.
- Confusion matrix.
- Brier score.
- Calibration curve.
- Positive-class prevalence.

PR-AUC should be the provisional primary ranking metric until the target distribution
is known. Numeric success thresholds must not be invented in Phase 0. Numeric
acceptance targets should be defined after data profiling and baseline training.

The future model must be compared with:

- A majority-class baseline.
- A rule-based baseline.
- A logistic-regression baseline.

The rule-based baseline and its inputs must be specified later without using
blacklisted or post-outcome information. Evaluation must report the positive-class
prevalence and enough counts to make precision, recall, calibration, and operational
capacity interpretable.

## 14. Validation principles

The required future validation design is:

- Chronological training, validation, and test split.
- Untouched final test period.
- No random row-only split as the primary evaluation.
- Additional evaluation on unseen trucks.
- Additional evaluation on unseen production batches.
- Additional evaluation on unseen component lots.
- Additional evaluation on unseen suppliers.
- Additional evaluation on unseen service centers.
- Separation of synthetic scenario templates or generator seeds when possible.

The split is not implemented in Phase 0. Later split design must prevent temporal,
group, and synthetic-generator leakage and must document how claims with shared truck,
batch, lot, supplier, service-center, or scenario relationships are assigned.

## 15. Model limitations

The initial model contract has the following known limitations:

- The current data is synthetic.
- Synthetic test accuracy is not production accuracy.
- The target-generation formula is not documented in the schema.
- The schema has dates but not precise event timestamps.
- Software-version, DTC-event, production-shift, corrective-action, and
  reviewer-decision data are not documented.
- The three excluded ML tables have not been assessed.
- The effective claim-level supervised sample size is approximately 8,500 claims,
  subject to eligibility filtering.
- Some categorical groups may have very few observations.
- Current data may contain generator shortcuts that inflate validation results.

The schema document covers 16 included tables, 209 columns, and approximately 392,352
rows by catalog estimates. The included table inventory reports 8,500 rows for
dbo.fact_warranty_claim; the effective supervised sample remains subject to eligibility
and label-quality checks.

## 16. Intended use

Permitted intended uses are:

- Estimate high-cost probability.
- Prioritize claims for human review.
- Analyze major risk drivers.
- Support warranty-cost planning.
- Compare patterns across valid business segments.

Any operational use must preserve human and business-process review and must use only
data available at the agreed prediction point.

## 17. Prohibited use

The model must not be used for:

- Automatic claim rejection.
- Automatic claim approval.
- Automatic customer charging.
- Automatic supplier penalties.
- Employee or technician disciplinary decisions.
- Reporting synthetic accuracy as verified real-world performance.
- Scoring with post-outcome data.
- Using the output without human and business-process review.

## 18. Confirmed facts and assumptions

### Confirmed facts for this contract

The following are treated as documented facts supplied by the Phase 0 brief:

- The documented database is warranty_analytics.
- The target column is dbo.fact_warranty_claim.high_cost_claim_flag.
- The source schema is described as containing 16 tables, 209 columns, and
  approximately 392,352 rows.
- The source document identifies the fact and dimension tables listed in this
  contract.
- The source document explicitly excludes the three dbo.ml_* tables listed above.
- The current data is synthetic.
- The effective claim-level supervised sample size is approximately 8,500 claims,
  subject to eligibility filtering.
- The schema documents dates but not precise claim-submission timestamps.

The source document was inspected from its repository copy,
`warranty_analytics_schema_document.docx`. It documents the table inventory, columns, keys, relationships, catalog row estimates,
and the explicit exclusion of the three dbo.ml_* tables. It does not document the
synthetic target-generation formula, claim-submission timestamps, or the operational
time at which every claim field becomes available. Those remain open questions.

### Assumptions requiring approval

The following are assumptions or provisional interpretations, not confirmed facts:

- claim_date approximates the prediction date.
- complaint_description is available at claim submission.
- Odometer and engine hours at failure are available at submission.
- Historical telemetry is loaded before scoring.
- Historical data can be reconstructed using claim date.
- The target is consistently generated across all claims.

Each assumption must be converted into a confirmed data or business rule, or the
affected field or use case must be removed before later implementation.

## 19. Decision log

| Decision | Current status | Reason | Owner | Required approval |
| --- | --- | --- | --- | --- |
| Initial target selected: dbo.fact_warranty_claim.high_cost_claim_flag | Proposed | It is the documented binary outcome aligned to the high-cost claim objective. | Warranty business owner and ML owner | Business and data-owner approval; target-generation confirmation |
| Prediction timing selected: initial claim submission | Provisional | This is the earliest useful intervention point named by the business objective. | Warranty operations lead | Business-process approval and field-availability confirmation |
| claim_date used as the provisional date-level prediction reference | Provisional | The schema documents dates but not a precise claim-submission timestamp. | Data owner and ML owner | Data-owner approval; production timestamp decision |
| Claim-level grain selected: one row per warranty claim | Proposed | It matches the prediction unit and prevents one-to-many joins from duplicating labels. | ML owner and data owner | Data-model approval |
| Synthetic data accepted for development only | Accepted for development only | Synthetic data can support proof-of-concept development but cannot establish production performance. | Business owner and data owner | Explicit limitation acknowledgement |
| Excluded ML tables remain out of scope | Proposed | Their columns, label-generation logic, and leakage risk have not been separately inspected. | Data owner and synthetic-data owner | Data-owner approval before any future inspection or inclusion |
| Numeric success threshold remains pending | Pending | The positive-class prevalence, review capacity, and cost trade-offs are not established. | Warranty operations lead and business owner | Business acceptance criteria after profiling and baseline training |
| High-cost target formula remains pending confirmation | Blocking | The schema does not document the monetary threshold or synthetic generation rule. | Synthetic-data owner and data owner | Formal target-definition approval |

Statuses in this table do not constitute approval. They indicate the current Phase 0
decision state and the evidence still required.

## 20. Phase 0 approval checklist

### Business owner

- [ ] Business objective approved.
- [ ] Target approved.
- [ ] Target-generation rule confirmed.
- [ ] Prediction timing approved.
- [ ] Eligible population approved.
- [ ] Allowed feature classes approved.
- [ ] Leakage blacklist approved.
- [ ] Intended use approved.
- [ ] Prohibited use approved.
- [ ] Evaluation metrics approved.
- [ ] Synthetic-data limitation acknowledged.
- [ ] Open questions assigned to owners.

### Data owner

- [ ] Business objective approved.
- [ ] Target approved.
- [ ] Target-generation rule confirmed.
- [ ] Prediction timing approved.
- [ ] Eligible population approved.
- [ ] Allowed feature classes approved.
- [ ] Leakage blacklist approved.
- [ ] Intended use approved.
- [ ] Prohibited use approved.
- [ ] Evaluation metrics approved.
- [ ] Synthetic-data limitation acknowledged.
- [ ] Open questions assigned to owners.

### Machine-learning owner

- [ ] Business objective approved.
- [ ] Target approved.
- [ ] Target-generation rule confirmed.
- [ ] Prediction timing approved.
- [ ] Eligible population approved.
- [ ] Allowed feature classes approved.
- [ ] Leakage blacklist approved.
- [ ] Intended use approved.
- [ ] Prohibited use approved.
- [ ] Evaluation metrics approved.
- [ ] Synthetic-data limitation acknowledged.
- [ ] Open questions assigned to owners.

## Phase 0 handoff criteria

Phase 0 is complete when this contract and the open-questions register are reviewed,
the target and timing are approved, eligibility and feature availability are
confirmed, leakage controls are accepted, and unresolved items have named owners.
No model training or feature-pipeline implementation is part of this handoff.
