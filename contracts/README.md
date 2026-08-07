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
