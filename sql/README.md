# SQL directory policy

The SQL directory is reserved for version-controlled queries that use the
documented warranty_analytics schema names.

- sql/source_validation/: future Phase 2 source-table and schema validation.
- sql/target_queries/: future target construction work after the Phase 0 target
  rule is confirmed.
- sql/feature_queries/: future feature-source queries after claim-time
  availability and leakage rules are approved.

SQL must not contain credentials. Phase 2 will implement source validation and
data access. Phase 4 or later will implement target and feature queries as
appropriate. No SQL query is added merely as a placeholder in Phase 1.
