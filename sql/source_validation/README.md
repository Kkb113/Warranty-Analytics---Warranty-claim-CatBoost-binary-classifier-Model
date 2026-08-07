# Source validation SQL

Phase 2 uses only fixed, parameterized SQL Server catalog queries. The reviewed
queries are packaged under
`src/warranty_analytics_model/database/sql/` so they remain available in an
installed wheel; this avoids maintaining a second authoritative copy under the
repository-level `sql/` directory.

The queries read `sys.schemas`, `sys.tables`, `sys.columns`, `sys.types`,
`sys.default_constraints`, `sys.computed_columns`, `sys.key_constraints`,
`sys.indexes`, `sys.index_columns`, `sys.foreign_keys`,
`sys.foreign_key_columns`, `sys.partitions`, `sys.views`, and `sys.sequences`.
They use explicit columns and bind parameters. They do not inspect excluded ML
table columns, read business rows, run DML/DDL, call stored procedures, create
temporary objects, or perform full `COUNT(*)` scans. Row counts are partition
estimates from `sys.partitions`.

The loader allow-lists query resource names, and the metadata collector calls
only the focused operations needed for the approved 16-table contract.
