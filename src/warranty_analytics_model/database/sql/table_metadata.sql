SELECT
    s.name AS schema_name,
    t.name AS table_name,
    t.is_memory_optimized,
    t.temporal_type_desc,
    t.is_filetable
FROM sys.tables AS t
INNER JOIN sys.schemas AS s ON s.schema_id = t.schema_id
WHERE s.name = :schema_name
  AND t.name = :table_name;
