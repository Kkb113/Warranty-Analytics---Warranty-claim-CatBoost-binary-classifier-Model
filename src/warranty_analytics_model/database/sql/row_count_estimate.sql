SELECT
    CAST(COALESCE(SUM(CASE WHEN p.index_id IN (0, 1) THEN p.rows ELSE 0 END), 0) AS bigint)
        AS estimated_rows
FROM sys.partitions AS p
INNER JOIN sys.tables AS t ON t.object_id = p.object_id
INNER JOIN sys.schemas AS s ON s.schema_id = t.schema_id
WHERE s.name = :schema_name
  AND t.name = :table_name
  AND p.index_id IN (0, 1);
