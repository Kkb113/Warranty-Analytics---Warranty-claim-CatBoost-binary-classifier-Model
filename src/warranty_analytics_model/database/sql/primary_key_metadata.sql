SELECT
    kc.name AS constraint_name,
    kc.is_system_named,
    c.name AS column_name,
    ic.key_ordinal
FROM sys.key_constraints AS kc
INNER JOIN sys.tables AS t ON t.object_id = kc.parent_object_id
INNER JOIN sys.schemas AS s ON s.schema_id = t.schema_id
INNER JOIN sys.index_columns AS ic
    ON ic.object_id = kc.parent_object_id AND ic.index_id = kc.unique_index_id
INNER JOIN sys.columns AS c
    ON c.object_id = ic.object_id AND c.column_id = ic.column_id
WHERE kc.type = 'PK'
  AND s.name = :schema_name
  AND t.name = :table_name
ORDER BY ic.key_ordinal;
