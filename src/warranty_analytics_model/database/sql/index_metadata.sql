SELECT
    i.name AS index_name,
    i.type_desc AS index_type,
    i.is_unique,
    i.is_disabled,
    i.filter_definition,
    c.name AS column_name,
    ic.key_ordinal,
    ic.is_included_column
FROM sys.indexes AS i
INNER JOIN sys.tables AS t ON t.object_id = i.object_id
INNER JOIN sys.schemas AS s ON s.schema_id = t.schema_id
INNER JOIN sys.index_columns AS ic
    ON ic.object_id = i.object_id AND ic.index_id = i.index_id
INNER JOIN sys.columns AS c
    ON c.object_id = ic.object_id AND c.column_id = ic.column_id
WHERE i.index_id > 0
  AND i.is_hypothetical = 0
  AND i.name IS NOT NULL
  AND s.name = :schema_name
  AND t.name = :table_name
ORDER BY i.name, ic.is_included_column, ic.key_ordinal, ic.index_column_id;
