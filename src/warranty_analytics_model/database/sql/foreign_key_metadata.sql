SELECT
    fk.name AS constraint_name,
    ps.name AS parent_schema_name,
    pt.name AS parent_table_name,
    pc.name AS parent_column_name,
    rs.name AS referenced_schema_name,
    rt.name AS referenced_table_name,
    rc.name AS referenced_column_name,
    fk.delete_referential_action_desc AS on_delete,
    fk.update_referential_action_desc AS on_update,
    CAST(CASE WHEN fk.is_not_trusted = 1 THEN 0 ELSE 1 END AS bit) AS trusted,
    fkc.constraint_column_id
FROM sys.foreign_keys AS fk
INNER JOIN sys.foreign_key_columns AS fkc ON fkc.constraint_object_id = fk.object_id
INNER JOIN sys.tables AS pt ON pt.object_id = fk.parent_object_id
INNER JOIN sys.schemas AS ps ON ps.schema_id = pt.schema_id
INNER JOIN sys.columns AS pc
    ON pc.object_id = pt.object_id AND pc.column_id = fkc.parent_column_id
INNER JOIN sys.tables AS rt ON rt.object_id = fk.referenced_object_id
INNER JOIN sys.schemas AS rs ON rs.schema_id = rt.schema_id
INNER JOIN sys.columns AS rc
    ON rc.object_id = rt.object_id AND rc.column_id = fkc.referenced_column_id
WHERE ps.name = :schema_name
  AND pt.name = :table_name
ORDER BY fk.name, fkc.constraint_column_id;
