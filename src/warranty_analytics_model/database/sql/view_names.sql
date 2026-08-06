SELECT
    s.name AS schema_name,
    v.name AS view_name
FROM sys.views AS v
INNER JOIN sys.schemas AS s ON s.schema_id = v.schema_id
WHERE s.name = :schema_name
ORDER BY v.name;
