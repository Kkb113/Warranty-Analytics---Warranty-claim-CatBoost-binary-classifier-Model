SELECT
    s.name AS schema_name,
    seq.name AS sequence_name
FROM sys.sequences AS seq
INNER JOIN sys.schemas AS s ON s.schema_id = seq.schema_id
WHERE s.name = :schema_name
ORDER BY seq.name;
