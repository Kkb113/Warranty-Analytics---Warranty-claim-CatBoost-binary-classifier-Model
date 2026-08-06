SELECT TOP (1)
    s.name AS schema_name
FROM sys.schemas AS s
WHERE s.name = :schema_name;
