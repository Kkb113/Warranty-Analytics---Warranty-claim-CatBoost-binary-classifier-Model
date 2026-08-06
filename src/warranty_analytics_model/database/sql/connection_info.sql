SELECT
    DB_NAME() AS database_name,
    CAST(SERVERPROPERTY('ServerName') AS nvarchar(128)) AS server_name,
    CAST(SERVERPROPERTY('ProductVersion') AS nvarchar(128)) AS product_version,
    CAST(@@VERSION AS nvarchar(4000)) AS sql_version;
