# Controlled data area

Do not commit warranty extracts or database exports. Do not commit customer or
VIN-level data, credentials, or connection strings in data files.

Future data locations must be configured through the project configuration
system; they must not be hardcoded in source code. Synthetic data is still
controlled project data and must be handled with the same care as real data.

Generated contents of this directory are ignored by default. Keep only
documentation and deliberately reviewed, non-sensitive metadata in source
control.
