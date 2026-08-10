# Development setup

## Windows PowerShell

From the repository root:

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev,database,profiling,mart]"

Select the development environment explicitly when needed:

    $env:WARRANTY_MODEL_ENV = "development"

The supported values are development and test. The test environment can be
selected for a command with:

    $env:WARRANTY_MODEL_ENV = "test"

## Other platforms

    python3.11 -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev,database,profiling,mart]"

The package also supports the platform's Python 3.11 launcher where available.

## Optional .env file

Copy .env.example to .env for local non-committed overrides:

    Copy-Item .env.example .env

The file is optional and is never required for the package or tests. It is
ignored by Git. Keep database values empty unless a live Phase 2 check is
explicitly approved. Use a secure local environment for credentials; never put
them in YAML or committed files. Ordinary Phase 1 commands do not connect to
SQL Server.

Configuration precedence is:

    Typed defaults < configs/base.yaml < configs/<environment>.yaml
    < .env < operating-system environment variables

The doctor and show-config commands redact secret values. Do not paste their
output into tickets or logs if local environment metadata is sensitive.

## CLI and quality checks

    python -m warranty_analytics_model --help
    python -m warranty_analytics_model doctor
    python -m warranty_analytics_model show-config
    python -m warranty_analytics_model version
    python -m warranty_analytics_model schema-contract-check
    python -m warranty_analytics_model db-check
    python -m warranty_analytics_model schema-validate
    python -m warranty_analytics_model data-profile
    python -m warranty_analytics_model synthetic-audit
    python -m warranty_analytics_model data-quality-check
    python -m warranty_analytics_model phase3-run --no-charts
    python -m warranty_analytics_model phase4-contract-check
    python -m warranty_analytics_model phase4-validate
    python -m warranty_analytics_model phase5-plan-check
    python -m warranty_analytics_model phase5-build
    python -m warranty_analytics_model phase5-validate --mart-dir artifacts/feature_mart/<run_id>
    python -m ruff check .
    python -m ruff format --check .
    python -m mypy src
    python -m pytest

Install local pre-commit hooks and run them manually:

    pre-commit install
    pre-commit run --all-files

## Deactivate the virtual environment

    deactivate

## Common setup errors

### Python launcher is unavailable

Install Python 3.11 or later and use the platform's Python executable. Confirm
with:

    python --version

### PowerShell blocks Activate.ps1

Use a permitted local execution policy for the current user, or activate the
environment through another supported shell. Do not disable security controls
system-wide solely for this project.

### Package import fails

Confirm the virtual environment is active and rerun:

    python -m pip install -e ".[dev,database,profiling,mart]"

Run commands from the repository root so configuration discovery can locate
pyproject.toml and configs/.

### Configuration fails

Check that configs/base.yaml, configs/development.yaml, and configs/test.yaml
exist. WARRANTY_MODEL_ENV must be development or test. Secret-bearing settings
are not allowed in YAML; use local environment variables instead.

Phase 4 contract validation is offline and database-independent. It validates the
stored target, all 209 field-policy entries, historical as-of rules, allowlists,
lineage metadata, and the hard leakage blacklist. Phase 4 live validation is
read-only and uses the existing local SQL Server configuration; it never creates
a feature mart or writes business data. Reports are generated under
reports/phase4_validation/ and contain aggregate policy and eligibility results only.
