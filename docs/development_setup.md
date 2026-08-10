# Development setup

## Windows PowerShell

From the repository root:

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev,database,profiling,mart,modeling]"

Select the development environment explicitly when needed:

    $env:WARRANTY_MODEL_ENV = "development"

The supported values are development and test. The test environment can be
selected for a command with:

    $env:WARRANTY_MODEL_ENV = "test"

## Other platforms

    python3.11 -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev,database,profiling,mart,modeling]"

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
    python -m warranty_analytics_model phase6-contract-check
    python -m warranty_analytics_model phase6-plan-check --mart-dir artifacts/feature_mart/<run_id>
    python -m warranty_analytics_model phase6-build --mart-dir artifacts/feature_mart/<run_id>
    python -m warranty_analytics_model phase6-validate --split-dir artifacts/splits/<run_id>
    python -m warranty_analytics_model phase7-contract-check
    python -m warranty_analytics_model phase7-plan-check --mart-dir artifacts/feature_mart/<run_id> --split-dir artifacts/splits/<run_id>
    python -m warranty_analytics_model phase7-build --mart-dir artifacts/feature_mart/<run_id> --split-dir artifacts/splits/<run_id>
    python -m warranty_analytics_model phase7-validate --feature-dir artifacts/structured_features/<run_id>
    python -m warranty_analytics_model phase8-contract-check
    python -m warranty_analytics_model phase8-plan-check --mart-dir artifacts/feature_mart/<run_id> --split-dir artifacts/splits/<run_id> --structured-dir artifacts/structured_features/<run_id>
    python -m warranty_analytics_model phase8-build --mart-dir artifacts/feature_mart/<run_id> --split-dir artifacts/splits/<run_id> --structured-dir artifacts/structured_features/<run_id>
    python -m warranty_analytics_model phase8-validate --text-dir artifacts/text_features/<run_id>
    python -m warranty_analytics_model phase9-contract-check
    python -m warranty_analytics_model phase9-plan-check --mart-dir <phase5> --split-dir <phase6> --structured-dir <phase7> --text-dir <phase8>
    python -m warranty_analytics_model phase9-train --mart-dir <phase5> --split-dir <phase6> --structured-dir <phase7> --text-dir <phase8>
    python -m warranty_analytics_model phase9-validate --model-dir artifacts/baseline_models/<run_id>
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

Phase 6 is fully offline after Phase 5 has completed. Run
`phase5-validate`, `phase6-plan-check`, `phase6-build`, and
`phase6-validate` against the same timestamped mart directory. The split build
does not query SQL Server and does not copy model features into train,
validation, or test files. It stores claim membership, group exposure,
evaluation-cohort metadata, manifests, and a hash-only immutable test lock.
Use a new `--run-id` for a changed Phase 5 mart; completed split runs are not
silently overwritten.

Phase 7 is fully offline after the exact Phase 5 mart and corrected Phase 6
split have completed. Run `phase5-validate` against the selected mart,
`phase6-validate` against the corrected split, then `phase7-plan-check`,
`phase7-build`, and `phase7-validate` using those same inputs. Feature
matrices, manifests, diagnostics, and reports are generated locally under
ignored `artifacts/structured_features/` and
`reports/phase7_structured_features/` directories.

Phase 8 is fully offline after the exact Phase 5 mart, corrected Phase 6 split,
and hardened Phase 7 artifact have completed. Run `phase8-contract-check`,
`phase8-plan-check`, `phase8-build`, and `phase8-validate` with those exact
inputs. Historical documents, lexical features, manifests, diagnostics, and
aggregate-only reports are generated locally under ignored
`artifacts/text_features/` and `reports/phase8_text_features/` directories.

Phase 9 is also offline. Install the `modeling` extra, run the Phase 9 contract
and plan gates against the exact Phase 5–8 directories, then train. Models fit
on TRAIN only and are evaluated on VALIDATION only. Generated `.cbm` files,
validation-only predictions, manifests, and aggregate reports remain ignored;
the TEST target is not read.
