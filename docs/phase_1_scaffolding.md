# Phase 1 scaffolding record

## Phase objective

Create an installable, configuration-driven, testable, lintable, type-checked,
reproducible Python foundation without connecting to SQL Server, processing
warranty records, engineering features, training a model, or implementing
inference.

## Repository state before implementation

The repository already contained a Git directory, the Phase 0 documents, and
warranty_analytics_schema_document.docx. It had no Python source, package
metadata, virtual-environment instructions, tests, CI, README, .gitignore,
pre-commit configuration, or existing linting/formatting/type-checking
configuration. The Git remote is GitHub and no CI workflow was present.

The Phase 0 documents were read and preserved as the source of truth. The schema
DOCX was inspected for its documented table inventory and columns; no warranty
record data was read or processed.

## Files created

- pyproject.toml
- configs/base.yaml
- configs/development.yaml
- configs/test.yaml
- .env.example
- .gitignore
- .pre-commit-config.yaml
- .github/workflows/quality.yml
- README.md
- CONTRIBUTING.md
- docs/architecture.md
- docs/development_setup.md
- docs/phase_1_scaffolding.md
- data/README.md
- artifacts/README.md
- reports/README.md
- logs/README.md
- notebooks/README.md
- sql/README.md
- sql/source_validation/README.md
- sql/target_queries/README.md
- sql/feature_queries/README.md
- src/warranty_analytics_model/ and its phase-boundary packages
- tests/ and its unit/integration documentation and tests

## Files preserved

- docs/model_contract.md
- docs/phase_0_open_questions.md
- warranty_analytics_schema_document.docx

No unrelated existing file was overwritten.

## Technical decisions

- Python version: Python 3.11 or later.
- Package and build backend: pip for installation with the standard setuptools
  build backend because no existing package manager or lock policy was present.
- Package layout: src/warranty_analytics_model/.
- Runtime configuration: pydantic-settings for typed validation, PyYAML for
  non-secret files, and python-dotenv for optional local .env loading.
- Configuration precedence: typed defaults, base YAML, environment YAML, local
  .env, then operating-system environment variables.
- Logging: standard-library logging with reusable marked handlers, console output
  by default, and opt-in file logging.
- Paths: pathlib with repository-relative defaults and explicit directory
  creation only.
- Reproducibility: Python random seed utility only; NumPy and future ML
  frameworks are intentionally not dependencies.
- CLI: standard-library argparse with doctor, show-config, and version only.
- Tests: pytest and pytest-cov with an 80 percent minimum coverage gate.
- Quality tools: Ruff linting and formatting, mypy, and pre-commit.
- CI: a single GitHub Actions quality workflow was added because the repository
  has a GitHub remote and no existing CI configuration.

## Deviations from the requested structure

- README files were added to generated-output directories and SQL subdirectories
  so Git can retain the directories while generated contents remain ignored.
- No SQL files were added because the prompt explicitly defers SQL implementation
  and prohibits placeholder queries.
- No SQL Server driver or machine-learning package was added.
- The schema DOCX remains at its existing repository-root location and was not
  moved or rewritten.

## Intentionally deferred

- SQL Server data access, schema validation, source-table checks, and database
  connectivity: Phase 2.
- Data profiling, synthetic-data audit, and logical data-quality analysis:
  Phase 3.
- Executable target construction and leakage enforcement against actual data:
  Phase 4.
- Feature-mart construction, feature engineering, training, tuning, evaluation,
  calibration, inference, API work, and monitoring: Phase 5 and later.
- All unresolved Phase 0 business questions remain unresolved.

## Quality gates and definition of done

Definition-of-done status: Complete with a documented runtime note.

Phase 1 is complete after the editable installation, Ruff checks, formatting
check, mypy, pytest with coverage, CLI help, doctor, and version commands pass.
The package imports successfully, configuration redaction is tested, no
developer-specific absolute paths are embedded, and generated data/output
contents are ignored by default.

Validation results:

| Check | Result |
| --- | --- |
| Editable installation with the bundled Python 3.12 runtime | Passed |
| Ruff lint | Passed |
| Ruff formatting check | Passed |
| Mypy | Passed |
| Pytest | Passed: 29 tests |
| Coverage | Passed: 87.37 percent, minimum 80 percent |
| CLI help | Passed |
| Doctor command | Passed |
| Version command | Passed: 0.1.0 |
| Pre-commit hooks on all created source, test, and configuration files | Passed |

The host default Python executable is 3.10.10; validation used the available
bundled Python 3.12.13 runtime, which satisfies the project requirement of
Python 3.11 or later. Because the repository has no commits yet, the
pre-commit all-files mode had no tracked paths; explicit file validation was
run and passed.

No database connection was implemented, no warranty records were processed, no
feature engineering was implemented, no model was trained, and no inference
pipeline or API was implemented.
