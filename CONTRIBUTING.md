# Contributing

## Change guidance

Keep changes focused and explain the reason for each structural or behavioral
change. Use a short-lived branch or another isolated change workflow that fits
the repository's hosting process; this repository does not impose a corporate
branch naming policy.

Preserve the Phase 0 prediction contract. Any future change affecting prediction
timing, claim-level grain, feature availability, or leakage controls must update
the model contract and obtain the required business and data-owner review before
implementation.

## Environment setup

Follow docs/development_setup.md. Use a local virtual environment and install
the project in editable mode with development dependencies.

## Code style and tests

- Use Python 3.11 or later and type hints on public functions.
- Keep reusable logic under src/warranty_analytics_model/.
- Keep notebooks exploratory; do not make them the only implementation of
  reusable workflows.
- Add meaningful unit tests for isolated behavior and integration tests only
  when components interact.
- Run the quality commands before submitting a change:

      python -m ruff check .
      python -m ruff format --check .
      python -m mypy src
      python -m pytest

- Do not use wildcard imports or broad type-checking suppressions.
- Document non-obvious decisions and update the relevant architecture or phase
  record.

## Secrets and data

- Never commit .env files, credentials, database passwords, or connection
  strings containing secrets.
- Never add VINs, customer records, technician notes, repair notes, or
  production-like warranty records to source control or test fixtures.
- Keep generated data, artifacts, reports, and logs in their configured
  directories; their contents are ignored by default.
- Do not add external telemetry or analytics integrations without explicit
  review.

## Change checklist

- [ ] The change stays within the intended project phase.
- [ ] Phase 0 target, timing, grain, and leakage controls remain intact.
- [ ] New public behavior has type hints and meaningful tests.
- [ ] Documentation is updated when behavior or structure changes.
- [ ] No credentials or record-level data were added.
- [ ] Ruff lint and format checks pass.
- [ ] Mypy passes.
- [ ] Pytest and coverage pass.
