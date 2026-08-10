"""Opt-in live Phase 4 validation coverage."""

from __future__ import annotations

import os

import pytest


@pytest.mark.database
def test_live_phase4_is_explicitly_operator_run() -> None:
    """Normal CI does not connect to SQL Server or build a training dataset."""

    if os.environ.get("WARRANTY_RUN_DB_TESTS", "false").casefold() != "true":
        pytest.skip("Set WARRANTY_RUN_DB_TESTS=true for an explicit local live run.")
    pytest.skip("Live Phase 4 validation is executed with the phase4-validate CLI command.")
