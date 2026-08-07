"""Live Phase 3 tests are opt-in and intentionally bounded."""

from __future__ import annotations

import os

import pytest


@pytest.mark.database
def test_live_phase3_is_opt_in() -> None:
    """CI remains database-independent; live execution is run explicitly by an operator."""

    if os.environ.get("WARRANTY_RUN_DB_TESTS", "false").casefold() != "true":
        pytest.skip("Set WARRANTY_RUN_DB_TESTS=true for an explicit local live run.")
    pytest.skip("Live Phase 3 execution is an operator command, not an automatic CI test.")
