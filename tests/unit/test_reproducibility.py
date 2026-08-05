"""Tests for deterministic random seeding."""

from __future__ import annotations

import random

from warranty_analytics_model.reproducibility import set_random_seed


def test_python_random_output_is_repeatable() -> None:
    """The same seed produces the same sequence from Python random."""

    set_random_seed(42)
    first = [random.random() for _ in range(5)]
    set_random_seed(42)
    second = [random.random() for _ in range(5)]

    assert first == second
