"""Deterministic seed utilities for the libraries used in Phase 1."""

from __future__ import annotations

import random


def set_random_seed(seed: int = 42) -> None:
    """Seed Python's random module.

    Full model reproducibility may require framework-specific settings in later
    phases. NumPy and future ML frameworks are intentionally not imported here.
    """

    random.seed(seed)
