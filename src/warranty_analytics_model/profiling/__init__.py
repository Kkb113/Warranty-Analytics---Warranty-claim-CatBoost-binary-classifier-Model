"""Phase 3 data profiling and synthetic-data audit utilities.

The package is intentionally diagnostic.  It produces aggregate profiles and
quality findings; it does not construct production features or train models.
"""

from .config import ProfilingSettings, load_profiling_settings
from .findings import Finding, FindingSeverity
from .runner import profile_dataframes, run_live_phase3

__all__ = [
    "Finding",
    "FindingSeverity",
    "ProfilingSettings",
    "load_profiling_settings",
    "profile_dataframes",
    "run_live_phase3",
]
