"""Package import smoke tests for optional analytical dependencies."""

from __future__ import annotations


def test_profiling_package_import_exposes_phase3_api() -> None:
    """The installed package can import its Phase 3 API in offline CI."""

    import warranty_analytics_model.profiling as profiling

    assert callable(profiling.profile_dataframes)
    assert callable(profiling.run_live_phase3)
