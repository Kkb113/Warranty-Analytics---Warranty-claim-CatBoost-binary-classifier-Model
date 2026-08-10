"""Fictional tests for deterministic date-level boundary selection."""

from __future__ import annotations

import pandas as pd
import pytest

from warranty_analytics_model.splits.assignments import (
    assignment_date_order_errors,
    build_split_assignments,
    validate_assignment_frame,
)
from warranty_analytics_model.splits.boundary import determine_boundaries
from warranty_analytics_model.splits.config import validate_split_settings
from warranty_analytics_model.splits.manifest import assignment_content_sha256
from warranty_analytics_model.splits.models import SplitError, SplitSettings


def _snapshot(
    dates: list[str],
    targets: list[int] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "warranty_claim_key": list(range(1, len(dates) + 1)),
            "claim__claim_date": pd.to_datetime(dates),
            "target__high_cost_claim_flag": targets or [0] * len(dates),
        }
    )


def _settings(**updates: object) -> SplitSettings:
    payload: dict[str, object] = {
        "strategy": "chronological",
        "train_fraction": 0.70,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "preserve_same_date": True,
        "tie_break": "earlier_date",
    }
    payload.update(updates)
    return SplitSettings.model_validate(payload)


def test_default_boundary_is_deterministic_and_date_based() -> None:
    snapshot = _snapshot([f"2025-01-{day:02d}" for day in range(1, 21)])
    first = determine_boundaries(snapshot, _settings())
    second = determine_boundaries(snapshot.sample(frac=1.0, random_state=11), _settings())

    assert first == second
    assert first.train_end_date.isoformat() == "2025-01-14"
    assert first.validation_end_date.isoformat() == "2025-01-17"


def test_boundary_does_not_change_when_only_target_changes() -> None:
    dates = [f"2025-01-{day:02d}" for day in range(1, 21)]
    first = determine_boundaries(_snapshot(dates, [0] * 20), _settings())
    second = determine_boundaries(_snapshot(dates, [1] * 20), _settings())

    assert first == second


def test_tie_break_selects_earlier_date() -> None:
    snapshot = _snapshot([f"2025-01-{day:02d}" for day in range(1, 6)])
    result = determine_boundaries(snapshot, _settings())

    # 70% of five rows is 3.5; cumulative counts 3 and 4 are equally close.
    assert result.train_end_date.isoformat() == "2025-01-03"
    assert result.validation_end_date.isoformat() == "2025-01-04"


def test_same_day_claims_remain_together_and_order_is_strict() -> None:
    snapshot = _snapshot(
        [
            "2025-01-01",
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
        ]
    )
    assignments = build_split_assignments(snapshot, determine_boundaries(snapshot, _settings()))
    assert assignments.groupby("claim_date")["split"].nunique().max() == 1
    assert assignment_date_order_errors(assignments) == []


def test_assignment_hash_is_reproducible_after_reordering() -> None:
    snapshot = _snapshot([f"2025-01-{day:02d}" for day in range(1, 21)])
    assignments = build_split_assignments(snapshot, determine_boundaries(snapshot, _settings()))
    reordered = assignments.sample(frac=1.0, random_state=22).reset_index(drop=True)

    assert assignment_content_sha256(assignments) == assignment_content_sha256(reordered)


def test_invalid_dates_and_duplicate_keys_block() -> None:
    invalid = pd.DataFrame(
        {
            "warranty_claim_key": [1, 1, 2],
            "claim__claim_date": ["2025-01-01", None, "2025-01-03"],
        }
    )
    with pytest.raises(SplitError):
        determine_boundaries(invalid, _settings())


def test_invalid_split_settings_fail_closed() -> None:
    invalid = _settings(test_fraction=0.20)
    errors = validate_split_settings(invalid)

    assert any("sum to 1.0" in error for error in errors)


def test_assignment_integrity_guards_reject_bad_metadata() -> None:
    valid = pd.DataFrame(
        {
            "warranty_claim_key": [1, 2, 3],
            "claim_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "split": ["TRAIN", "VALIDATION", "TEST"],
        }
    )
    with pytest.raises(SplitError, match="target"):
        validate_assignment_frame(valid.assign(target__high_cost_claim_flag=[0, 1, 0]))
    with pytest.raises(SplitError, match="duplicate"):
        validate_assignment_frame(valid.assign(warranty_claim_key=[1, 1, 3]))
    with pytest.raises(SplitError, match="invalid split"):
        validate_assignment_frame(valid.assign(split=["TRAIN", "OTHER", "TEST"]))
    with pytest.raises(SplitError, match="expected"):
        validate_assignment_frame(valid, expected_claim_count=4)
    with pytest.raises(SplitError, match="invalid claim dates"):
        validate_assignment_frame(valid.assign(claim_date=["bad", "2025-01-02", "2025-01-03"]))


def test_date_order_diagnostics_report_same_date_and_partition_order_errors() -> None:
    assignments = pd.DataFrame(
        {
            "warranty_claim_key": [1, 2, 3, 4, 5, 6],
            "claim_date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-05",
                ]
            ),
            "split": ["TRAIN", "TRAIN", "VALIDATION", "VALIDATION", "TEST", "TEST"],
        }
    )
    errors = assignment_date_order_errors(assignments)

    assert any("claim date" in error for error in errors)
