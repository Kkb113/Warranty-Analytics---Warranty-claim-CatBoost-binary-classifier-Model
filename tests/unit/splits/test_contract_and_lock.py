"""Offline contract and test-lock tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from warranty_analytics_model.splits.manifest import (
    assignment_content_sha256,
    build_test_lock,
    claim_key_sha256,
    mart_input_fingerprint,
    unordered_claim_key_sha256,
)
from warranty_analytics_model.splits.split_contract import validate_current_split_contract

ROOT = Path(__file__).resolve().parents[3]


def _test_assignments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "warranty_claim_key": [3, 1, 2],
            "claim_date": pd.to_datetime(["2025-01-03", "2025-01-01", "2025-01-02"]),
            "split": ["TEST", "TEST", "TEST"],
        }
    )


def test_phase6_contract_loads_and_validates_against_phase4_phase5() -> None:
    result = validate_current_split_contract(ROOT)

    assert result.valid
    assert result.requested_fractions == {"TRAIN": 0.7, "VALIDATION": 0.15, "TEST": 0.15}


def test_test_lock_hashes_are_order_independent_where_documented() -> None:
    assignments = _test_assignments()
    reordered = assignments.iloc[[2, 0, 1]].reset_index(drop=True)
    lock = build_test_lock(
        split_contract_version="1.0.0",
        split_contract_checksum="split-contract",
        input_mart_checksum="mart-contract",
        input_mart_fingerprint=mart_input_fingerprint(
            mart_contract_checksum="mart-contract",
            claim_snapshot_content_sha256="snapshot",
            group_membership_content_sha256="groups",
        ),
        claim_snapshot_content_sha256="snapshot",
        test_assignments=assignments,
        test_assignment_content_sha256=assignment_content_sha256(assignments),
        test_start_date="2025-01-01",
        test_end_date="2025-01-03",
    )

    assert lock["ordered_test_claim_keys_sha256"] == claim_key_sha256(reordered)
    assert lock["unordered_test_claim_keys_sha256"] == unordered_claim_key_sha256(reordered)
    assert lock["locked"] is True
    assert lock["allowed_first_target_evaluation_phase"] == 15
    assert "warranty_claim_key" not in str(lock)
