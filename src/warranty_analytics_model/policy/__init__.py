"""Machine-enforced Phase 4 target, availability, and leakage policies."""

from .coverage import (
    build_future_allowlists,
    validate_feature_policy_coverage,
)
from .live import run_live_phase4
from .loader import (
    load_feature_policy,
    load_leakage_policy,
    load_phase4_contracts,
    load_target_contract,
    policy_checksum,
)
from .models import (
    FeaturePolicyContract,
    FeaturePolicyEntry,
    FeaturePolicyName,
    LeakagePolicyContract,
    Phase4ContractBundle,
    Phase4ContractError,
    TargetContract,
)
from .reporting import write_phase4_reports
from .target_contract import validate_claim_eligibility
from .validator import (
    assert_phase4_contracts_valid,
    validate_historical_source_rules,
    validate_phase4_contracts,
)

__all__ = [
    "FeaturePolicyContract",
    "FeaturePolicyEntry",
    "FeaturePolicyName",
    "LeakagePolicyContract",
    "Phase4ContractBundle",
    "Phase4ContractError",
    "TargetContract",
    "build_future_allowlists",
    "load_feature_policy",
    "load_leakage_policy",
    "load_phase4_contracts",
    "load_target_contract",
    "policy_checksum",
    "run_live_phase4",
    "validate_claim_eligibility",
    "validate_feature_policy_coverage",
    "validate_historical_source_rules",
    "assert_phase4_contracts_valid",
    "validate_phase4_contracts",
    "write_phase4_reports",
]
