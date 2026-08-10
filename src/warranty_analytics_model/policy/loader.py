"""Load and checksum the version-controlled Phase 4 contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from ..paths import discover_repository_root
from .models import (
    FeaturePolicyContract,
    LeakagePolicyContract,
    Phase4ContractBundle,
    Phase4ContractError,
    TargetContract,
)

_ModelT = TypeVar("_ModelT", bound=BaseModel)

TARGET_CONTRACT_NAME = "high_cost_target_v1.yaml"
FEATURE_POLICY_NAME = "claim_time_feature_policy_v1.yaml"
LEAKAGE_POLICY_NAME = "leakage_policy_v1.yaml"


def policy_checksum(path: Path) -> str:
    """Return the SHA-256 checksum of the exact policy file bytes."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise Phase4ContractError(f"Could not read Phase 4 contract: {path}") from exc


def _contract_path(project_root: Path | None, name: str) -> Path:
    root = discover_repository_root(project_root)
    return root / "contracts" / name


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Phase4ContractError(f"Phase 4 contract is missing: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise Phase4ContractError(f"Could not read Phase 4 contract: {path}") from exc
    if not isinstance(payload, dict):
        raise Phase4ContractError(f"Phase 4 contract must be a top-level mapping: {path}")
    return payload


def _validate_model(path: Path, model_type: type[_ModelT]) -> _ModelT:
    payload = _load_payload(path)
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in item.get('loc', ())) or 'contract'}: "
            f"{item.get('msg', 'invalid value')}"
            for item in exc.errors()
        )
        raise Phase4ContractError(f"Invalid Phase 4 contract {path.name}: {details}") from exc


def load_target_contract(
    project_root: Path | None = None,
    path: Path | None = None,
) -> tuple[TargetContract, str]:
    """Load the target contract and return it with its exact checksum."""

    contract_path = path or _contract_path(project_root, TARGET_CONTRACT_NAME)
    return _validate_model(contract_path, TargetContract), policy_checksum(contract_path)


def load_feature_policy(
    project_root: Path | None = None,
    path: Path | None = None,
) -> tuple[FeaturePolicyContract, str]:
    """Load the claim-time field policy and return it with its exact checksum."""

    contract_path = path or _contract_path(project_root, FEATURE_POLICY_NAME)
    return _validate_model(contract_path, FeaturePolicyContract), policy_checksum(contract_path)


def load_leakage_policy(
    project_root: Path | None = None,
    path: Path | None = None,
) -> tuple[LeakagePolicyContract, str]:
    """Load the hard leakage contract and return it with its exact checksum."""

    contract_path = path or _contract_path(project_root, LEAKAGE_POLICY_NAME)
    return _validate_model(contract_path, LeakagePolicyContract), policy_checksum(contract_path)


def load_phase4_contracts(project_root: Path | None = None) -> Phase4ContractBundle:
    """Load all Phase 4 contracts without opening a database connection."""

    target, target_checksum = load_target_contract(project_root)
    feature_policy, feature_checksum = load_feature_policy(project_root)
    leakage, leakage_checksum = load_leakage_policy(project_root)
    return Phase4ContractBundle(
        target=target,
        feature_policy=feature_policy,
        leakage=leakage,
        target_checksum=target_checksum,
        feature_policy_checksum=feature_checksum,
        leakage_checksum=leakage_checksum,
    )
