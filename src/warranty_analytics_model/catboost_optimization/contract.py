"""Offline Phase 10 contract validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ..paths import discover_repository_root
from .config import SEARCH_PARAMETER_NAMES, TRACKS, load_optimization_settings


def load_optimization_contract(
    project_root: Path | None = None,
) -> tuple[dict[str, Any], str]:
    root = discover_repository_root(project_root)
    path = root / "contracts" / "catboost_optimization.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Phase 10 contract must be a YAML object.")
    checksum = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return payload, checksum


def validate_optimization_contract(project_root: Path | None = None) -> dict[str, Any]:
    """Validate the committed Phase 10 policy and configuration without data access."""

    errors: list[str] = []
    warnings: list[str] = []
    try:
        contract, checksum = load_optimization_contract(project_root)
        settings = load_optimization_settings(project_root)
        policy = contract.get("phase10")
        if not isinstance(policy, dict):
            errors.append("Phase 10 contract must contain phase10 mapping.")
        else:
            if tuple(policy.get("tracks", [])) != TRACKS:
                errors.append("Phase 10 contract tracks are not exactly T1 and T3.")
            if tuple(policy.get("search_parameters", [])) != SEARCH_PARAMETER_NAMES:
                errors.append("Phase 10 contract search parameter allowlist changed.")
            if policy.get("test_target_access") != "FORBIDDEN_UNTIL_PHASE_15":
                errors.append("Phase 10 TEST seal policy is missing or changed.")
        if settings.tracks != TRACKS:
            errors.append("Phase 10 configuration tracks are not exactly T1 and T3.")
    except Exception as exc:
        checksum = None
        errors.append(str(exc))
    return {
        "status": "BLOCKED" if errors else ("PASS WITH WARNINGS" if warnings else "PASS"),
        "valid": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "contract_checksum": checksum,
    }
