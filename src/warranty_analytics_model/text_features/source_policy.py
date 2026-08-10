"""Fail-closed Phase 8 text-source allowlist."""

from __future__ import annotations

from typing import Any

ALLOWED_PHASE8_TEXT_VALUE_SOURCES = frozenset({"prior_failure__failure_description"})
APPROVED_PHASE8_TEXT_ARTIFACT = "prior_claim_history"
PROHIBITED_PHASE8_TEXT_SOURCES = frozenset(
    {
        "complaint_description",
        "diagnostic_summary",
        "technician_notes",
        "technician_note",
        "repair_notes",
        "current_failure_description",
        "failure_description",
        "total_claim_cost",
        "warranty_claim_key",
        "supplier_key",
    }
)


def validate_text_source(source: str, *, source_artifact: str | None = None) -> None:
    """Reject every Phase 8 text value source outside the explicit allowlist."""

    if source not in ALLOWED_PHASE8_TEXT_VALUE_SOURCES:
        raise ValueError(f"Phase 8 text source is not approved: {source}")
    if source_artifact != APPROVED_PHASE8_TEXT_ARTIFACT:
        raise ValueError(f"Phase 8 text artifact is not approved for {source}: {source_artifact}")


def validate_text_lineage_sources(lineage: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Validate every model-feature value source and document artifact."""

    errors: list[str] = []
    for feature_name, item in lineage.items():
        if item.get("is_model_feature") is not True:
            continue
        values = item.get("value_sources", [])
        controls = item.get("control_sources", [])
        if not isinstance(values, list) or not isinstance(controls, list):
            errors.append(f"Feature {feature_name} has invalid Phase 8 source metadata.")
            continue
        if set(str(value) for value in values) & set(str(value) for value in controls):
            errors.append(f"Feature {feature_name} assigns a source as both value and control.")
        try:
            validate_text_source(
                str(item.get("value_sources", [""])[0]),
                source_artifact=str(item.get("source_artifacts", [""])[0]),
            )
        except (IndexError, TypeError, ValueError) as exc:
            errors.append(f"Feature {feature_name}: {exc}")
        if any(str(value) in PROHIBITED_PHASE8_TEXT_SOURCES for value in values):
            errors.append(f"Feature {feature_name} contains a prohibited text source.")
        if item.get("target_dependent") is not False:
            errors.append(f"Feature {feature_name} is target-dependent.")
        if item.get("fitted_transformation") is not None:
            errors.append(f"Feature {feature_name} has a fitted transformation.")
    return {"valid": not errors, "errors": list(dict.fromkeys(errors))}
