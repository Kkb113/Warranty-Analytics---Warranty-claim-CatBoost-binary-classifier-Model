"""Typed, secret-safe Phase 3 findings."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FindingSeverity = Literal["ERROR", "WARNING", "INFO"]


class Finding(BaseModel):
    """One aggregate data-quality or modeling-risk finding."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str
    issue_code: str
    severity: FindingSeverity
    category: str
    table: str | None = None
    columns: list[str] = Field(default_factory=list)
    affected_rows: int = Field(default=0, ge=0)
    affected_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    description: str
    evidence: dict[str, object] = Field(default_factory=dict)
    modeling_impact: str
    recommendation: str
    required_resolution_phase: str
    blocking_for_next_phase: bool = False
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def make_finding(
    issue_code: str,
    severity: FindingSeverity,
    category: str,
    description: str,
    *,
    table: str | None = None,
    columns: Sequence[str] = (),
    affected_rows: int = 0,
    affected_percentage: float = 0.0,
    evidence: dict[str, object] | None = None,
    modeling_impact: str = "Requires diagnostic review before feature design.",
    recommendation: str = "Review with the data owner before Phase 4.",
    required_resolution_phase: str = "Phase 4",
    blocking_for_next_phase: bool = False,
) -> Finding:
    """Construct a stable finding ID without embedding row-level data."""

    safe_code = "_".join(part for part in issue_code.lower().split() if part)
    table_part = (table or "project").replace(".", "_").replace(" ", "_")
    finding_id = f"{safe_code}:{table_part}"
    return Finding(
        finding_id=finding_id,
        issue_code=issue_code,
        severity=severity,
        category=category,
        table=table,
        columns=list(columns),
        affected_rows=max(0, int(affected_rows)),
        affected_percentage=min(100.0, max(0.0, float(affected_percentage))),
        description=description,
        evidence=evidence or {},
        modeling_impact=modeling_impact,
        recommendation=recommendation,
        required_resolution_phase=required_resolution_phase,
        blocking_for_next_phase=blocking_for_next_phase,
    )


def finding_counts(findings: Sequence[Finding]) -> dict[str, int]:
    """Return stable severity counts for CLI and reports."""

    return {
        severity: sum(1 for finding in findings if finding.severity == severity)
        for severity in ("ERROR", "WARNING", "INFO")
    }


def overall_status(findings: Sequence[Finding]) -> str:
    """Map findings to the Phase 3 readiness vocabulary."""

    if any(finding.severity == "ERROR" for finding in findings):
        return "BLOCKED"
    if any(finding.severity == "WARNING" for finding in findings):
        return "READY WITH WARNINGS"
    return "READY"
