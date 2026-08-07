"""JSON and Markdown report generation for Phase 3."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .findings import Finding, finding_counts

REQUIRED_REPORTS = (
    "phase_3_summary.md",
    "phase_3_summary.json",
    "table_profiles.json",
    "target_profile.json",
    "data_quality_findings.json",
    "synthetic_data_audit.json",
    "leakage_diagnostics.json",
)


def _json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _status_summary(result: dict[str, Any]) -> list[str]:
    target = result.get("target_profile", {})
    if not isinstance(target, dict):
        target = {}
    findings = result.get("findings", [])
    counts = (
        finding_counts([Finding.model_validate(item) for item in findings])
        if isinstance(findings, list)
        else {"ERROR": 0, "WARNING": 0, "INFO": 0}
    )
    claims = target.get("claims", 0)
    positive = target.get("positive_percentage", 0.0)
    status = result.get("status", "UNKNOWN")
    target_audit = target.get("target_generation_audit", {})
    deterministic = bool(
        isinstance(target_audit, dict) and target_audit.get("total_claim_cost_deterministic")
    )
    leakage = result.get("leakage_diagnostics", {})
    leakage_fields = len(leakage.get("fields", [])) if isinstance(leakage, dict) else 0
    duplicate = result.get("synthetic_data_audit", {})
    duplicate_flag = bool(
        isinstance(duplicate, dict)
        and isinstance(duplicate.get("duplicate_audit"), dict)
        and duplicate["duplicate_audit"].get("duplicates_found")
    )
    return [
        "# Phase 3 summary",
        "",
        "## Executive summary",
        "",
        f"- Status: **{status}**",
        f"- Claims available: **{claims}**",
        f"- High-cost percentage: **{positive}%**",
        f"- Target balance: **{target.get('class_balance', 'unknown')}**",
        f"- Apparent total-cost threshold: **{'suspected' if deterministic else 'not established'}**",
        f"- Suspected post-outcome leakage fields audited: **{leakage_fields}**",
        f"- Duplicate/scenario contamination detected: **{'yes' if duplicate_flag else 'no'}**",
        f"- Data-quality findings: **{counts['ERROR']} errors, {counts['WARNING']} warnings, {counts['INFO']} info**",
        "",
        "The target and outcome fields are diagnostic evidence only. They must not be used as prediction-time features.",
        "",
        "## Required questions",
        "",
        _question_answer("1. How many claims are available?", str(claims)),
        _question_answer("2. What percentage are high-cost?", f"{positive}%"),
        _question_answer(
            "3. Is the target balanced or imbalanced?", str(target.get("class_balance", "unknown"))
        ),
        _question_answer(
            "4. Is the target apparently derived from a cost threshold?",
            "Yes—empirical suspicion."
            if deterministic
            else "No exact single threshold established.",
        ),
        _question_answer(
            "5. Are there suspected leakage fields?",
            "Yes—post-outcome fields were quantified."
            if leakage_fields
            else "No relationship was measurable.",
        ),
        _question_answer(
            "6. Are there identifier leakage patterns?",
            _flag_answer(result, "SYNTHETIC_IDENTIFIER_LEAKAGE"),
        ),
        _question_answer(
            "7. Are there duplicate synthetic scenarios?",
            "Yes—review duplicate audit."
            if duplicate_flag
            else "No repeated scenario family detected.",
        ),
        _question_answer("8. Are there major temporal inconsistencies?", _temporal_answer(result)),
        _question_answer("9. Are there major missing-data issues?", _missingness_answer(result)),
        _question_answer(
            "10. Are there major sparse categories?",
            "Review category sparsity in table profiles and group purity.",
        ),
        _question_answer("11. Is the data suitable to proceed to Phase 4?", str(status)),
        _question_answer(
            "12. What must be resolved first?",
            "Confirm target generation, as-of availability, leakage exclusions, and split groups.",
        ),
        "",
        "## Synthetic-data audit",
        "",
        "| Test | Result | Severity | Evidence | Modeling impact | Recommendation |",
        "|---|---|---|---|---|---|",
    ]


def _question_answer(question: str, answer: str) -> str:
    return f"- **{question}** {answer}"


def _flag_answer(result: dict[str, Any], flag: str) -> str:
    audit = result.get("synthetic_data_audit", {})
    identifier = audit.get("identifier_audit", {}) if isinstance(audit, dict) else {}
    flags = identifier.get("flags", []) if isinstance(identifier, dict) else []
    return (
        "Yes—synthetic identifier leakage was flagged."
        if flag in flags
        else "No supported pure identifier group was flagged."
    )


def _temporal_answer(result: dict[str, Any]) -> str:
    rows = result.get("temporal_violations", [])
    return (
        "Review temporal findings."
        if any(isinstance(row, dict) and row.get("violation_count", 0) for row in rows)
        else "No violation was observed in the supplied records."
    )


def _missingness_answer(result: dict[str, Any]) -> str:
    rows = result.get("missingness", [])
    high = [
        row for row in rows if isinstance(row, dict) and float(row.get("null_percentage", 0)) >= 50
    ]
    return (
        f"Yes—{len(high)} fields have at least 50% missingness."
        if high
        else "No field has at least 50% missingness."
    )


def _synthetic_rows(result: dict[str, Any]) -> list[str]:
    audit = result.get("synthetic_data_audit", {})
    if not isinstance(audit, dict):
        return []
    rows: list[str] = []
    target = audit.get("target_generation", {})
    deterministic = isinstance(target, dict) and target.get("total_claim_cost_deterministic")
    rows.append(
        _audit_row(
            "Target-generation determinism",
            "suspected" if deterministic else "not established",
            "WARNING" if deterministic else "INFO",
            "Aggregate threshold separation test",
            "Outcome-derived target would leak if cost fields are available",
            "Exclude post-outcome costs and confirm the label definition.",
        )
    )
    identifiers = audit.get("identifier_audit", {})
    flag = isinstance(identifiers, dict) and "SYNTHETIC_IDENTIFIER_LEAKAGE" in identifiers.get(
        "flags", []
    )
    rows.append(
        _audit_row(
            "Identifier leakage",
            "flagged" if flag else "not flagged",
            "WARNING" if flag else "INFO",
            "Hashed prefix/suffix and supported purity groups",
            "Identifiers must not become production features",
            "Exclude identifiers and evaluate unseen groups.",
        )
    )
    purity = audit.get("group_purity", [])
    pure = isinstance(purity, list) and any(
        item.get("target_pure") for item in purity if isinstance(item, dict)
    )
    rows.append(
        _audit_row(
            "Category purity",
            "supported pure groups found" if pure else "no supported pure group found",
            "WARNING" if pure else "INFO",
            "Group sizes and target rates",
            "Potential memorization across splits",
            "Use group-aware validation for meaningful groups.",
        )
    )
    duplicate = audit.get("duplicate_audit", {})
    duplicate_flag = isinstance(duplicate, dict) and duplicate.get("duplicates_found")
    rows.append(
        _audit_row(
            "Duplicate scenario templates",
            "found" if duplicate_flag else "not found",
            "WARNING" if duplicate_flag else "INFO",
            "Deterministic fingerprints exclude target",
            "Random splits may overstate generalization",
            "Use fingerprint-aware Phase 6 splits.",
        )
    )
    text = audit.get("text_audit", {})
    text_flag = isinstance(text, dict) and "SYNTHETIC_TEXT_TEMPLATE_LEAKAGE" in text.get(
        "flags", []
    )
    rows.append(
        _audit_row(
            "Repeated text templates",
            "flagged" if text_flag else "not flagged",
            "WARNING" if text_flag else "INFO",
            "Normalized duplicate text hashes",
            "Text may reveal outcome labels",
            "Exclude or sanitize outcome-bearing text.",
        )
    )
    rows.append(
        _audit_row(
            "Missingness leakage",
            "see missingness-by-target diagnostics",
            "WARNING",
            "Target-stratified null rates",
            "Missingness may encode generation process",
            "Confirm process timing before adding missing indicators.",
        )
    )
    rows.append(
        _audit_row(
            "Date-pattern leakage",
            "see temporal audit",
            "INFO",
            "Date ranges and rule checks",
            "Date fragments may encode generation order",
            "Use claim-time availability rules.",
        )
    )
    rows.append(
        _audit_row(
            "Batch/lot leakage",
            "see group-purity audit",
            "WARNING" if pure else "INFO",
            "Hashed group support and rates",
            "Group memorization can contaminate splits",
            "Hold out meaningful batches/lots.",
        )
    )
    rows.append(
        _audit_row(
            "Supplier leakage",
            "see group-purity audit",
            "INFO",
            "Supplier group support and rates",
            "Association is not causation",
            "Review governance and unseen-supplier coverage.",
        )
    )
    rows.append(
        _audit_row(
            "Cost arithmetic determinism",
            "see repair arithmetic diagnostics",
            "INFO",
            "Line-cost comparison",
            "Synthetic arithmetic may produce shortcuts",
            "Do not assume a business accounting formula.",
        )
    )
    rows.append(
        _audit_row(
            "Outcome fields generated directly from target",
            "post-outcome fields present",
            "WARNING",
            "Leakage evidence table",
            "Unavailable at scoring time",
            "Exclude until availability is proven.",
        )
    )
    return rows


def _audit_row(
    test: str, result: str, severity: str, evidence: str, impact: str, recommendation: str
) -> str:
    return f"| {test} | {result} | {severity} | {evidence} | {impact} | {recommendation} |"


def markdown_summary(result: dict[str, Any]) -> str:
    """Render the required concise executive report."""

    lines = _status_summary(result)
    lines.extend(_synthetic_rows(result))
    lines.extend(
        [
            "",
            "## Phase 4 recommendations",
            "",
            "- Confirm the target-generation rule and document whether the dataset is synthetic proof-of-concept only.",
            "- Define the prediction-time as-of snapshot and exclude post-outcome costs, status, repair, and finalized diagnosis fields.",
            "- Use group/fingerprint-aware temporal evaluation for repeated trucks, batches, lots, suppliers, and scenarios.",
            "- Resolve ERROR findings and obtain business/data-owner decisions for the Phase 0 open questions before feature design.",
            "",
            "## Scope confirmations",
            "",
            "No database writes, excluded `ml_*` table reads, production feature engineering, train/test split, or predictive model training were performed.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase3_reports(
    result: dict[str, Any],
    output_root: Path,
    *,
    formats: Iterable[str] = ("json", "markdown"),
    run_timestamp: datetime | None = None,
) -> Path:
    """Write required report files below a timestamped ignored directory."""

    requested = {item.casefold() for item in formats}
    if "both" in requested:
        requested.update({"json", "markdown"})
    invalid = requested - {"json", "markdown"}
    if invalid:
        raise ValueError(f"Unsupported Phase 3 report format(s): {', '.join(sorted(invalid))}")
    timestamp = run_timestamp or datetime.now(UTC)
    run_dir = output_root / timestamp.strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    if "json" in requested:
        _write_json(run_dir / "phase_3_summary.json", result)
        _write_json(run_dir / "table_profiles.json", result.get("table_profiles", []))
        _write_json(run_dir / "target_profile.json", result.get("target_profile", {}))
        _write_json(run_dir / "data_quality_findings.json", result.get("findings", []))
        _write_json(run_dir / "synthetic_data_audit.json", result.get("synthetic_data_audit", {}))
        _write_json(run_dir / "leakage_diagnostics.json", result.get("leakage_diagnostics", {}))
    if "markdown" in requested:
        (run_dir / "phase_3_summary.md").write_text(
            markdown_summary(result), encoding="utf-8", newline="\n"
        )
    return run_dir
