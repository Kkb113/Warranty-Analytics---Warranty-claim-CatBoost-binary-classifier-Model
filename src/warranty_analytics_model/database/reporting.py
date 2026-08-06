"""Explicit JSON and Markdown reporting for schema validation results."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .models import SchemaIssue, ValidationResult


def _display_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _markdown(result: ValidationResult) -> str:
    lines = [
        "# Schema validation report",
        "",
        f"- Status: **{result.status.upper()}**",
        f"- Validation ID: `{result.validation_id}`",
        f"- Contract: `{result.contract_version}`",
        f"- Contract checksum: `{result.contract_checksum}`",
        f"- Executed: `{result.execution_timestamp.isoformat()}`",
        f"- Environment: `{result.environment}`",
        f"- Database: `{result.database}`",
        f"- Server: `{result.server or '-'}`",
        f"- Duration seconds: `{result.duration_seconds:.3f}`",
        "",
        "## Totals",
        "",
        "| Metric | Contract | Actual matched |",
        "|---|---:|---:|",
        f"| Tables | {result.included_table_count} | {result.actual_table_count} |",
        f"| Columns | {result.included_column_count} | {result.actual_column_count} |",
        f"| Foreign keys | {result.included_foreign_key_count} | {result.actual_foreign_key_count} |",
        f"| Errors | {result.error_count} | - |",
        f"| Warnings | {result.warning_count} | - |",
        f"| Info | {result.info_count} | - |",
        "",
        "## Excluded objects",
        "",
    ]
    if result.excluded_objects:
        lines.extend(f"- `{name}` (name-only status)" for name in result.excluded_objects)
    else:
        lines.append("- None detected by name")
    lines.extend(["", "## Findings", ""])
    if not result.issues:
        lines.append("No findings.")
    else:
        lines.extend(
            [
                "| Severity | Code | Object | Expected | Actual | Message |",
                "|---|---|---|---|---|---|",
            ]
        )
        for issue in result.issues:
            lines.append(_markdown_issue(issue))
    lines.extend(
        [
            "",
            "## Final status",
            "",
            f"Schema validation **{result.status.upper()}** with {result.error_count} blocking error(s).",
        ]
    )
    return "\n".join(lines) + "\n"


def _markdown_issue(issue: SchemaIssue) -> str:
    object_name = (
        ".".join(
            part for part in (issue.schema, issue.table, issue.column or issue.constraint) if part
        )
        or issue.object_type
    )
    message = issue.message.replace("|", "\\|")
    return (
        f"| {issue.severity} | `{issue.code}` | `{object_name}` | "
        f"`{_display_value(issue.expected)}` | `{_display_value(issue.actual)}` | {message} |"
    )


def write_validation_reports(
    result: ValidationResult,
    output_dir: Path,
    formats: Iterable[str] = ("json", "markdown"),
) -> list[Path]:
    """Write reports only when explicitly called by the CLI or caller."""

    requested = tuple(dict.fromkeys(format.casefold() for format in formats))
    invalid = set(requested) - {"json", "markdown"}
    if invalid:
        raise ValueError(f"Unsupported report format(s): {', '.join(sorted(invalid))}")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = result.execution_timestamp.strftime("%Y%m%dT%H%M%SZ")
    stem = f"schema_validation_{timestamp}_{result.validation_id[:12]}"
    written: list[Path] = []
    if "json" in requested:
        json_path = output_dir / f"{stem}.json"
        json_path.write_text(
            json.dumps(result.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        written.append(json_path)
    if "markdown" in requested:
        markdown_path = output_dir / f"{stem}.md"
        markdown_path.write_text(_markdown(result), encoding="utf-8", newline="\n")
        written.append(markdown_path)
    return written
