"""Pure Markdown reporting helpers for LLM-judged answer metrics."""

from __future__ import annotations

import html
import json
import statistics
from collections.abc import Sequence
from typing import Protocol

from src.evaluation.judges import AnswerJudgment


class JudgedRecord(Protocol):
    """Record fields required by the judged report."""

    case_id: str
    error: str | None
    judgment: AnswerJudgment | None


def faithfulness_result(records: Sequence[JudgedRecord]) -> str:
    """Aggregate supported claims without counting judge errors as failures."""
    judgments = [
        record.judgment.faithfulness
        for record in records
        if record.judgment is not None
        and record.judgment.faithfulness is not None
    ]
    errors = sum(
        record.judgment is not None
        and record.judgment.faithfulness_error is not None
        for record in records
    )
    total_claims = sum(len(judgment.claims) for judgment in judgments)
    supported = sum(
        judgment.supported_claims for judgment in judgments
    )
    if not total_claims:
        return f"n/a (cases: 0; errors: {errors})"
    return (
        f"{supported / total_claims:.3f} "
        f"({supported}/{total_claims} claims; "
        f"{len(judgments)} cases; errors: {errors})"
    )


def relevance_result(records: Sequence[JudgedRecord]) -> str:
    """Average successful relevance verdicts and disclose error count."""
    judgments = [
        record.judgment.relevance
        for record in records
        if record.judgment is not None
        and record.judgment.relevance is not None
    ]
    errors = sum(
        record.judgment is not None
        and record.judgment.relevance_error is not None
        for record in records
    )
    if not judgments:
        return f"n/a (cases: 0; errors: {errors})"
    average = statistics.fmean(
        judgment.score for judgment in judgments
    )
    return (
        f"{average:.2f}/5.00 "
        f"({len(judgments)} cases; errors: {errors})"
    )


def _safe_markdown_text(value: str) -> str:
    """Keep model-generated audit text on one Markdown line."""
    return (
        html.escape(value, quote=False)
        .replace("|", r"\|")
        .replace("\r", r"\r")
        .replace("\n", r"\n")
    )


def judged_detail_lines(records: Sequence[JudgedRecord]) -> list[str]:
    """Render per-case scores, rationales, rejected claims, and safe errors."""
    lines = [
        "## LLM-as-judge case details",
        "",
        "| case | faithfulness | relevance | status |",
        "|---|---:|---:|---|",
    ]
    rejected: list[tuple[str, str]] = []
    errors: list[tuple[str, str, str]] = []

    for record in records:
        case_id = _safe_markdown_text(record.case_id)
        if record.error is not None:
            lines.append(
                f"| `{case_id}` | n/a | n/a | pipeline_error |"
            )
            continue
        if record.judgment is None:
            lines.append(
                f"| `{case_id}` | n/a | n/a | judge_not_run |"
            )
            continue

        judgment = record.judgment
        faithfulness = "n/a"
        relevance = "n/a"
        statuses: list[str] = []
        if judgment.faithfulness is not None:
            faithfulness = f"{judgment.faithfulness.score:.3f}"
            rejected.extend(
                (record.case_id, claim)
                for claim in judgment.faithfulness.rejected_claims
            )
        if judgment.faithfulness_error is not None:
            statuses.append("faithfulness:judge_error")
            errors.append(
                (
                    record.case_id,
                    "faithfulness",
                    judgment.faithfulness_error,
                )
            )
        if judgment.relevance is not None:
            relevance = f"{judgment.relevance.score}/5"
        if judgment.relevance_error is not None:
            statuses.append("relevance:judge_error")
            errors.append(
                (
                    record.case_id,
                    "relevance",
                    judgment.relevance_error,
                )
            )
        lines.append(
            f"| `{case_id}` | {faithfulness} | {relevance} | "
            f"{', '.join(statuses) if statuses else 'ok'} |"
        )

    lines.append("")
    relevance_records = [
        record
        for record in records
        if record.judgment is not None
        and record.judgment.relevance is not None
    ]
    if relevance_records:
        lines += ["### Relevance reasons", ""]
        for record in relevance_records:
            relevance = record.judgment.relevance
            if relevance is None:  # Narrow the optional type for checkers.
                continue
            lines.append(
                f"- `{_safe_markdown_text(record.case_id)}` "
                f"({relevance.score}/5) — "
                f"{_safe_markdown_text(relevance.reason)}"
            )
        lines.append("")

    lines += ["### Rejected faithfulness claims", ""]
    if rejected:
        for case_id, claim in rejected:
            raw_claim = _safe_markdown_text(
                json.dumps(claim, ensure_ascii=False)
            )
            safe_case_id = _safe_markdown_text(case_id)
            lines.append(f"- `{safe_case_id}` — {raw_claim}")
    else:
        lines.append("None.")
    lines.append("")

    if errors:
        lines += ["### Judge errors", ""]
        for case_id, axis, error in errors:
            safe_case_id = _safe_markdown_text(case_id)
            lines.append(
                f"- `{safe_case_id}` — {axis}: `judge_error:{error}`"
            )
        lines.append("")
    return lines
