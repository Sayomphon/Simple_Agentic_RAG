"""Opt-in live answer evaluation over the real two-agent pipeline.

Usage:
    RUN_LIVE_LLM_TESTS=1 python -m src.evaluation.run_answer_eval

Runs every labeled query (answer cases + original calibration/held-out
fixtures) through the real graph once and scores deterministic guardrail and
answer-quality axes. It also judges the answer cases for claim-level
faithfulness and answer relevance. Judged axes are reported soft metrics, not
CI gates. The report always records model and prompt provenance.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.evaluation.judges import (
    JUDGE_PROMPT_VERSION,
    AnswerJudgment,
)
from src.evaluation.judge_reporting import (
    faithfulness_result,
    judged_detail_lines,
    relevance_result,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = _PROJECT_ROOT / "answer_eval_results.md"
ANSWER_CASES_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "answer_cases.json"

_NUMBER_PATTERN = re.compile(r"\d[\d,.]*%?")


@dataclass
class QueryRecord:
    """Everything observed for one live query, plus its per-axis verdicts."""

    case_id: str
    query: str
    snippets: list[str] = field(default_factory=list)
    report: str = ""
    llm_calls: int = 0
    error: str | None = None
    failures: list[str] = field(default_factory=list)
    judgment: AnswerJudgment | None = None


def _normalize(text: str) -> str:
    """Casefold, collapse whitespace, and drop commas for fact matching."""
    return " ".join(text.replace(",", "").split()).casefold()


def _extract_numbers(text: str) -> set[str]:
    return {
        match.rstrip(".,")
        for match in _NUMBER_PATTERN.findall(text)
    }


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator * 100:.1f}% ({numerator}/{denominator})"


def main() -> int:
    if os.getenv("RUN_LIVE_LLM_TESTS") != "1":
        print("Skipped: set RUN_LIVE_LLM_TESTS=1 to run the live answer eval.")
        return 0

    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is required for the live answer eval.",
              file=sys.stderr)
        return 1

    # Imported here so the offline gate never needs these modules to have
    # provider credentials available at import time.
    from langchain_core.callbacks import BaseCallbackHandler

    from src.agents import get_llm
    from src.agents.reporter import (
        _CITATION_PATTERN,
        _SNIPPET_TITLE_PATTERN,
        _normalize_title,
        NOT_FOUND_SENTENCE,
    )
    from src.config import (
        MODEL_NAME,
        REPORTER_MODEL_NAME,
        RETRIEVER_MODEL_NAME,
    )
    from src.evaluation.dataset import load_cases
    from src.evaluation.judges import AnswerJudge
    from src.graph import build_graph
    from src.tools.retrieval import load_knowledge_base, search

    class LLMCallCounter(BaseCallbackHandler):
        def __init__(self) -> None:
            self.count = 0

        def on_chat_model_start(self, *args: object, **kwargs: object) -> None:
            self.count += 1

    answer_cases = json.loads(ANSWER_CASES_PATH.read_text(encoding="utf-8"))
    retrieval_cases = [
        {
            "id": f"{dataset_name}:{case.id}",
            "query": case.query,
            "expect_not_found": case.is_negative,
        }
        for dataset_name in ("calibration", "heldout")
        for case in load_cases(dataset_name)
    ]

    corpus_chunks = set(load_knowledge_base())
    graph = build_graph()
    judge_model_name = (
        os.getenv("JUDGE_MODEL_NAME") or MODEL_NAME
    ).strip()

    records: list[QueryRecord] = []
    deep_records: list[tuple[dict[str, object], QueryRecord]] = []

    all_cases = [(case, True) for case in answer_cases] + [
        (case, False) for case in retrieval_cases
    ]
    for position, (case, is_answer_case) in enumerate(all_cases, start=1):
        record = QueryRecord(case_id=str(case["id"]), query=str(case["query"]))
        counter = LLMCallCounter()
        print(
            f"[{position}/{len(all_cases)}] {record.case_id}",
            file=sys.stderr,
        )
        try:
            result = graph.invoke(
                {"query": record.query, "snippets": [], "report": ""},
                config={"callbacks": [counter]},
            )
            record.snippets = list(result["snippets"])
            record.report = str(result["report"])
        except Exception as exc:  # noqa: BLE001 — record and continue.
            record.error = type(exc).__name__
        record.llm_calls = counter.count
        records.append(record)
        if is_answer_case:
            deep_records.append((case, record))

    # --- Guardrail axes over every successfully executed query. ----------
    executed = [record for record in records if record.error is None]

    provenance_ok = citation_ok = coverage_ok = 0
    empty_retrieval = empty_retrieval_ok = 0
    not_found_expected = not_found_ok = 0

    expect_not_found_ids = {
        str(case["id"])
        for case in answer_cases
        if case.get("expect_not_found")
    } | {
        str(case["id"])
        for case in retrieval_cases
        if case["expect_not_found"]
    }

    for record in executed:
        if all(snippet in corpus_chunks for snippet in record.snippets):
            provenance_ok += 1
        else:
            record.failures.append("evidence_provenance")

        allowed_titles = set()
        for snippet in record.snippets:
            first_line = snippet.partition("\n")[0]
            match = _SNIPPET_TITLE_PATTERN.match(first_line)
            if match:
                allowed_titles.add(_normalize_title(match.group("title")))
        citations = _CITATION_PATTERN.findall(record.report)
        if all(
            _normalize_title(citation) in allowed_titles
            for citation in citations
        ):
            citation_ok += 1
        else:
            record.failures.append("citation_validity")

        if set(search(record.query)) <= set(record.snippets):
            coverage_ok += 1
        else:
            record.failures.append("baseline_coverage")

        if not record.snippets:
            empty_retrieval += 1
            if record.llm_calls == 1:
                empty_retrieval_ok += 1
            else:
                record.failures.append("no_llm_on_empty")

        if record.case_id in expect_not_found_ids:
            not_found_expected += 1
            if record.report == NOT_FOUND_SENTENCE:
                not_found_ok += 1
            else:
                record.failures.append("not_found_discipline")

    # --- Deep answer metrics over the answer cases. ----------------------
    required_total = required_found = 0
    numbers_total = numbers_unsupported = 0
    forbidden_violations: list[str] = []

    for case, record in deep_records:
        if record.error is not None:
            continue
        normalized_report = _normalize(record.report)
        for fact in case.get("required_facts", []):
            required_total += 1
            if _normalize(str(fact)) in normalized_report:
                required_found += 1
            else:
                record.failures.append(f"missing_fact:{fact}")

        if record.report != NOT_FOUND_SENTENCE:
            normalized_evidence = _normalize("\n".join(record.snippets))
            for number in _extract_numbers(record.report):
                numbers_total += 1
                if _normalize(number) not in normalized_evidence:
                    numbers_unsupported += 1
                    record.failures.append(f"unsupported_number:{number}")

        for fact in case.get("forbidden_facts", []):
            if _normalize(str(fact)) in normalized_report:
                forbidden_violations.append(
                    f"{record.case_id}: {fact}"
                )
                record.failures.append(f"forbidden_fact:{fact}")

    # --- Soft LLM-as-judge metrics over answer cases only. ---------------
    deep_answer_records = [record for _, record in deep_records]
    try:
        judge = AnswerJudge.from_chat_model(get_llm(judge_model_name))
    except Exception as exc:  # noqa: BLE001 - report setup failure safely.
        setup_error = type(exc).__name__
        for record in deep_answer_records:
            if record.error is None:
                record.judgment = AnswerJudgment(
                    faithfulness=None,
                    relevance=None,
                    faithfulness_error=setup_error,
                    relevance_error=setup_error,
                )
    else:
        judgeable = [
            record
            for record in deep_answer_records
            if record.error is None
        ]
        for position, record in enumerate(judgeable, start=1):
            print(
                f"[judge {position}/{len(judgeable)}] {record.case_id}",
                file=sys.stderr,
            )
            record.judgment = judge.evaluate(
                record.query,
                record.snippets,
                record.report,
            )

    # --- Report. ---------------------------------------------------------
    errored = [record for record in records if record.error is not None]
    lines = [
        "# Answer Evaluation Results",
        "",
        "Generated by `RUN_LIVE_LLM_TESTS=1 python -m "
        "src.evaluation.run_answer_eval`.",
        "",
        "Metrics are computed by deterministic matching, but the generator "
        "output itself is probabilistic — this is **not** a claim of "
        "deterministic answer quality.",
        "",
        f"- Retriever model: `{RETRIEVER_MODEL_NAME}`",
        f"- Reporter model: `{REPORTER_MODEL_NAME}`",
        f"- Judge model: `{judge_model_name}`",
        f"- Judge prompt version: `{JUDGE_PROMPT_VERSION}`",
        f"- Prompt version (commit): `{_git_commit()}`",
        "- Runs per case: 1",
        f"- Queries executed: {len(executed)}/{len(records)} "
        f"({len(deep_records)} answer cases, "
        f"{len(retrieval_cases)} retrieval-fixture cases; "
        f"errors: {len(errored)})",
        "",
        "Deterministic axes remain release gates. LLM-judged axes are "
        "**reported soft metrics only**: one judge, no ensemble, one run, "
        "and scores inherit the judge model's strictness.",
        "",
        "| axis | method | result | threshold |",
        "|---|---|---|---|",
        f"| citation_validity (runtime-enforced) | deterministic | "
        f"{_percent(citation_ok, len(executed))} | 100% |",
        f"| not_found_discipline | "
        f"deterministic | {_percent(not_found_ok, not_found_expected)} | "
        "100% |",
        f"| evidence_provenance | deterministic | "
        f"{_percent(provenance_ok, len(executed))} | 100% |",
        f"| no_llm_on_empty | deterministic | "
        f"{_percent(empty_retrieval_ok, empty_retrieval)} | 100% |",
        f"| baseline_coverage | deterministic | "
        f"{_percent(coverage_ok, len(executed))} | 100% |",
        f"| required_fact_coverage | deterministic | "
        f"{_percent(required_found, required_total)} | 100% |",
        f"| unsupported_number_rate | deterministic | "
        f"{_percent(numbers_unsupported, numbers_total)} | 0% |",
        f"| forbidden_fact_violations | deterministic | "
        f"{len(forbidden_violations)} | 0 |",
        f"| faithfulness | single LLM judge, claim-level | "
        f"{faithfulness_result(deep_answer_records)} | ≥ 0.900 (soft) |",
        f"| answer_relevance | single LLM judge, 1–5 | "
        f"{relevance_result(deep_answer_records)} | ≥ 4.00 (soft) |",
        "",
    ]

    lines += judged_detail_lines(deep_answer_records)

    imperfect = [record for record in records if record.failures or record.error]
    if imperfect:
        lines += ["## Deterministic imperfect cases", ""]
        for record in imperfect:
            details = record.error or ", ".join(record.failures)
            lines.append(f"- `{record.case_id}` — {details}")
            lines.append(f"  - query: {record.query!r}")
            if record.error is None:
                lines.append(f"  - report: {record.report!r}")
        lines.append("")
    else:
        lines += ["Every case passed every deterministic axis.", ""]

    RESULTS_PATH.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
