"""Answer-level evaluation: does the final answer deserve to be trusted?

Usage:
    python -m src.evaluation.run_answer_eval [mode]   # default "hybrid"

Complements the retrieval-layer suites by scoring the *generated answer*
on four axes — two deterministic (no LLM, immune to judge bias) and two
LLM-as-judge:

    citation validity    deterministic  every [Title] cited in the answer
                                        must name a section actually handed
                                        to the generator
    negative discipline  deterministic  negative queries must return the
                                        not-found sentence byte-exactly
    faithfulness         LLM judge      answer decomposed into atomic
                                        claims; each checked against the
                                        snippets; score = supported/total
    answer relevance     LLM judge      1-5: does the answer address the
                                        question, completely, given the
                                        evidence?

The judge sees only (query, snippets, answer) — never a golden answer —
so no reference answers need to be maintained. Unsupported claims are
printed raw in the report rather than hidden. Exits non-zero when a
deterministic axis fails or a judged axis lands under its threshold.

Requires OPENAI_API_KEY (drives the real pipeline + judge calls).
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents import get_llm
from src.agents.reporter import NOT_FOUND_SENTENCE
from src.config import MODEL_NAME, TOP_K
from src.evaluation.run_qa import QUESTIONS, run_question
from src.graph import build_graph

RESULTS_PATH = Path("answer_eval_results.md")
FAITHFULNESS_THRESHOLD = 0.9
RELEVANCE_THRESHOLD = 4.0

_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")


class ClaimVerdict(BaseModel):
    """One atomic claim from the answer, judged against the snippets."""

    claim: str = Field(description="The atomic factual claim, verbatim or lightly normalized")
    supported: bool = Field(description="True only if the snippets state this claim")


class FaithfulnessVerdict(BaseModel):
    """Full claim-by-claim faithfulness decomposition of one answer."""

    claims: list[ClaimVerdict]


class RelevanceVerdict(BaseModel):
    """Overall answer relevance on a 1-5 scale."""

    score: int = Field(ge=1, le=5, description="5 = fully answers the question; 1 = off-topic")
    reason: str = Field(description="One-sentence justification")


FAITHFULNESS_PROMPT = """\
You are a strict evaluation judge. Decompose the ANSWER into its atomic
factual claims (one verifiable statement each; ignore citations in square
brackets and pure formatting). For each claim, decide whether the
SNIPPETS explicitly support it. A claim is supported only if the snippets
state it — reasonable-sounding additions, paraphrases that change
meaning, or outside knowledge are NOT supported.
"""

RELEVANCE_PROMPT = """\
You are a strict evaluation judge. Score how well the ANSWER addresses
the QUESTION on a 1-5 scale, judging only against the SNIPPETS provided:
5 = directly and completely answers what was asked (given the available
evidence); 3 = partially answers or buries the answer in unrelated
material; 1 = does not address the question. Do not reward extra detail
that the question did not ask for.
"""


def judge_faithfulness(query: str, snippets: list[str], answer: str) -> FaithfulnessVerdict:
    llm = get_llm().with_structured_output(FaithfulnessVerdict)
    return llm.invoke(
        [
            SystemMessage(content=FAITHFULNESS_PROMPT),
            HumanMessage(
                content=(
                    f"QUESTION:\n{query}\n\nSNIPPETS:\n"
                    + "\n\n".join(snippets)
                    + f"\n\nANSWER:\n{answer}"
                )
            ),
        ]
    )


def judge_relevance(query: str, snippets: list[str], answer: str) -> RelevanceVerdict:
    llm = get_llm().with_structured_output(RelevanceVerdict)
    return llm.invoke(
        [
            SystemMessage(content=RELEVANCE_PROMPT),
            HumanMessage(
                content=(
                    f"QUESTION:\n{query}\n\nSNIPPETS:\n"
                    + "\n\n".join(snippets)
                    + f"\n\nANSWER:\n{answer}"
                )
            ),
        ]
    )


def check_citations(answer: str, hits) -> tuple[bool, list[str]]:
    """Deterministic: every cited [Title] must be a handed-off section title."""
    available = {hit.title for hit in hits}
    cited = set(_CITATION_PATTERN.findall(answer))
    invalid = sorted(cited - available)
    return not invalid, invalid


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
    graph = build_graph()

    rows: list[dict] = []
    unsupported_blocks: list[str] = []

    for item in QUESTIONS:
        query, category = item["query"], item["category"]
        print(f"[{category:>11}] {query}", flush=True)
        state = run_question(graph, query, mode)
        answer = state["report"]
        hits = state.get("hits") or []
        row: dict = {"query": query, "category": category, "answer": answer}

        if category == "negative":
            row["negative_exact"] = answer == NOT_FOUND_SENTENCE
        elif category == "greeting":
            pass  # direct route: no evidence, nothing to score
        elif answer == NOT_FOUND_SENTENCE:
            # Answerable query that degraded to not-found: no citations or
            # claims to judge — recorded so the miss is visible, not hidden.
            row["degraded_to_not_found"] = True
        else:
            citations_ok, invalid = check_citations(answer, hits)
            row["citations_ok"], row["invalid_citations"] = citations_ok, invalid

            faith = judge_faithfulness(query, state["snippets"], answer)
            total = len(faith.claims)
            supported = sum(c.supported for c in faith.claims)
            row["faithfulness"] = supported / total if total else 1.0
            row["claims_total"], row["claims_supported"] = total, supported
            bad = [c.claim for c in faith.claims if not c.supported]
            if bad:
                unsupported_blocks.append(
                    f"**{query}**\n" + "\n".join(f"- {claim}" for claim in bad)
                )

            row["relevance"] = judge_relevance(query, state["snippets"], answer).score
        rows.append(row)

    # ---- aggregate --------------------------------------------------------
    judged = [r for r in rows if "faithfulness" in r]
    negatives = [r for r in rows if r["category"] == "negative"]
    degraded = [r for r in rows if r.get("degraded_to_not_found")]

    citation_validity = (
        sum(r["citations_ok"] for r in judged) / len(judged) if judged else 1.0
    )
    negative_exact = (
        sum(r["negative_exact"] for r in negatives) / len(negatives)
        if negatives
        else 1.0
    )
    avg_faithfulness = (
        sum(r["faithfulness"] for r in judged) / len(judged) if judged else 0.0
    )
    avg_relevance = (
        sum(r["relevance"] for r in judged) / len(judged) if judged else 0.0
    )

    passed = (
        citation_validity == 1.0
        and negative_exact == 1.0
        and avg_faithfulness >= FAITHFULNESS_THRESHOLD
        and avg_relevance >= RELEVANCE_THRESHOLD
    )

    summary = [
        "| axis | method | result | threshold | pass |",
        "|---|---|---|---|---|",
        f"| citation validity | deterministic | {citation_validity:.0%} of answers "
        f"cite only handed-off sections | 100% | "
        f"{'✅' if citation_validity == 1.0 else '❌'} |",
        f"| negative discipline | deterministic | {negative_exact:.0%} byte-exact "
        f"not-found ({len(negatives)} queries) | 100% | "
        f"{'✅' if negative_exact == 1.0 else '❌'} |",
        f"| faithfulness | LLM judge | {avg_faithfulness:.3f} avg "
        f"(claims supported/total) | ≥ {FAITHFULNESS_THRESHOLD} | "
        f"{'✅' if avg_faithfulness >= FAITHFULNESS_THRESHOLD else '❌'} |",
        f"| answer relevance | LLM judge | {avg_relevance:.2f} avg (1-5) | "
        f"≥ {RELEVANCE_THRESHOLD} | "
        f"{'✅' if avg_relevance >= RELEVANCE_THRESHOLD else '❌'} |",
    ]

    per_question = [
        "| # | category | citations | faithfulness | relevance |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, start=1):
        if r["category"] == "negative":
            detail = "byte-exact ✅" if r["negative_exact"] else "NOT byte-exact ❌"
            per_question.append(f"| {i} | negative | — | — | {detail} |")
        elif r["category"] == "greeting":
            per_question.append(f"| {i} | greeting | — | — | direct route, not scored |")
        elif r.get("degraded_to_not_found"):
            per_question.append(
                f"| {i} | {r['category']} | — | — | degraded to not-found |"
            )
        else:
            cite = "✅" if r["citations_ok"] else "❌ " + ", ".join(r["invalid_citations"])
            per_question.append(
                f"| {i} | {r['category']} | {cite} | "
                f"{r['claims_supported']}/{r['claims_total']} "
                f"({r['faithfulness']:.2f}) | {r['relevance']}/5 |"
            )

    report = [
        "# Answer-Level Evaluation Results",
        "",
        f"- Run date: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Search mode: **{mode}**  ·  model: `{MODEL_NAME}`  ·  `TOP_K={TOP_K}`",
        f"- Questions: {len(rows)} (from `src/evaluation/run_qa.py`); "
        f"{len(judged)} scored on all axes, {len(negatives)} negative, "
        f"{len(degraded)} degraded to not-found",
        "- Judges see only (query, snippets, answer) — no golden answers. "
        "Deterministic axes use no LLM at all.",
        "",
        "## Summary",
        "",
        *summary,
        "",
        "## Per question",
        "",
        *per_question,
        "",
        "## Unsupported claims (raw, judge verdicts)",
        "",
        *(
            ["\n\n".join(unsupported_blocks)]
            if unsupported_blocks
            else ["(none — every claim in every scored answer was judged supported)"]
        ),
        "",
    ]
    RESULTS_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"\nWritten to {RESULTS_PATH}")
    print("\n".join(summary[2:]))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
