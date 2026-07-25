"""Run sample questions through the full agentic pipeline, end to end.

Usage:
    python -m src.evaluation.run_qa [mode]      # mode defaults to "hybrid"

Unlike ``run_eval`` (retrieval layer only, no LLM), this harness drives
the real compiled graph — router, retriever with its rewrite/retry loop,
and generator — exactly as the CLI and UI do, and records every decision
the pipeline made per question: the chosen route, each search attempt,
the retrieved evidence with scores and provenance, and the final answer.
The transcript is written to ``sample_qa_results.md`` so reviewers can
read real Q→A behaviour without running anything.

Requires OPENAI_API_KEY (real LLM + embedding calls).
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

from src.config import MAX_SEARCH_ATTEMPTS, MODEL_NAME, TOP_K
from src.evaluation.testset import TEST_SET
from src.graph import build_graph

RESULTS_PATH = Path("sample_qa_results.md")

# Every golden-set question (13 answerable across all categories + 2
# negative), plus two specials the golden set cannot express: a Thai
# query (cross-lingual retrieval) and a greeting (router's direct path).
QUESTIONS: list[dict[str, str]] = [
    *(
        {"id": str(case["id"]), "category": str(case["category"]),
         "query": str(case["query"])}
        for case in TEST_SET
    ),
    {
        "id": "thai_ordination",
        "category": "thai",
        "query": "ลาบวชได้กี่วัน และต้องแจ้งล่วงหน้าอย่างไร?",
    },
    {
        "id": "greeting_router",
        "category": "greeting",
        "query": "Hello! What can you do?",
    },
]


def run_question(graph, query: str, mode: str) -> dict[str, object]:
    """Invoke the compiled graph once and time it."""
    started = time.perf_counter()
    state = graph.invoke(
        {"query": query, "snippets": [], "report": "", "search_mode": mode}
    )
    state["elapsed_s"] = time.perf_counter() - started
    return state


def format_question(number: int, item: dict[str, str], state: dict) -> str:
    """Render one Q→A transcript block as markdown."""
    lines = [
        f"## Q{number}. {item['query']}",
        "",
        f"- category: `{item['category']}`  ·  route: "
        f"`{state.get('route', 'kb_query')}`  ·  "
        f"{state['elapsed_s']:.1f}s",
    ]

    if state.get("route") != "direct":
        attempts = state.get("search_attempts", [])
        lines.append("- search attempts:")
        for i, attempt in enumerate(attempts, start=1):
            # Every attempt before the last found nothing by construction.
            found = len(state.get("snippets", [])) if i == len(attempts) else 0
            lines.append(f'  {i}. "{attempt}" → {found} result(s)')
        hits = state.get("hits") or []
        if hits:
            lines.append("- retrieved evidence:")
            lines.extend(
                f"  - **{hit.title}** (score {hit.score:.4f}, {hit.source}, "
                f"`{hit.source_file}`)"
                for hit in hits
            )
        else:
            lines.append("- retrieved evidence: none passed the relevance gates")

    lines.append("- answer:")
    lines.extend(f"  > {line}" for line in state["report"].splitlines())
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
    graph = build_graph()
    blocks = [
        "# Sample Q→A Results (end-to-end)",
        "",
        f"- Run date: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Pipeline: full agentic graph via `build_graph()` — router → "
        f"data retriever (with query-rewrite retry loop) → report generator",
        f"- Search mode: **{mode}**  ·  model: `{MODEL_NAME}`  ·  "
        f"`TOP_K={TOP_K}`  ·  `MAX_SEARCH_ATTEMPTS={MAX_SEARCH_ATTEMPTS}`",
        f"- Questions: {len(QUESTIONS)} — the 15 golden-set queries "
        "(lexical, semantic, multi-chunk, negative) plus a Thai query and "
        "a greeting",
        "",
        "Each block records the agent's actual decisions: the route, every "
        "search attempt (a new attempt means the previous one returned "
        "nothing and the query was rewritten), the evidence handed to the "
        "generator, and the final answer verbatim.",
        "",
    ]

    for number, item in enumerate(QUESTIONS, start=1):
        print(f"[{number:2}/{len(QUESTIONS)}] {item['query']}", flush=True)
        state = run_question(graph, item["query"], mode)
        blocks.append(format_question(number, item, state))

    RESULTS_PATH.write_text("\n".join(blocks), encoding="utf-8")
    print(f"\nWritten to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
