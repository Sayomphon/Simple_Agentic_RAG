"""Command-line interface for the two-agent RAG pipeline.

Usage:
    python main.py "What is the policy on international travel?"
    python main.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph

from src.graph import build_graph

BANNER = "=" * 68
DIVIDER = "-" * 68


class QueryExecutionError(RuntimeError):
    """Raised when one graph execution fails at the application boundary."""


def require_api_key() -> None:
    """Load local environment values and stop safely when no API key exists."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "ERROR: OPENAI_API_KEY is not set.\n"
            "Copy .env.example to .env and add your OpenAI API key."
        )


def _print_snippet_block(snippets: list[str]) -> None:
    """Render the evidence handoff as soon as the Retriever finishes."""
    print("[2] RETRIEVED SNIPPETS (Data Retriever -> Report Generator)")
    if snippets:
        for index, snippet in enumerate(snippets, start=1):
            print(f"\n({index})")
            print(snippet)
    else:
        print("(none)")
    print(DIVIDER)
    print("[3] FINAL ANSWER")


def run_query(graph: CompiledStateGraph, query: str) -> dict[str, object]:
    """Run one query, printing evidence and answer tokens as they arrive.

    The streamed tokens are a preview only; the ``report`` field from the
    final graph state remains the source of truth, and the text shown on
    screen always ends byte-equal with it.
    """
    initial_state = {"query": query, "snippets": [], "report": ""}

    print(BANNER)
    print("[1] USER QUERY")
    print(query)
    print(DIVIDER)

    snippets: list[str] = []
    report = ""
    streamed = ""
    emitted = ""
    try:
        events = graph.stream(initial_state, stream_mode=["updates", "messages"])
        for mode, payload in events:
            if mode == "updates":
                for node_name, update in payload.items():
                    if node_name == "data_retriever":
                        snippets = list(update["snippets"])
                        _print_snippet_block(snippets)
                    elif node_name == "report_generator":
                        report = update["report"]
            elif mode == "messages":
                chunk, metadata = payload
                if metadata.get("langgraph_node") != "report_generator":
                    continue
                # The stripped running text only ever grows by appending, so
                # printing its unseen suffix streams the answer while matching
                # the generator's whitespace-trimmed final report.
                streamed += str(chunk.text)
                visible = streamed.strip()
                if len(visible) > len(emitted):
                    print(visible[len(emitted) :], end="", flush=True)
                    emitted = visible
    except Exception as exc:
        raise QueryExecutionError(
            "The RAG pipeline could not process this query"
        ) from exc

    if report.startswith(emitted):
        print(report[len(emitted) :])
    else:
        # The preview diverged from the authoritative report (unusual model
        # content); finish with the exact report so the screen stays truthful.
        if emitted:
            print()
        print(report)
    print(BANNER)
    return {"query": query, "snippets": snippets, "report": report}


def _print_query_error(error: QueryExecutionError) -> None:
    """Render a concise failure without exposing query or pipeline payloads."""
    print(f"ERROR: {error}", file=sys.stderr)


def main() -> int:
    """Validate configuration, compile the graph once, and accept CLI input."""
    require_api_key()
    graph = build_graph()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
        if not query:
            raise SystemExit('Usage: python main.py "<your question>"')
        try:
            run_query(graph, query)
        except QueryExecutionError as exc:
            _print_query_error(exc)
            return 1
        return 0

    print("Simple Agentic RAG — type an empty line, 'exit', or Ctrl-C to quit.")
    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if query.lower() in {"", "exit", "quit"}:
            return 0
        try:
            run_query(graph, query)
        except QueryExecutionError as exc:
            _print_query_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
