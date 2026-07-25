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


def run_query(graph: CompiledStateGraph, query: str) -> dict[str, object]:
    """Run one query and print the evidence handoff before the final answer."""
    initial_state = {"query": query, "snippets": [], "report": ""}
    try:
        result = graph.invoke(initial_state)
    except Exception as exc:
        raise QueryExecutionError(
            "The RAG pipeline could not process this query"
        ) from exc

    print(BANNER)
    print("[1] USER QUERY")
    print(query)
    print(DIVIDER)
    print("[2] RETRIEVED SNIPPETS (Data Retriever -> Report Generator)")
    if result["snippets"]:
        for index, snippet in enumerate(result["snippets"], start=1):
            print(f"\n({index})")
            print(snippet)
    else:
        print("(none)")
    print(DIVIDER)
    print("[3] FINAL ANSWER")
    print(result["report"])
    print(BANNER)
    return dict(result)


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
