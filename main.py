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

from src.agents.retriever import InvalidQueryError
from src.graph import build_graph
from src.retrievers.base import SearchTelemetry, SnippetTrace

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


def _section_title(snippet: str) -> str:
    """Extract a display title from a validated raw section."""
    header = snippet.partition("\n")[0].strip()
    if header.startswith("---") and header.endswith("---"):
        return header[3:-3].strip()
    return header


def _first_trace_by_title(
    telemetry: list[SearchTelemetry],
) -> dict[str, SnippetTrace]:
    """Map unioned snippets to the first retrieval attempt that returned them."""
    traces: dict[str, SnippetTrace] = {}
    for attempt in telemetry:
        for trace in attempt.snippets:
            traces.setdefault(trace.title, trace)
    return traces


def _print_snippet_block(
    snippets: list[str],
    telemetry: list[SearchTelemetry],
) -> None:
    """Render the evidence handoff as soon as the Retriever finishes."""
    print("[2] RETRIEVED SNIPPETS (Data Retriever -> Report Generator)")
    traces = _first_trace_by_title(telemetry)
    if snippets:
        for index, snippet in enumerate(snippets, start=1):
            trace = traces.get(_section_title(snippet))
            if trace is None:
                print(f"\n({index})")
            else:
                print(
                    f"\n#{index} [{trace.title}] "
                    f"score={trace.score:.4f} method={trace.method}"
                )
            print(snippet)
    else:
        print("(none)")

    if telemetry:
        modes = "/".join(dict.fromkeys(item.mode for item in telemetry))
        total_latency = sum(item.latency_ms for item in telemetry)
        print(
            f"\nmode={modes} attempts={len(telemetry)} "
            f"retrieval={total_latency:.2f}ms"
        )
        for index, attempt in enumerate(telemetry, start=1):
            if attempt.empty_reason == "no_query_terms":
                print(
                    f"attempt {index}: query produced no searchable terms "
                    f"({attempt.mode} mode)"
                )
            elif attempt.empty_reason == "gated_out":
                print(
                    f"attempt {index}: no section passed the relevance gate "
                    f"({attempt.mode} mode)"
                )
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
    telemetry: list[SearchTelemetry] = []
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
                        telemetry = list(
                            update.get("retrieval_telemetry", [])
                        )
                        _print_snippet_block(snippets, telemetry)
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
    except InvalidQueryError as exc:
        # Pre-LLM validation messages are written by this codebase, name the
        # limit that was hit, and never quote the query — safe to surface.
        raise QueryExecutionError(str(exc)) from exc
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
    return {
        "query": query,
        "snippets": snippets,
        "report": report,
        "retrieval_telemetry": telemetry,
    }


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
