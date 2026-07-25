"""Tests for the agentic retry loop: rewriter node + conditional routing.

All LLM calls are faked (no API key needed). The full compiled graph is
exercised so the conditional edges — found -> generate, empty -> rewrite,
budget spent -> deterministic not-found — are what is under test.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.agents.reporter import NOT_FOUND_SENTENCE
from src.agents.retriever import retriever_node
from src.config import MAX_SEARCH_ATTEMPTS
from src.graph import build_graph
from src.retrievers.base import Chunk, ScoredChunk


def _fake_hits(count: int) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(title=f"T{i}", text=f"snippet-{i}", index=i),
            score=float(count - i),
            source="bm25",
        )
        for i in range(count)
    ]


class _FakeBoundLLM:
    def __init__(self, tool_calls: list[dict[str, object]]) -> None:
        self._tool_calls = tool_calls

    def invoke(self, _messages: object) -> SimpleNamespace:
        return SimpleNamespace(tool_calls=self._tool_calls)


class _FakeToolLLM:
    """Fake for nodes that force a tool call (retriever, rewriter)."""

    def __init__(self, tool_calls: list[dict[str, object]]) -> None:
        self._bound = _FakeBoundLLM(tool_calls)

    def bind_tools(self, _tools: object, **_kwargs: object) -> _FakeBoundLLM:
        return self._bound


class _FakeChatLLM:
    """Fake for the reporter's plain chat call."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0

    def invoke(self, _messages: object) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(content=self._content)


def _initial_state(query: str) -> dict[str, object]:
    return {"query": query, "snippets": [], "report": ""}


@patch("src.agents.router.get_llm")
@patch("src.agents.reporter.get_llm")
@patch("src.agents.rewriter.get_llm")
@patch("src.agents.retriever.search_scored")
@patch("src.agents.retriever.get_llm")
class RewriterLoopTests(unittest.TestCase):
    """Exercise the compiled graph's retry loop end to end (LLM-free).

    The router LLM is faked to answer ``kb_query`` so every test enters
    the retrieval path without a real API call.
    """

    def test_first_attempt_hit_skips_rewriter(
        self,
        mock_retriever_llm: Mock,
        mock_search: Mock,
        mock_rewriter_llm: Mock,
        mock_reporter_llm: Mock,
        mock_router_llm: Mock,
    ) -> None:
        mock_router_llm.return_value = _FakeToolLLM([{"args": {"route": "kb_query"}}])
        mock_retriever_llm.return_value = _FakeToolLLM(
            [{"args": {"query": "annual leave"}}]
        )
        mock_search.return_value = _fake_hits(2)
        mock_reporter_llm.return_value = _FakeChatLLM("grounded answer")

        result = build_graph().invoke(_initial_state("annual leave"))

        mock_rewriter_llm.assert_not_called()
        mock_search.assert_called_once()
        self.assertEqual(result["search_attempts"], ["annual leave"])
        self.assertEqual(result["report"], "grounded answer")

    def test_empty_first_attempt_retries_with_rewritten_query(
        self,
        mock_retriever_llm: Mock,
        mock_search: Mock,
        mock_rewriter_llm: Mock,
        mock_reporter_llm: Mock,
        mock_router_llm: Mock,
    ) -> None:
        mock_router_llm.return_value = _FakeToolLLM([{"args": {"route": "kb_query"}}])
        mock_retriever_llm.return_value = _FakeToolLLM(
            [{"args": {"query": "quit my job"}}]
        )
        mock_rewriter_llm.return_value = _FakeToolLLM(
            [{"args": {"query": "resignation notice period process"}}]
        )
        # First attempt misses; the rewritten query hits.
        mock_search.side_effect = [[], _fake_hits(1)]
        mock_reporter_llm.return_value = _FakeChatLLM("resignation answer")

        result = build_graph().invoke(_initial_state("I want to quit my job"))

        self.assertEqual(mock_search.call_count, 2)
        self.assertEqual(
            mock_search.call_args_list[1].args[0],
            "resignation notice period process",
        )
        self.assertEqual(
            result["search_attempts"],
            ["I want to quit my job", "resignation notice period process"],
        )
        self.assertEqual(result["report"], "resignation answer")
        # The retriever LLM ran only on the first attempt; the retry
        # trusted the rewriter's query directly.
        self.assertEqual(mock_retriever_llm.call_count, 1)

    def test_budget_exhausted_falls_back_deterministically(
        self,
        mock_retriever_llm: Mock,
        mock_search: Mock,
        mock_rewriter_llm: Mock,
        mock_reporter_llm: Mock,
        mock_router_llm: Mock,
    ) -> None:
        mock_router_llm.return_value = _FakeToolLLM([{"args": {"route": "kb_query"}}])
        mock_retriever_llm.return_value = _FakeToolLLM(
            [{"args": {"query": "executive compensation"}}]
        )
        mock_rewriter_llm.return_value = _FakeToolLLM(
            [{"args": {"query": "CEO pay disclosure"}}]
        )
        mock_search.return_value = []  # every attempt misses

        result = build_graph().invoke(_initial_state("What is the CEO's salary?"))

        self.assertEqual(result["report"], NOT_FOUND_SENTENCE)
        self.assertEqual(len(result["search_attempts"]), MAX_SEARCH_ATTEMPTS)
        self.assertEqual(mock_search.call_count, MAX_SEARCH_ATTEMPTS)
        # Deterministic fallback: the reporter never called an LLM.
        mock_reporter_llm.assert_not_called()

    def test_rewritten_query_is_not_overridden_by_query_selection(
        self,
        mock_retriever_llm: Mock,
        mock_search: Mock,
        mock_rewriter_llm: Mock,
        mock_reporter_llm: Mock,
        mock_router_llm: Mock,
    ) -> None:
        # English user query would normally make _select_search_query copy
        # the user's wording — the rewritten query must win instead, with
        # no retriever LLM call at all.
        mock_search.return_value = _fake_hits(1)

        result = retriever_node(
            {
                "query": "I want to quit my job",
                "snippets": [],
                "report": "",
                "search_attempts": ["I want to quit my job"],
                "rewritten_query": "resignation process",
            }
        )

        mock_retriever_llm.assert_not_called()
        self.assertEqual(mock_search.call_args.args[0], "resignation process")
        self.assertEqual(result["search_query"], "resignation process")
        self.assertEqual(
            result["search_attempts"],
            ["I want to quit my job", "resignation process"],
        )
        # Cleared after use so a stale rewrite cannot leak forward.
        self.assertEqual(result["rewritten_query"], "")


if __name__ == "__main__":
    unittest.main()
