"""Tests for the Router agent: route plumbing and the fail-safe default.

All LLM calls are faked. What the mocks can prove is the wiring — small
talk skips retrieval entirely, informational routes take the full
retrieval path, and every ambiguous or malformed verdict fails safe to
``kb_query`` (the path with the not-found guardrail behind it). Whether
the real model classifies specific queries correctly is verified end to
end in the sample Q&A run, not here.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.agents.router import ROUTE_DIRECT, ROUTE_KB, router_node
from src.graph import build_graph
from src.retrievers.base import Chunk, ScoredChunk


class _FakeBoundLLM:
    def __init__(self, tool_calls: list[dict[str, object]]) -> None:
        self._tool_calls = tool_calls

    def invoke(self, _messages: object) -> SimpleNamespace:
        return SimpleNamespace(tool_calls=self._tool_calls)


class _FakeRouterLLM:
    """Fake covering both router uses: forced tool call + plain chat."""

    def __init__(self, route: str, direct_reply: str = "hi!") -> None:
        self._bound = _FakeBoundLLM([{"args": {"route": route}}])
        self._direct_reply = direct_reply

    def bind_tools(self, _tools: object, **_kwargs: object) -> _FakeBoundLLM:
        return self._bound

    def invoke(self, _messages: object) -> SimpleNamespace:
        return SimpleNamespace(content=self._direct_reply)


class _FakeToolLLM:
    def __init__(self, tool_calls: list[dict[str, object]]) -> None:
        self._bound = _FakeBoundLLM(tool_calls)

    def bind_tools(self, _tools: object, **_kwargs: object) -> _FakeBoundLLM:
        return self._bound


class _FakeChatLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, _messages: object) -> SimpleNamespace:
        return SimpleNamespace(content=self._content)


def _initial_state(query: str) -> dict[str, object]:
    return {"query": query, "snippets": [], "report": ""}


class RouterNodeTests(unittest.TestCase):
    """Fail-safe behaviour of the classifier itself."""

    @patch("src.agents.router.get_llm")
    def test_direct_verdict_routes_direct(self, mock_get_llm: Mock) -> None:
        mock_get_llm.return_value = _FakeRouterLLM(" Direct ")
        self.assertEqual(
            router_node(_initial_state("hello")), {"route": ROUTE_DIRECT}
        )

    @patch("src.agents.router.get_llm")
    def test_unknown_verdict_fails_safe_to_kb(self, mock_get_llm: Mock) -> None:
        mock_get_llm.return_value = _FakeRouterLLM("chitchat")
        self.assertEqual(
            router_node(_initial_state("hello")), {"route": ROUTE_KB}
        )

    @patch("src.agents.router.get_llm")
    def test_missing_tool_call_fails_safe_to_kb(self, mock_get_llm: Mock) -> None:
        mock_get_llm.return_value = _FakeToolLLM([])
        self.assertEqual(
            router_node(_initial_state("hello")), {"route": ROUTE_KB}
        )


@patch("src.agents.reporter.get_llm")
@patch("src.agents.router.get_llm")
@patch("src.agents.retriever.search_scored")
@patch("src.agents.retriever.get_llm")
class RouterGraphTests(unittest.TestCase):
    """Route wiring through the compiled graph."""

    def test_direct_route_never_touches_retrieval(
        self,
        mock_retriever_llm: Mock,
        mock_search: Mock,
        mock_router_llm: Mock,
        mock_reporter_llm: Mock,
    ) -> None:
        mock_router_llm.return_value = _FakeRouterLLM(
            ROUTE_DIRECT, direct_reply="Hello! Ask me about the handbook."
        )

        result = build_graph().invoke(_initial_state("Hello, what can you do?"))

        mock_search.assert_not_called()
        mock_retriever_llm.assert_not_called()
        mock_reporter_llm.assert_not_called()
        self.assertEqual(result["route"], ROUTE_DIRECT)
        self.assertEqual(result["report"], "Hello! Ask me about the handbook.")

    def test_kb_route_runs_the_full_retrieval_path(
        self,
        mock_retriever_llm: Mock,
        mock_search: Mock,
        mock_router_llm: Mock,
        mock_reporter_llm: Mock,
    ) -> None:
        mock_router_llm.return_value = _FakeRouterLLM(ROUTE_KB)
        mock_retriever_llm.return_value = _FakeToolLLM(
            [{"args": {"query": "international travel policy"}}]
        )
        mock_search.return_value = [
            ScoredChunk(
                chunk=Chunk(title="International Travel", text="…", index=0),
                score=5.0,
                source="bm25",
            )
        ]
        mock_reporter_llm.return_value = _FakeChatLLM("travel answer")

        result = build_graph().invoke(
            _initial_state("What is the policy on international travel?")
        )

        mock_search.assert_called_once()
        self.assertEqual(result["route"], ROUTE_KB)
        self.assertEqual(result["report"], "travel answer")


if __name__ == "__main__":
    unittest.main()
