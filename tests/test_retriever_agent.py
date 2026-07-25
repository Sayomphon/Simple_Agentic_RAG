"""Tests for tool-call and output-bound guardrails in the Retriever Agent."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.agents.retriever import retriever_node
from src.config import TOP_K
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


class _FakeLLM:
    def __init__(self, tool_calls: list[dict[str, object]]) -> None:
        self._bound = _FakeBoundLLM(tool_calls)

    def bind_tools(self, _tools: object, **_kwargs: object) -> _FakeBoundLLM:
        return self._bound


class RetrieverAgentTests(unittest.TestCase):
    """Ensure provider behaviour cannot bypass retrieval output limits."""

    @patch("src.agents.retriever.search_scored")
    @patch("src.agents.retriever.get_llm")
    def test_executes_only_first_tool_call_and_caps_output(
        self,
        mock_get_llm: Mock,
        mock_search: Mock,
    ) -> None:
        mock_get_llm.return_value = _FakeLLM(
            [
                {"args": {"query": "expanded unrelated HR compensation terms"}},
                {"args": {"query": "second query"}},
            ]
        )
        # A misbehaving lower layer returns more hits than requested; the
        # node's defensive slice must still cap the evidence set.
        mock_search.return_value = _fake_hits(TOP_K + 3)

        result = retriever_node(
            {"query": "What is the CEO's salary?", "snippets": [], "report": ""}
        )

        mock_search.assert_called_once_with(
            "What is the CEO's salary?", top_k=TOP_K, mode=None
        )
        self.assertEqual(len(result["snippets"]), TOP_K)
        self.assertEqual(len(result["hits"]), TOP_K)
        self.assertEqual(
            result["snippets"],
            [hit.as_snippet() for hit in mock_search.return_value[:TOP_K]],
        )

    @patch("src.agents.retriever.search_scored")
    @patch("src.agents.retriever.get_llm")
    def test_non_english_query_uses_model_translation(
        self,
        mock_get_llm: Mock,
        mock_search: Mock,
    ) -> None:
        mock_get_llm.return_value = _FakeLLM(
            [{"args": {"query": "remote work approval"}}]
        )
        mock_search.return_value = _fake_hits(1)

        result = retriever_node(
            {
                "query": (
                    "ทำงานจากบ้านต้อง"
                    "ขออนุมัติอย่างไร"
                ),
                "snippets": [],
                "report": "",
            }
        )

        mock_search.assert_called_once_with(
            "remote work approval", top_k=TOP_K, mode=None
        )
        self.assertEqual(result["search_query"], "remote work approval")
        self.assertEqual(result["snippets"], ["[T0]\nsnippet-0"])

    @patch("src.agents.retriever.search_scored")
    @patch("src.agents.retriever.get_llm")
    def test_state_overrides_reach_the_search(
        self,
        mock_get_llm: Mock,
        mock_search: Mock,
    ) -> None:
        mock_get_llm.return_value = _FakeLLM([{"args": {"query": "annual leave"}}])
        mock_search.return_value = _fake_hits(2)

        retriever_node(
            {
                "query": "annual leave",
                "snippets": [],
                "report": "",
                "search_mode": "hybrid",
                "top_k": 2,
            }
        )

        mock_search.assert_called_once_with("annual leave", top_k=2, mode="hybrid")

    @patch("src.agents.retriever.get_llm")
    def test_missing_tool_call_fails_closed(self, mock_get_llm: Mock) -> None:
        mock_get_llm.return_value = _FakeLLM([])

        result = retriever_node({"query": "question", "snippets": [], "report": ""})

        # Fail closed, but the attempt is still recorded — the retry loop
        # terminates on the attempt count, so it must grow on every pass.
        self.assertEqual(
            result,
            {
                "snippets": [],
                "hits": [],
                "search_attempts": ["question"],
                "rewritten_query": "",
            },
        )


if __name__ == "__main__":
    unittest.main()
