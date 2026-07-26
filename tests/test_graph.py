"""Offline tests for agent behavior and the sequential LangGraph handoff."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

from src.agents import get_llm
from src.agents.reporter import (
    NOT_FOUND_SENTENCE,
    ReportGenerationError,
    generator_node,
)
from src.agents.retriever import (
    MAX_QUERY_CHARS,
    InvalidQueryError,
    RetrievalProtocolError,
    retriever_node,
)
from src.config import LLM_MAX_RETRIES, LLM_TIMEOUT_SECONDS


class AgentAndGraphTests(unittest.TestCase):
    @staticmethod
    def _state(query: str = "international travel") -> dict[str, object]:
        return {"query": query, "snippets": [], "report": ""}

    @staticmethod
    def _configure_tool(mock_tool: Mock) -> None:
        mock_tool.name = "search_knowledge_base"

    @patch("src.agents.retriever.search_knowledge_base")
    @patch("src.agents.retriever.get_llm")
    def test_retriever_executes_tool_with_exact_original_query(
        self,
        mock_get_llm: Mock,
        mock_tool: Mock,
    ) -> None:
        self._configure_tool(mock_tool)
        bound_llm = Mock()
        bound_llm.invoke.return_value = SimpleNamespace(
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "args": {"query": "international travel"},
                }
            ]
        )
        mock_get_llm.return_value.bind_tools.return_value = bound_llm
        expected = [
            "--- International Travel Approval Process ---\nRaw evidence."
        ]
        mock_tool.invoke.return_value = expected

        result = retriever_node(self._state())

        mock_get_llm.return_value.bind_tools.assert_called_once_with(
            [mock_tool],
            tool_choice="required",
        )
        mock_tool.invoke.assert_called_once_with({"query": "international travel"})
        self.assertEqual(result, {"snippets": expected})

    @patch("src.agents.retriever.search_knowledge_base")
    @patch("src.agents.retriever.get_llm")
    def test_retriever_raises_without_tool_call(
        self,
        mock_get_llm: Mock,
        mock_tool: Mock,
    ) -> None:
        self._configure_tool(mock_tool)
        bound_llm = Mock()
        bound_llm.invoke.return_value = SimpleNamespace(tool_calls=[])
        mock_get_llm.return_value.bind_tools.return_value = bound_llm

        with self.assertRaises(RetrievalProtocolError):
            retriever_node(self._state("annual leave"))

        mock_tool.invoke.assert_not_called()

    @patch("src.agents.retriever.search_knowledge_base")
    @patch("src.agents.retriever.get_llm")
    def test_retriever_raises_for_multiple_tool_calls(
        self,
        mock_get_llm: Mock,
        mock_tool: Mock,
    ) -> None:
        self._configure_tool(mock_tool)
        tool_call = {
            "name": "search_knowledge_base",
            "args": {"query": "annual leave"},
        }
        bound_llm = Mock()
        bound_llm.invoke.return_value = SimpleNamespace(
            tool_calls=[tool_call, tool_call]
        )
        mock_get_llm.return_value.bind_tools.return_value = bound_llm

        with self.assertRaises(RetrievalProtocolError):
            retriever_node(self._state("annual leave"))

        mock_tool.invoke.assert_not_called()

    @patch("src.agents.retriever.search_knowledge_base")
    @patch("src.agents.retriever.get_llm")
    def test_retriever_raises_for_unexpected_tool_name(
        self,
        mock_get_llm: Mock,
        mock_tool: Mock,
    ) -> None:
        self._configure_tool(mock_tool)
        bound_llm = Mock()
        bound_llm.invoke.return_value = SimpleNamespace(
            tool_calls=[
                {
                    "name": "different_tool",
                    "args": {"query": "annual leave"},
                }
            ]
        )
        mock_get_llm.return_value.bind_tools.return_value = bound_llm

        with self.assertRaises(RetrievalProtocolError):
            retriever_node(self._state("annual leave"))

        mock_tool.invoke.assert_not_called()

    @patch("src.agents.retriever.search_knowledge_base")
    @patch("src.agents.retriever.get_llm")
    def test_retriever_rejects_altered_query_before_tool_execution(
        self,
        mock_get_llm: Mock,
        mock_tool: Mock,
    ) -> None:
        self._configure_tool(mock_tool)
        bound_llm = Mock()
        bound_llm.invoke.return_value = SimpleNamespace(
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "args": {"query": "rewritten query"},
                }
            ]
        )
        mock_get_llm.return_value.bind_tools.return_value = bound_llm

        with self.assertRaises(RetrievalProtocolError):
            retriever_node(self._state("exact original query"))

        mock_tool.invoke.assert_not_called()

    @patch("src.agents.retriever.search_knowledge_base")
    @patch("src.agents.retriever.get_llm")
    def test_retriever_preserves_legitimate_empty_search_result(
        self,
        mock_get_llm: Mock,
        mock_tool: Mock,
    ) -> None:
        self._configure_tool(mock_tool)
        bound_llm = Mock()
        bound_llm.invoke.return_value = SimpleNamespace(
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "args": {"query": "unknown topic"},
                }
            ]
        )
        mock_get_llm.return_value.bind_tools.return_value = bound_llm
        mock_tool.invoke.return_value = []

        result = retriever_node(self._state("unknown topic"))

        self.assertEqual(result, {"snippets": []})
        mock_tool.invoke.assert_called_once_with({"query": "unknown topic"})

    @patch("src.agents.reporter.get_llm")
    def test_generator_normalizes_plain_string_content(
        self,
        mock_get_llm: Mock,
    ) -> None:
        mock_get_llm.return_value.invoke.return_value = AIMessage(
            content="  Grounded answer.  "
        )

        result = generator_node(
            {
                "query": "question",
                "snippets": ["--- Evidence ---\nGrounded fact."],
                "report": "",
            }
        )

        self.assertEqual(result, {"report": "Grounded answer."})

    @patch("src.agents.reporter.get_llm")
    def test_generator_extracts_structured_text_content(
        self,
        mock_get_llm: Mock,
    ) -> None:
        mock_get_llm.return_value.invoke.return_value = AIMessage(
            content=[{"type": "text", "text": "Structured answer."}]
        )

        result = generator_node(
            {
                "query": "question",
                "snippets": ["--- Evidence ---\nGrounded fact."],
                "report": "",
            }
        )

        self.assertEqual(result, {"report": "Structured answer."})

    @patch("src.agents.reporter.get_llm")
    def test_generator_combines_multiple_text_blocks(
        self,
        mock_get_llm: Mock,
    ) -> None:
        mock_get_llm.return_value.invoke.return_value = AIMessage(
            content=[
                {"type": "text", "text": "First part. "},
                {"type": "text", "text": "Second part."},
            ]
        )

        result = generator_node(
            {
                "query": "question",
                "snippets": ["--- Evidence ---\nGrounded fact."],
                "report": "",
            }
        )

        self.assertEqual(result, {"report": "First part. Second part."})

    @patch("src.agents.reporter.get_llm")
    def test_generator_raises_for_empty_string_content(
        self,
        mock_get_llm: Mock,
    ) -> None:
        mock_get_llm.return_value.invoke.return_value = AIMessage(content="  ")

        with self.assertRaises(ReportGenerationError):
            generator_node(
                {
                    "query": "question",
                    "snippets": ["--- Evidence ---\nGrounded fact."],
                    "report": "",
                }
            )

    @patch("src.agents.reporter.get_llm")
    def test_generator_raises_for_content_blocks_without_text(
        self,
        mock_get_llm: Mock,
    ) -> None:
        mock_get_llm.return_value.invoke.return_value = AIMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.invalid/image.png"},
                }
            ]
        )

        with self.assertRaises(ReportGenerationError):
            generator_node(
                {
                    "query": "question",
                    "snippets": ["--- Evidence ---\nGrounded fact."],
                    "report": "",
                }
            )

    @patch("src.agents.reporter.get_llm")
    def test_empty_retrieval_uses_not_found_without_llm(
        self,
        mock_get_llm: Mock,
    ) -> None:
        result = generator_node(
            {"query": "What is the CEO's salary?", "snippets": [], "report": ""}
        )

        self.assertEqual(result, {"report": NOT_FOUND_SENTENCE})
        mock_get_llm.assert_not_called()

    def test_graph_handoff(self) -> None:
        received_by_generator: list[list[str]] = []

        def fake_retriever(_state: object) -> dict[str, list[str]]:
            return {"snippets": ["--- Evidence ---\nGrounded fact."]}

        def fake_generator(state: dict[str, object]) -> dict[str, str]:
            snippets = list(state["snippets"])
            received_by_generator.append(snippets)
            return {"report": "Grounded answer."}

        with (
            patch("src.graph.retriever_node", fake_retriever),
            patch("src.graph.generator_node", fake_generator),
        ):
            from src.graph import build_graph

            graph = build_graph()

        result = graph.invoke({"query": "question", "snippets": [], "report": ""})

        self.assertEqual(
            received_by_generator,
            [["--- Evidence ---\nGrounded fact."]],
        )
        self.assertEqual(result["report"], "Grounded answer.")

    def test_graph_contains_only_two_agent_nodes(self) -> None:
        from src.graph import build_graph

        node_names = set(build_graph().get_graph().nodes)

        self.assertEqual(
            node_names,
            {"__start__", "data_retriever", "report_generator", "__end__"},
        )


class QueryValidationTests(unittest.TestCase):
    """Invalid queries must be rejected before any LLM call is attempted."""

    @patch("src.agents.retriever.get_llm")
    def test_rejected_queries_never_reach_the_llm(self, mock_get_llm: Mock) -> None:
        invalid_queries = ["", "   \t\n", "x" * (MAX_QUERY_CHARS + 1), None]

        for query in invalid_queries:
            with self.subTest(query=repr(query)[:30]):
                with self.assertRaises(InvalidQueryError):
                    retriever_node(
                        {"query": query, "snippets": [], "report": ""}
                    )

        mock_get_llm.assert_not_called()

    @patch("src.agents.retriever.search_knowledge_base")
    @patch("src.agents.retriever.get_llm")
    def test_query_at_the_limit_is_accepted(
        self,
        mock_get_llm: Mock,
        mock_tool: Mock,
    ) -> None:
        query = "x" * MAX_QUERY_CHARS
        mock_tool.name = "search_knowledge_base"
        bound_llm = Mock()
        bound_llm.invoke.return_value = SimpleNamespace(
            tool_calls=[{"name": "search_knowledge_base", "args": {"query": query}}]
        )
        mock_get_llm.return_value.bind_tools.return_value = bound_llm
        mock_tool.invoke.return_value = []

        result = retriever_node({"query": query, "snippets": [], "report": ""})

        self.assertEqual(result, {"snippets": []})

    def test_too_long_error_names_the_limit(self) -> None:
        with self.assertRaisesRegex(InvalidQueryError, str(MAX_QUERY_CHARS)):
            retriever_node(
                {
                    "query": "x" * (MAX_QUERY_CHARS + 1),
                    "snippets": [],
                    "report": "",
                }
            )


class LLMClientConstructionTests(unittest.TestCase):
    """get_llm builds one hardened client per distinct model name."""

    def setUp(self) -> None:
        get_llm.cache_clear()
        self._env = patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        self._env.start()
        self.addCleanup(self._env.stop)
        self.addCleanup(get_llm.cache_clear)

    def test_same_model_name_reuses_one_client(self) -> None:
        self.assertIs(get_llm("gpt-5-mini"), get_llm("gpt-5-mini"))

    def test_distinct_model_names_get_distinct_clients(self) -> None:
        retriever_client = get_llm("gpt-5-mini")
        reporter_client = get_llm("gpt-4o-mini")

        self.assertIsNot(retriever_client, reporter_client)
        self.assertEqual(retriever_client.model_name, "gpt-5-mini")
        self.assertEqual(reporter_client.model_name, "gpt-4o-mini")

    def test_timeout_and_retry_budget_reach_the_client(self) -> None:
        client = get_llm("gpt-5-mini")

        self.assertEqual(client.request_timeout, LLM_TIMEOUT_SECONDS)
        self.assertEqual(client.max_retries, LLM_MAX_RETRIES)

    def test_gpt5_models_keep_the_default_temperature(self) -> None:
        gpt5_client = get_llm("gpt-5-mini")
        other_client = get_llm("gpt-4o-mini")

        self.assertIsNone(gpt5_client.temperature)
        self.assertEqual(other_client.temperature, 0.0)


if __name__ == "__main__":
    unittest.main()
