"""Offline tests for agent behavior and the sequential LangGraph handoff."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.agents.reporter import NOT_FOUND_SENTENCE, generator_node
from src.agents.retriever import retriever_node
class AgentAndGraphTests(unittest.TestCase):
    @patch("src.agents.retriever.search_knowledge_base")
    @patch("src.agents.retriever.get_llm")
    def test_retriever_executes_tool(
        self,
        mock_get_llm: Mock,
        mock_tool: Mock,
    ) -> None:
        bound_llm = Mock()
        bound_llm.invoke.return_value = SimpleNamespace(
            tool_calls=[{"args": {"query": "international travel"}}]
        )
        mock_get_llm.return_value.bind_tools.return_value = bound_llm
        expected = [
            "--- International Travel Approval Process ---\nRaw evidence."
        ]
        mock_tool.invoke.return_value = expected

        result = retriever_node(
            {"query": "international travel", "snippets": [], "report": ""}
        )

        mock_get_llm.return_value.bind_tools.assert_called_once_with(
            [mock_tool],
            tool_choice="required",
        )
        mock_tool.invoke.assert_called_once_with({"query": "international travel"})
        self.assertEqual(result, {"snippets": expected})

    @patch("src.agents.retriever.get_llm")
    def test_retriever_fails_closed_without_tool_call(
        self,
        mock_get_llm: Mock,
    ) -> None:
        bound_llm = Mock()
        bound_llm.invoke.return_value = SimpleNamespace(tool_calls=[])
        mock_get_llm.return_value.bind_tools.return_value = bound_llm

        result = retriever_node(
            {"query": "annual leave", "snippets": [], "report": ""}
        )

        self.assertEqual(result, {"snippets": []})

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


if __name__ == "__main__":
    unittest.main()
