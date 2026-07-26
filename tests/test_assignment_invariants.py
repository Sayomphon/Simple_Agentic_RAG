"""Executable guards for the assignment's non-negotiable architecture."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from langgraph.graph import END, START

from src import config
from src.agents.reporter import generator_node
from src.graph import PipelineState, build_graph
from src.retrievers.factory import clear_retriever_cache
from src.tools.retrieval import search_knowledge_base


class AssignmentInvariantTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_retriever_cache()

    def test_graph_has_exactly_two_agent_nodes(self) -> None:
        graph = build_graph()

        self.assertEqual(
            set(graph.builder.nodes),
            {"data_retriever", "report_generator"},
        )

    def test_graph_edges_are_fixed_and_sequential(self) -> None:
        graph = build_graph()

        self.assertEqual(
            graph.builder.edges,
            {
                (START, "data_retriever"),
                ("data_retriever", "report_generator"),
                ("report_generator", END),
            },
        )
        self.assertFalse(graph.builder.branches)

    def test_pipeline_required_handoff_fields_are_unchanged(self) -> None:
        self.assertEqual(
            PipelineState.__required_keys__,
            frozenset({"query", "snippets", "report"}),
        )
        self.assertEqual(
            PipelineState.__optional_keys__,
            frozenset({"retrieval_telemetry"}),
        )

    def test_tool_schema_and_raw_chunk_contract_are_unchanged(self) -> None:
        snippets = search_knowledge_base.invoke(
            {"query": "international travel"}
        )
        corpus = Path(config.KB_PATH).read_text(encoding="utf-8")
        schema = search_knowledge_base.args_schema.model_json_schema()

        self.assertEqual(search_knowledge_base.name, "search_knowledge_base")
        self.assertEqual(schema["required"], ["query"])
        self.assertEqual(set(schema["properties"]), {"query"})
        self.assertIsInstance(snippets, list)
        self.assertTrue(snippets)
        self.assertTrue(all(isinstance(chunk, str) for chunk in snippets))
        self.assertTrue(all(chunk in corpus for chunk in snippets))

    @patch("src.agents.reporter.get_llm")
    def test_reporter_invokes_no_tools(self, mock_get_llm: Mock) -> None:
        llm = Mock()
        llm.invoke.return_value = AIMessage(
            content="Supported fact. [Remote Work Policy]"
        )
        mock_get_llm.return_value = llm
        snippet = (
            "--- Remote Work Policy ---\n"
            "Employees may work remotely up to 3 days per week."
        )

        generator_node(
            {
                "query": "How many remote days are allowed?",
                "snippets": [snippet],
                "report": "",
            }
        )

        llm.invoke.assert_called_once()
        llm.bind_tools.assert_not_called()

    def test_default_lexical_mode_needs_no_api_key(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}),
            patch.object(config, "SEARCH_MODE", "lexical"),
        ):
            clear_retriever_cache()
            snippets = search_knowledge_base.invoke(
                {"query": "international travel"}
            )

        self.assertTrue(snippets)


if __name__ == "__main__":
    unittest.main()
