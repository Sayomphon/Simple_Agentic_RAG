"""Opt-in live tests for the real OpenAI/LangChain provider boundary."""

from __future__ import annotations

import os
import unittest
from contextlib import redirect_stdout
from io import StringIO

LIVE_TESTS_ENABLED = os.getenv("RUN_LIVE_LLM_TESTS") == "1"

if LIVE_TESTS_ENABLED:
    # src.config reads MODEL_NAME at import time, so the live-test override
    # must be installed before importing the application graph.
    os.environ["MODEL_NAME"] = os.getenv(
        "LIVE_LLM_TEST_MODEL",
        "gpt-5-mini",
    )

from src.agents.reporter import NOT_FOUND_SENTENCE
from src.graph import build_graph
from src.tools.retrieval import load_knowledge_base


@unittest.skipUnless(
    LIVE_TESTS_ENABLED,
    "Set RUN_LIVE_LLM_TESTS=1 to run live provider tests",
)
class LiveLLMEndToEndTests(unittest.TestCase):
    """Exercise the real Retriever -> Tool -> Reporter integration."""

    @classmethod
    def setUpClass(cls) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise unittest.SkipTest(
                "OPENAI_API_KEY is required for live provider tests"
            )
        cls.graph = build_graph()

    def _invoke(self, state: dict[str, object]) -> dict[str, object]:
        """Fail without echoing provider payloads or raw exception details."""
        try:
            return dict(self.graph.invoke(state))
        except Exception as exc:
            raise AssertionError(
                "Live provider pipeline failed "
                f"({type(exc).__name__}); inspect secure provider logs"
            ) from None

    def test_known_query_runs_real_two_agent_pipeline(self) -> None:
        result = self._invoke(
            {
                "query": "What is the policy on international travel?",
                "snippets": [],
                "report": "",
            }
        )

        titles = [
            snippet.splitlines()[0]
            for snippet in result["snippets"]
        ]
        self.assertEqual(
            titles,
            [
                "--- International Travel Approval Process ---",
                "--- International Travel Daily Allowance ---",
                "--- International Travel Insurance ---",
            ],
        )

        corpus_chunks = set(load_knowledge_base())
        for snippet in result["snippets"]:
            self.assertIn(snippet, corpus_chunks)

        self.assertIsInstance(result["report"], str)
        self.assertTrue(result["report"].strip())
        self.assertNotEqual(result["report"], NOT_FOUND_SENTENCE)

    def test_unknown_query_uses_exact_not_found_response(self) -> None:
        result = self._invoke(
            {
                "query": "What is the CEO's salary?",
                "snippets": [],
                "report": "",
            }
        )

        self.assertEqual(result["snippets"], [])
        self.assertEqual(result["report"], NOT_FOUND_SENTENCE)

    def test_streamed_cli_answer_is_byte_equal_with_the_state_report(self) -> None:
        import main as cli

        output = StringIO()
        try:
            with redirect_stdout(output):
                result = cli.run_query(
                    self.graph,
                    "How much is the daily allowance for international travel?",
                )
        except cli.QueryExecutionError:
            raise AssertionError(
                "Live streamed pipeline failed; inspect secure provider logs"
            ) from None

        rendered_answer = (
            output.getvalue()
            .split("[3] FINAL ANSWER\n", 1)[1]
            .rsplit(cli.BANNER, 1)[0]
        )
        self.assertEqual(rendered_answer.strip("\n"), result["report"])


if __name__ == "__main__":
    unittest.main()
