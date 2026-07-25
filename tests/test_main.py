"""Offline tests for CLI rendering and application error boundaries."""

from __future__ import annotations

import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import Mock, call, patch

import main as cli


class MainTests(unittest.TestCase):
    def test_run_query_wraps_graph_error_and_preserves_cause(self) -> None:
        graph = Mock()
        original_error = RuntimeError("provider failure")
        graph.invoke.side_effect = original_error

        with self.assertRaises(cli.QueryExecutionError) as context:
            cli.run_query(graph, "sensitive user query")

        self.assertIs(context.exception.__cause__, original_error)
        self.assertNotIn("sensitive user query", str(context.exception))

    def test_successful_run_query_renders_and_returns_result(self) -> None:
        graph = Mock()
        graph.invoke.return_value = {
            "query": "international travel",
            "snippets": ["--- Travel ---\nGrounded evidence."],
            "report": "Grounded answer.",
        }
        output = StringIO()

        with redirect_stdout(output):
            result = cli.run_query(graph, "international travel")

        self.assertEqual(result["report"], "Grounded answer.")
        self.assertIn("--- Travel ---\nGrounded evidence.", output.getvalue())
        self.assertIn("Grounded answer.", output.getvalue())

    @patch("main.run_query")
    @patch("main.build_graph")
    @patch("main.require_api_key")
    def test_interactive_mode_continues_after_query_failure(
        self,
        _mock_require_api_key: Mock,
        mock_build_graph: Mock,
        mock_run_query: Mock,
    ) -> None:
        graph = mock_build_graph.return_value
        mock_run_query.side_effect = [
            cli.QueryExecutionError("The RAG pipeline could not process this query"),
            {"report": "Recovered."},
        ]
        stdout = StringIO()
        stderr = StringIO()

        with (
            patch.object(sys, "argv", ["main.py"]),
            patch(
                "builtins.input",
                side_effect=["first private query", "second query", "exit"],
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_status = cli.main()

        self.assertEqual(exit_status, 0)
        self.assertEqual(
            mock_run_query.call_args_list,
            [
                call(graph, "first private query"),
                call(graph, "second query"),
            ],
        )
        self.assertIn("ERROR:", stderr.getvalue())
        self.assertNotIn("first private query", stderr.getvalue())

    @patch("main.build_graph")
    @patch("main.require_api_key")
    def test_single_query_returns_nonzero_without_sensitive_error_data(
        self,
        _mock_require_api_key: Mock,
        mock_build_graph: Mock,
    ) -> None:
        raw_query = "customer secret query"
        secret = "credential-shaped-secret"
        mock_build_graph.return_value.invoke.side_effect = RuntimeError(
            f"{raw_query}: {secret}"
        )
        stderr = StringIO()

        with (
            patch.object(sys, "argv", ["main.py", raw_query]),
            redirect_stderr(stderr),
        ):
            exit_status = cli.main()

        self.assertEqual(exit_status, 1)
        self.assertIn("ERROR:", stderr.getvalue())
        self.assertNotIn(raw_query, stderr.getvalue())
        self.assertNotIn(secret, stderr.getvalue())

    @patch("main.run_query")
    @patch("main.build_graph")
    @patch("main.require_api_key")
    def test_interactive_exit_inputs_do_not_execute_queries(
        self,
        _mock_require_api_key: Mock,
        _mock_build_graph: Mock,
        mock_run_query: Mock,
    ) -> None:
        for exit_input in ("", "exit", "QUIT"):
            with self.subTest(exit_input=exit_input):
                with (
                    patch.object(sys, "argv", ["main.py"]),
                    patch("builtins.input", return_value=exit_input),
                    redirect_stdout(StringIO()),
                ):
                    self.assertEqual(cli.main(), 0)

        mock_run_query.assert_not_called()

    @patch("main.run_query")
    @patch("main.build_graph")
    @patch("main.require_api_key")
    def test_interactive_eof_and_keyboard_interrupt_exit_cleanly(
        self,
        _mock_require_api_key: Mock,
        _mock_build_graph: Mock,
        mock_run_query: Mock,
    ) -> None:
        for signal in (EOFError(), KeyboardInterrupt()):
            with self.subTest(signal=type(signal).__name__):
                with (
                    patch.object(sys, "argv", ["main.py"]),
                    patch("builtins.input", side_effect=signal),
                    redirect_stdout(StringIO()),
                ):
                    self.assertEqual(cli.main(), 0)

        mock_run_query.assert_not_called()

    def test_run_query_does_not_wrap_process_control_exceptions(self) -> None:
        graph = Mock()

        for process_signal in (KeyboardInterrupt(), SystemExit(2)):
            with self.subTest(signal=type(process_signal).__name__):
                graph.invoke.side_effect = process_signal
                with self.assertRaises(type(process_signal)):
                    cli.run_query(graph, "query")


if __name__ == "__main__":
    unittest.main()
