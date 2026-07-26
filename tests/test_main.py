"""Offline tests for CLI rendering and application error boundaries."""

from __future__ import annotations

import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import main as cli
from src.agents.reporter import NOT_FOUND_SENTENCE
from src.retrievers.base import SearchTelemetry, SnippetTrace


def _stream_events(
    snippets: list[str],
    report: str,
    chunks: tuple[str, ...] = (),
    telemetry: tuple[SearchTelemetry, ...] = (),
) -> list[tuple[str, object]]:
    """Build the (mode, payload) events graph.stream would yield."""
    retriever_update: dict[str, object] = {"snippets": snippets}
    if telemetry:
        retriever_update["retrieval_telemetry"] = list(telemetry)
    events: list[tuple[str, object]] = [
        ("updates", {"data_retriever": retriever_update})
    ]
    events.extend(
        (
            "messages",
            (
                SimpleNamespace(text=chunk),
                {"langgraph_node": "report_generator"},
            ),
        )
        for chunk in chunks
    )
    events.append(("updates", {"report_generator": {"report": report}}))
    return events


class MainTests(unittest.TestCase):
    def test_run_query_wraps_graph_error_and_preserves_cause(self) -> None:
        graph = Mock()
        original_error = RuntimeError("provider failure")
        graph.stream.side_effect = original_error

        with (
            self.assertRaises(cli.QueryExecutionError) as context,
            redirect_stdout(StringIO()),
        ):
            cli.run_query(graph, "sensitive user query")

        self.assertIs(context.exception.__cause__, original_error)
        self.assertNotIn("sensitive user query", str(context.exception))

    def test_successful_run_query_renders_and_returns_result(self) -> None:
        graph = Mock()
        graph.stream.return_value = iter(
            _stream_events(
                snippets=["--- Travel ---\nGrounded evidence."],
                report="Grounded answer.",
                chunks=("Grounded ", "answer."),
            )
        )
        output = StringIO()

        with redirect_stdout(output):
            result = cli.run_query(graph, "international travel")

        self.assertEqual(result["report"], "Grounded answer.")
        self.assertEqual(
            result["snippets"], ["--- Travel ---\nGrounded evidence."]
        )
        self.assertIn("--- Travel ---\nGrounded evidence.", output.getvalue())
        self.assertIn("Grounded answer.", output.getvalue())

    def test_streamed_answer_matches_final_report_byte_for_byte(self) -> None:
        # Chunks carry the leading/trailing whitespace that the generator
        # strips from the state report; the screen must show the report text.
        graph = Mock()
        graph.stream.return_value = iter(
            _stream_events(
                snippets=["--- Travel ---\nGrounded evidence."],
                report="Grounded answer.",
                chunks=("  Grounded ", "answer.", "  "),
            )
        )
        output = StringIO()

        with redirect_stdout(output):
            result = cli.run_query(graph, "international travel")

        rendered_answer = (
            output.getvalue().split("[3] FINAL ANSWER\n", 1)[1].rsplit(
                cli.BANNER, 1
            )[0]
        )
        self.assertEqual(rendered_answer.strip("\n"), result["report"])

    def test_not_found_path_renders_without_token_stream(self) -> None:
        graph = Mock()
        graph.stream.return_value = iter(
            _stream_events(snippets=[], report=NOT_FOUND_SENTENCE)
        )
        output = StringIO()

        with redirect_stdout(output):
            result = cli.run_query(graph, "What is the CEO's salary?")

        self.assertEqual(result["report"], NOT_FOUND_SENTENCE)
        self.assertIn("(none)", output.getvalue())
        self.assertIn(NOT_FOUND_SENTENCE, output.getvalue())

    def test_retrieval_telemetry_is_rendered_and_returned(self) -> None:
        graph = Mock()
        telemetry = SearchTelemetry(
            mode="semantic",
            query="international travel",
            latency_ms=12.34,
            empty_reason=None,
            snippets=(
                SnippetTrace(
                    title="Travel",
                    score=0.8765,
                    method="semantic",
                    detail="cosine=0.8765",
                ),
            ),
        )
        graph.stream.return_value = iter(
            _stream_events(
                snippets=["--- Travel ---\nGrounded evidence."],
                report="Grounded answer.",
                telemetry=(telemetry,),
            )
        )
        output = StringIO()

        with redirect_stdout(output):
            result = cli.run_query(graph, "international travel")

        rendered = output.getvalue()
        self.assertIn(
            "#1 [Travel] score=0.8765 method=semantic",
            rendered,
        )
        self.assertIn(
            "mode=semantic attempts=1 retrieval=12.34ms",
            rendered,
        )
        self.assertEqual(
            result["retrieval_telemetry"],
            [telemetry],
        )

    def test_no_query_terms_reason_is_rendered(self) -> None:
        graph = Mock()
        telemetry = SearchTelemetry(
            mode="lexical",
            query="นโยบาย",
            latency_ms=0.02,
            empty_reason="no_query_terms",
            snippets=(),
        )
        graph.stream.return_value = iter(
            _stream_events(
                snippets=[],
                report=NOT_FOUND_SENTENCE,
                telemetry=(telemetry,),
            )
        )
        output = StringIO()

        with redirect_stdout(output):
            cli.run_query(graph, "นโยบาย")

        self.assertIn(
            "query produced no searchable terms (lexical mode)",
            output.getvalue(),
        )
    def test_divergent_stream_preview_still_ends_with_the_report(self) -> None:
        graph = Mock()
        graph.stream.return_value = iter(
            _stream_events(
                snippets=["--- Travel ---\nGrounded evidence."],
                report="Authoritative answer.",
                chunks=("Different preview",),
            )
        )
        output = StringIO()

        with redirect_stdout(output):
            result = cli.run_query(graph, "international travel")

        self.assertEqual(result["report"], "Authoritative answer.")
        self.assertIn("Authoritative answer.", output.getvalue())

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
        mock_build_graph.return_value.stream.side_effect = RuntimeError(
            f"{raw_query}: {secret}"
        )
        stderr = StringIO()

        with (
            patch.object(sys, "argv", ["main.py", raw_query]),
            redirect_stdout(StringIO()),
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
                graph.stream.side_effect = process_signal
                with (
                    self.assertRaises(type(process_signal)),
                    redirect_stdout(StringIO()),
                ):
                    cli.run_query(graph, "query")


if __name__ == "__main__":
    unittest.main()
