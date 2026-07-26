"""Offline tests for retrieval telemetry at the stable tool boundary."""

from __future__ import annotations

import unittest
from threading import Thread
from unittest.mock import patch

from src import config
from src.retrievers.base import ScoredChunk
from src.tools.retrieval import (
    consume_last_telemetry,
    search_knowledge_base,
)


class StaticRetriever:
    """Return deterministic scored chunks without provider access."""

    def __init__(self, results: list[ScoredChunk]) -> None:
        self._results = results

    def search(self, _query: str) -> list[ScoredChunk]:
        return list(self._results)


class FailingRetriever:
    """Simulate an infrastructure failure after a prior successful search."""

    def search(self, _query: str) -> list[ScoredChunk]:
        raise RuntimeError("provider unavailable")


class RetrievalTelemetryTests(unittest.TestCase):
    def tearDown(self) -> None:
        consume_last_telemetry()

    @staticmethod
    def _result(
        *,
        score: float,
        method: str,
        extras: dict[str, object],
    ) -> ScoredChunk:
        return ScoredChunk(
            index=0,
            chunk="--- Evidence Title ---\nByte-exact evidence.",
            score=score,
            method=method,
            extras=extras,
        )

    def test_tool_populates_safe_trace_for_every_mode(self) -> None:
        cases = (
            (
                "lexical",
                self._result(
                    score=4.25,
                    method="lexical",
                    extras={
                        "matched_terms": ("evidence", "title"),
                        "unreviewed_secret": "must-not-leak",
                    },
                ),
                "matched_terms=evidence, title",
            ),
            (
                "semantic",
                self._result(
                    score=0.8123,
                    method="semantic",
                    extras={
                        "cosine": 0.8123,
                        "unreviewed_secret": "must-not-leak",
                    },
                ),
                "cosine=0.8123",
            ),
            (
                "hybrid",
                self._result(
                    score=0.0325,
                    method="both",
                    extras={
                        "lexical_score": 4.25,
                        "semantic_score": 0.8123,
                        "unreviewed_secret": "must-not-leak",
                    },
                ),
                "rrf=0.0325, lexical=4.2500, semantic=0.8123",
            ),
        )

        for mode, scored, expected_detail in cases:
            with self.subTest(mode=mode):
                with (
                    patch.object(config, "SEARCH_MODE", mode),
                    patch(
                        "src.tools.retrieval.get_retriever",
                        return_value=StaticRetriever([scored]),
                    ),
                ):
                    snippets = search_knowledge_base.invoke(
                        {"query": "evidence title"}
                    )
                    telemetry = consume_last_telemetry()

                self.assertEqual(snippets, [scored.chunk])
                self.assertIsNotNone(telemetry)
                assert telemetry is not None
                self.assertEqual(telemetry.mode, mode)
                self.assertEqual(telemetry.empty_reason, None)
                self.assertEqual(len(telemetry.snippets), 1)
                self.assertEqual(
                    telemetry.snippets[0].title,
                    "Evidence Title",
                )
                self.assertEqual(telemetry.snippets[0].score, scored.score)
                self.assertEqual(
                    telemetry.snippets[0].method,
                    scored.method,
                )
                self.assertEqual(
                    telemetry.snippets[0].detail,
                    expected_detail,
                )
                self.assertNotIn(
                    "must-not-leak",
                    telemetry.snippets[0].detail,
                )
                self.assertIsNone(consume_last_telemetry())

    def test_lexical_empty_reason_distinguishes_tokens_from_gate(self) -> None:
        with (
            patch.object(config, "SEARCH_MODE", "lexical"),
            patch(
                "src.tools.retrieval.get_retriever",
                return_value=StaticRetriever([]),
            ),
        ):
            self.assertEqual(
                search_knowledge_base.invoke({"query": "นโยบาย"}),
                [],
            )
            no_terms = consume_last_telemetry()

            self.assertEqual(
                search_knowledge_base.invoke(
                    {"query": "unlisted compensation"}
                ),
                [],
            )
            gated = consume_last_telemetry()

        self.assertIsNotNone(no_terms)
        self.assertIsNotNone(gated)
        assert no_terms is not None and gated is not None
        self.assertEqual(no_terms.empty_reason, "no_query_terms")
        self.assertEqual(gated.empty_reason, "gated_out")

    def test_failed_search_clears_stale_telemetry(self) -> None:
        scored = self._result(
            score=1.0,
            method="lexical",
            extras={"matched_terms": ("evidence",)},
        )
        with (
            patch.object(config, "SEARCH_MODE", "lexical"),
            patch(
                "src.tools.retrieval.get_retriever",
                return_value=StaticRetriever([scored]),
            ),
        ):
            search_knowledge_base.invoke({"query": "evidence"})

        with (
            patch.object(config, "SEARCH_MODE", "semantic"),
            patch(
                "src.tools.retrieval.get_retriever",
                return_value=FailingRetriever(),
            ),
            self.assertRaisesRegex(RuntimeError, "provider unavailable"),
        ):
            search_knowledge_base.invoke({"query": "evidence"})

        self.assertIsNone(consume_last_telemetry())

    def test_thread_local_buffer_does_not_leak_to_parent_thread(self) -> None:
        scored = self._result(
            score=1.0,
            method="lexical",
            extras={"matched_terms": ("evidence",)},
        )

        observed: list[object] = []

        def search_in_worker() -> None:
            with (
                patch.object(config, "SEARCH_MODE", "lexical"),
                patch(
                    "src.tools.retrieval.get_retriever",
                    return_value=StaticRetriever([scored]),
                ),
            ):
                search_knowledge_base.invoke({"query": "evidence"})
                observed.append(consume_last_telemetry())

        worker = Thread(target=search_in_worker)
        worker.start()
        worker.join()

        self.assertEqual(len(observed), 1)
        self.assertIsNotNone(observed[0])
        self.assertIsNone(consume_last_telemetry())


if __name__ == "__main__":
    unittest.main()
