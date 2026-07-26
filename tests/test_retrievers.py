"""Offline tests for semantic, hybrid, cache, and factory behavior."""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import config
from src.retrievers.base import Retriever, ScoredChunk
from src.retrievers.factory import (
    UnsupportedSearchModeError,
    clear_retriever_cache,
    get_retriever,
)
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.lexical import LexicalRetriever
from src.retrievers.semantic import (
    EmbeddingCacheError,
    InvalidEmbeddingError,
    MissingEmbeddingCredentialsError,
    SemanticRetriever,
    cosine_similarity,
)


class FakeEmbedder:
    """Deterministic title/query vectors with transparent call counts."""

    def __init__(
        self,
        document_vectors: dict[str, list[float]],
        query_vectors: dict[str, list[float]],
    ) -> None:
        self.document_vectors = document_vectors
        self.query_vectors = query_vectors
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [
            list(self.document_vectors[text.partition("\n")[0]])
            for text in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return list(self.query_vectors[text])


class StaticRetriever(Retriever):
    def __init__(self, results: list[ScoredChunk]) -> None:
        self.results = results

    def search(self, query: str) -> list[ScoredChunk]:
        return list(self.results)


class SemanticRetrieverTests(unittest.TestCase):
    def _write_kb(self, directory: str) -> Path:
        path = Path(directory) / "knowledge.txt"
        path.write_text(
            "--- Alpha Policy ---\nAlpha evidence.\n\n"
            "--- Beta Policy ---\nBeta evidence.\n",
            encoding="utf-8",
        )
        return path

    def _embedder(self) -> FakeEmbedder:
        return FakeEmbedder(
            {
                "--- Alpha Policy ---": [1.0, 0.0],
                "--- Beta Policy ---": [0.0, 1.0],
            },
            {
                "alpha question": [1.0, 0.1],
                "beta question": [0.1, 1.0],
            },
        )

    def test_cosine_similarity_is_validated(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 1.0]), 2**-0.5)
        with self.assertRaises(InvalidEmbeddingError):
            cosine_similarity([1.0], [1.0, 0.0])
        with self.assertRaises(InvalidEmbeddingError):
            cosine_similarity([0.0, 0.0], [1.0, 0.0])

    def test_threshold_preserves_raw_source_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_kb(temp_dir)
            embedder = self._embedder()
            retriever = SemanticRetriever(
                path,
                model_name="fake-model",
                min_cosine=0.80,
                cache_dir=Path(temp_dir) / "cache",
                embedder=embedder,
            )

            results = retriever.search("alpha question")

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].index, 0)
            self.assertEqual(
                results[0].chunk,
                "--- Alpha Policy ---\nAlpha evidence.",
            )
            self.assertEqual(results[0].method, "semantic")
            self.assertGreaterEqual(results[0].score, 0.80)
            self.assertEqual(embedder.document_calls, 1)
            self.assertEqual(embedder.query_calls, 1)
            self.assertEqual(retriever.embedding_api_calls, 2)

    def test_disk_cache_reuses_document_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_kb(temp_dir)
            cache_dir = Path(temp_dir) / "cache"
            first_embedder = self._embedder()
            first = SemanticRetriever(
                path,
                model_name="fake-model",
                min_cosine=0.80,
                cache_dir=cache_dir,
                embedder=first_embedder,
            )
            first.search("alpha question")

            second_embedder = self._embedder()
            second = SemanticRetriever(
                path,
                model_name="fake-model",
                min_cosine=0.80,
                cache_dir=cache_dir,
                embedder=second_embedder,
            )
            second.search("beta question")

            self.assertEqual(first_embedder.document_calls, 1)
            self.assertEqual(second_embedder.document_calls, 0)
            self.assertEqual(second_embedder.query_calls, 1)
            cache_file = next(cache_dir.glob("*.json"))
            self.assertEqual(
                stat.S_IMODE(cache_file.stat().st_mode),
                0o600,
            )

    def test_kb_change_invalidates_document_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_kb(temp_dir)
            cache_dir = Path(temp_dir) / "cache"
            first_embedder = self._embedder()
            SemanticRetriever(
                path,
                model_name="fake-model",
                cache_dir=cache_dir,
                embedder=first_embedder,
            ).search("alpha question")

            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    "Alpha evidence.",
                    "Expanded alpha evidence.",
                ),
                encoding="utf-8",
            )
            second_embedder = self._embedder()
            SemanticRetriever(
                path,
                model_name="fake-model",
                cache_dir=cache_dir,
                embedder=second_embedder,
            ).search("alpha question")

            self.assertEqual(second_embedder.document_calls, 1)
            self.assertEqual(len(list(cache_dir.glob("*.json"))), 2)

    def test_model_change_invalidates_document_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_kb(temp_dir)
            cache_dir = Path(temp_dir) / "cache"
            SemanticRetriever(
                path,
                model_name="first-model",
                cache_dir=cache_dir,
                embedder=self._embedder(),
            ).search("alpha question")

            replacement = self._embedder()
            SemanticRetriever(
                path,
                model_name="second-model",
                cache_dir=cache_dir,
                embedder=replacement,
            ).search("alpha question")

            self.assertEqual(replacement.document_calls, 1)
            self.assertEqual(len(list(cache_dir.glob("*.json"))), 2)

    def test_corrupt_cache_is_rebuilt_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_kb(temp_dir)
            cache_dir = Path(temp_dir) / "cache"
            SemanticRetriever(
                path,
                model_name="fake-model",
                cache_dir=cache_dir,
                embedder=self._embedder(),
            ).search("alpha question")
            cache_file = next(cache_dir.glob("*.json"))
            cache_file.write_text("{broken", encoding="utf-8")

            replacement = self._embedder()
            results = SemanticRetriever(
                path,
                model_name="fake-model",
                cache_dir=cache_dir,
                embedder=replacement,
            ).search("alpha question")

            self.assertTrue(results)
            self.assertEqual(replacement.document_calls, 1)

    def test_symlinked_cache_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_kb(temp_dir)
            cache_dir = Path(temp_dir) / "cache"
            SemanticRetriever(
                path,
                model_name="fake-model",
                cache_dir=cache_dir,
                embedder=self._embedder(),
            ).search("alpha question")
            cache_file = next(cache_dir.glob("*.json"))
            target = Path(temp_dir) / "untrusted.json"
            target.write_text(cache_file.read_text(encoding="utf-8"))
            cache_file.unlink()
            cache_file.symlink_to(target)

            with self.assertRaisesRegex(
                EmbeddingCacheError,
                "symbolic links",
            ):
                SemanticRetriever(
                    path,
                    model_name="fake-model",
                    cache_dir=cache_dir,
                    embedder=self._embedder(),
                ).search("alpha question")

    def test_missing_key_fails_loud_in_api_backed_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_kb(temp_dir)
            retriever = SemanticRetriever(
                path,
                model_name="fake-model",
                cache_dir=Path(temp_dir) / "cache",
            )
            with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
                with self.assertRaisesRegex(
                    MissingEmbeddingCredentialsError,
                    "OPENAI_API_KEY",
                ):
                    retriever.search("alpha question")


class HybridRetrieverTests(unittest.TestCase):
    @staticmethod
    def _result(
        index: int,
        title: str,
        score: float,
        method: str,
    ) -> ScoredChunk:
        return ScoredChunk(
            index=index,
            chunk=f"--- {title} ---\nEvidence.",
            score=score,
            method=method,
        )

    def test_both_sides_empty_returns_empty(self) -> None:
        retriever = HybridRetriever(StaticRetriever([]), StaticRetriever([]))

        self.assertEqual(retriever.search("query"), [])

    def test_one_nonempty_side_preserves_its_rank_and_scores(self) -> None:
        semantic = [
            self._result(1, "Beta", 0.9, "semantic"),
            self._result(0, "Alpha", 0.8, "semantic"),
        ]
        retriever = HybridRetriever(StaticRetriever([]), StaticRetriever(semantic))

        self.assertEqual(retriever.search("query"), semantic)

    def test_rrf_deduplicates_and_rewards_agreement(self) -> None:
        alpha_lexical = self._result(0, "Alpha", 8.0, "lexical")
        beta_lexical = self._result(1, "Beta", 5.0, "lexical")
        alpha_semantic = self._result(0, "Alpha", 0.88, "semantic")
        gamma_semantic = self._result(2, "Gamma", 0.82, "semantic")
        retriever = HybridRetriever(
            StaticRetriever([alpha_lexical, beta_lexical]),
            StaticRetriever([alpha_semantic, gamma_semantic]),
            rrf_k=60,
        )

        results = retriever.search("query")

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].chunk, alpha_lexical.chunk)
        self.assertEqual(results[0].method, "both")
        self.assertEqual(
            len({result.chunk for result in results}),
            len(results),
        )


class RetrieverFactoryTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_retriever_cache()

    def test_default_lexical_factory_is_lazy_and_cached(self) -> None:
        with (
            patch.object(config, "SEARCH_MODE", "lexical"),
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}),
        ):
            clear_retriever_cache()
            first = get_retriever()
            second = get_retriever()

        self.assertIsInstance(first, LexicalRetriever)
        self.assertIs(first, second)

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            UnsupportedSearchModeError,
            "Unsupported search mode",
        ):
            get_retriever("reranker")


if __name__ == "__main__":
    unittest.main()
