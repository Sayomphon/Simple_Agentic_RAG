"""Dense retrieval over OpenAI embeddings with a content-addressed disk cache.

Cost/latency model:
    - The corpus is embedded in ONE batched API call at index-build time,
      then cached to ``.cache/`` keyed by a hash of the chunk contents.
      Unchanged KB -> cache hit -> zero API calls to build the index.
    - Vectors are L2-normalized before caching, so query-time similarity is
      a single matrix–vector dot product (cosine == dot on unit vectors).
    - Query embeddings cost one API call each, memoized with an LRU cache.

Failure model: a build-time API failure raises ``EmbeddingIndexError`` for
the factory to catch (it falls back to keyword mode); a query-time API
failure logs a warning and delegates to the optional ``fallback`` retriever
instead of crashing the pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from openai import OpenAI, OpenAIError

from src.config import EMBEDDING_CACHE_DIR, EMBEDDING_MODEL, MIN_COSINE
from src.retrievers.base import Chunk, Retriever, ScoredChunk

logger = logging.getLogger(__name__)

_HAS_SEARCHABLE_TEXT = re.compile(r"[a-zA-Z0-9฀-๿]")  # ASCII or Thai


class EmbeddingIndexError(RuntimeError):
    """Raised when the embedding index cannot be built (no key, API down)."""


class OpenAIEmbeddingRetriever:
    """Semantic retriever: cosine similarity over ``text-embedding-3-small``.

    Complements BM25 on vocabulary-mismatch queries ("quit my job" vs the
    handbook's "resignation") but, unlike BM25, cosine similarity is never
    zero — so ``min_cosine`` is the relevance gate that keeps unanswerable
    queries returning [] instead of the least-unrelated chunk.
    """

    SOURCE = "dense"

    def __init__(
        self,
        chunks: list[Chunk],
        min_cosine: float = MIN_COSINE,
        model: str = EMBEDDING_MODEL,
        cache_dir: str = EMBEDDING_CACHE_DIR,
        fallback: Retriever | None = None,
    ) -> None:
        """Build (or load from disk cache) the embedding index.

        Args:
            chunks: Corpus to index.
            min_cosine: Cosine similarity a chunk must reach to be returned.
            model: OpenAI embedding model name.
            cache_dir: Directory for the ``.npz`` vector cache (git-ignored).
            fallback: Retriever used when a query-time API call fails.
                Leave ``None`` inside ``HybridRetriever`` — there the BM25
                side already covers the failure, and a fallback would count
                the same BM25 evidence twice during fusion.

        Raises:
            EmbeddingIndexError: If no cache exists and the API call fails.
        """
        self._chunks = chunks
        self._min_cosine = min_cosine
        self._model = model
        self._fallback = fallback
        # One client per retriever: reuses the HTTPS connection across
        # query embeddings — a fresh client per call would pay the TLS
        # handshake (~hundreds of ms) on every single query.
        self._client = OpenAI()
        self.last_index_tokens: int | None = None  # filled on a cache miss

        # Content-addressed cache key: any change to the KB text (or the
        # embedding model) changes the hash, forcing an automatic rebuild.
        fingerprint = hashlib.sha256(
            "\x00".join([model, *(c.as_snippet() for c in chunks)]).encode("utf-8")
        ).hexdigest()[:16]
        self._cache_path = Path(cache_dir) / f"embeddings-{fingerprint}.npz"

        self._matrix = self._load_or_build_index()
        # Bound per-instance so the memo dies with the retriever, and the
        # cache key is simply the query string.
        self._embed_query = lru_cache(maxsize=256)(self._embed_query_uncached)

    def _load_or_build_index(self) -> np.ndarray:
        """Return the (n_chunks, dim) unit-normalized embedding matrix."""
        if self._cache_path.is_file():
            matrix = np.load(self._cache_path)["embeddings"]
            if matrix.shape[0] == len(self._chunks):
                logger.info("Embedding cache hit: %s", self._cache_path)
                return matrix
            # A hash collision this shape mismatch implies is practically
            # impossible; treat defensively and rebuild.
            logger.warning("Embedding cache shape mismatch; rebuilding index.")

        texts = [chunk.as_snippet() for chunk in self._chunks]
        try:
            # ONE batched call for the whole corpus — never per chunk.
            response = self._client.embeddings.create(model=self._model, input=texts)
        except OpenAIError as exc:
            raise EmbeddingIndexError(
                f"Could not build the embedding index with model "
                f"'{self._model}': {exc}. Check OPENAI_API_KEY and network "
                "access, or set SEARCH_MODE=keyword."
            ) from exc

        matrix = np.array([item.embedding for item in response.data], dtype=np.float32)
        # Normalize once at build time so search is a plain dot product.
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        self.last_index_tokens = response.usage.total_tokens

        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._cache_path, embeddings=matrix)
        logger.info(
            "Embedded %d chunks in one call (%s tokens); cached to %s",
            len(texts),
            self.last_index_tokens,
            self._cache_path,
        )
        return matrix

    def _embed_query_uncached(self, query: str) -> np.ndarray:
        response = self._client.embeddings.create(model=self._model, input=[query])
        vector = np.array(response.data[0].embedding, dtype=np.float32)
        return vector / np.linalg.norm(vector)

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Return up to ``top_k`` chunks with cosine >= ``min_cosine``.

        Empty or symbol-only queries return [] without an API call. On a
        query-time API failure the configured ``fallback`` answers instead
        (its hits keep their own ``source`` label, making the degradation
        visible to evaluation); without a fallback the failure yields [].
        """
        if top_k <= 0 or not _HAS_SEARCHABLE_TEXT.search(query):
            return []
        try:
            query_vector = self._embed_query(query)
        except OpenAIError as exc:
            logger.warning(
                "Query embedding failed (%s); falling back to %s.",
                exc,
                type(self._fallback).__name__ if self._fallback else "no results",
            )
            return self._fallback.search(query, top_k) if self._fallback else []

        similarities = self._matrix @ query_vector  # cosine, via unit vectors
        order = np.argsort(similarities)[::-1][:top_k]
        return [
            ScoredChunk(
                chunk=self._chunks[i],
                score=float(similarities[i]),
                source=self.SOURCE,
            )
            for i in order
            if similarities[i] >= self._min_cosine
        ]
