"""Basic semantic retrieval using embeddings, cosine, and a fixed gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from threading import RLock
from typing import Protocol

from langchain_openai import OpenAIEmbeddings

from src.config import (
    EMBED_CACHE_DIR,
    EMBEDDING_MODEL_NAME,
    KB_PATH,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    MIN_COSINE,
)
from src.retrievers.base import ScoredChunk
from src.retrievers.lexical import load_knowledge_base

_CACHE_SCHEMA_VERSION = 1
_MAX_CACHE_BYTES = 64 * 1024 * 1024


class EmbeddingProvider(Protocol):
    """Provider surface used by the retriever and deterministic fake tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed source documents in one batch."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one query."""


class SemanticRetrievalError(RuntimeError):
    """Raised when semantic infrastructure cannot return trustworthy results."""


class MissingEmbeddingCredentialsError(SemanticRetrievalError):
    """Raised when an API-backed mode is selected without credentials."""


class EmbeddingCacheError(SemanticRetrievalError):
    """Raised when the configured cache cannot be used safely."""


class InvalidEmbeddingError(SemanticRetrievalError):
    """Raised when a provider returns malformed or incompatible vectors."""


def cosine_similarity(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    """Return cosine similarity after validating dimensions and norms."""
    if not first or len(first) != len(second):
        raise InvalidEmbeddingError(
            "Embedding vectors must be non-empty and have equal dimensions"
        )
    if not all(math.isfinite(value) for value in chain(first, second)):
        raise InvalidEmbeddingError("Embedding vectors must contain finite values")

    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0.0 or second_norm == 0.0:
        raise InvalidEmbeddingError("Embedding vectors must have non-zero norms")
    return sum(a * b for a, b in zip(first, second, strict=True)) / (
        first_norm * second_norm
    )


def _content_digest(chunks: Sequence[str]) -> str:
    """Hash exactly the raw chunks sent to the embedding provider."""
    digest = hashlib.sha256()
    for chunk in chunks:
        encoded = chunk.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def _cache_key(model_name: str, content_digest: str) -> str:
    payload = (
        f"schema={_CACHE_SCHEMA_VERSION}\0"
        f"model={model_name}\0content={content_digest}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validated_vectors(
    raw_vectors: object,
    *,
    expected_count: int,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(raw_vectors, list) or len(raw_vectors) != expected_count:
        raise InvalidEmbeddingError(
            "Embedding count does not match the knowledge-base section count"
        )

    vectors: list[tuple[float, ...]] = []
    dimension: int | None = None
    for raw_vector in raw_vectors:
        if not isinstance(raw_vector, list) or not raw_vector:
            raise InvalidEmbeddingError(
                "Every embedding must be a non-empty numeric list"
            )
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in raw_vector
        ):
            raise InvalidEmbeddingError(
                "Every embedding value must be a finite number"
            )
        vector = tuple(float(value) for value in raw_vector)
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise InvalidEmbeddingError(
                "Document embeddings must have one consistent dimension"
            )
        if not any(vector):
            raise InvalidEmbeddingError(
                "Document embeddings must have non-zero norms"
            )
        vectors.append(vector)
    return tuple(vectors)


class SemanticRetriever:
    """Linear-scan semantic retriever for the small local text corpus."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        model_name: str = EMBEDDING_MODEL_NAME,
        min_cosine: float = MIN_COSINE,
        cache_dir: str | Path = EMBED_CACHE_DIR,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")
        if not -1.0 <= min_cosine <= 1.0:
            raise ValueError("min_cosine must be between -1.0 and 1.0")

        self._path = Path(path if path is not None else KB_PATH)
        self._model_name = model_name
        self._min_cosine = min_cosine
        self._cache_dir = Path(cache_dir)
        self._embedder = embedder
        self._document_signature: str | None = None
        self._document_vectors: tuple[tuple[float, ...], ...] | None = None
        self._document_embedding_calls = 0
        self._query_embedding_calls = 0
        self._document_lock = RLock()

    @property
    def embedding_api_calls(self) -> int:
        """Number of provider method calls made by this instance."""
        return self._document_embedding_calls + self._query_embedding_calls

    @property
    def min_cosine(self) -> float:
        return self._min_cosine

    def _get_embedder(self) -> EmbeddingProvider:
        if self._embedder is not None:
            return self._embedder
        if not (os.getenv("OPENAI_API_KEY") or "").strip():
            raise MissingEmbeddingCredentialsError(
                "OPENAI_API_KEY is required for semantic or hybrid search"
            )
        try:
            self._embedder = OpenAIEmbeddings(
                model=self._model_name,
                timeout=LLM_TIMEOUT_SECONDS,
                max_retries=LLM_MAX_RETRIES,
            )
        except Exception as exc:
            raise SemanticRetrievalError(
                "Could not initialize the embedding provider"
            ) from exc
        return self._embedder

    def _cache_path(self, content_digest: str) -> Path:
        return self._cache_dir / f"{_cache_key(self._model_name, content_digest)}.json"

    def _read_cache(
        self,
        path: Path,
        *,
        content_digest: str,
        expected_count: int,
    ) -> tuple[tuple[float, ...], ...] | None:
        try:
            if path.is_symlink():
                raise EmbeddingCacheError(
                    "Embedding cache files must not be symbolic links"
                )
            if path.stat().st_size > _MAX_CACHE_BYTES:
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        except OSError as exc:
            raise EmbeddingCacheError(
                f"Could not read embedding cache: {path}"
            ) from exc

        if not isinstance(payload, dict):
            return None
        if (
            payload.get("schema_version") != _CACHE_SCHEMA_VERSION
            or payload.get("model") != self._model_name
            or payload.get("content_digest") != content_digest
        ):
            return None
        try:
            return _validated_vectors(
                payload.get("vectors"),
                expected_count=expected_count,
            )
        except InvalidEmbeddingError:
            return None

    def _write_cache(
        self,
        path: Path,
        *,
        content_digest: str,
        vectors: tuple[tuple[float, ...], ...],
    ) -> None:
        if self._cache_dir.exists() and self._cache_dir.is_symlink():
            raise EmbeddingCacheError(
                "Embedding cache directory must not be a symbolic link"
            )
        temporary_path: Path | None = None
        try:
            self._cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            payload = {
                "schema_version": _CACHE_SCHEMA_VERSION,
                "model": self._model_name,
                "content_digest": content_digest,
                "vectors": vectors,
            }
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._cache_dir,
                prefix=f".{path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(payload, temporary, separators=(",", ":"))
                temporary.flush()
                os.fsync(temporary.fileno())
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, path)
        except OSError as exc:
            raise EmbeddingCacheError(
                f"Could not write embedding cache: {path}"
            ) from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _embed_documents(
        self,
        chunks: list[str],
        content_digest: str,
    ) -> tuple[tuple[float, ...], ...]:
        cache_path = self._cache_path(content_digest)
        cached = self._read_cache(
            cache_path,
            content_digest=content_digest,
            expected_count=len(chunks),
        )
        if cached is not None:
            return cached

        try:
            embedder = self._get_embedder()
            self._document_embedding_calls += 1
            raw_vectors = embedder.embed_documents(chunks)
        except SemanticRetrievalError:
            raise
        except Exception as exc:
            raise SemanticRetrievalError(
                "The embedding provider could not embed the knowledge base"
            ) from exc
        vectors = _validated_vectors(raw_vectors, expected_count=len(chunks))
        self._write_cache(
            cache_path,
            content_digest=content_digest,
            vectors=vectors,
        )
        return vectors

    def _document_snapshot(
        self,
    ) -> tuple[list[str], tuple[tuple[float, ...], ...]]:
        chunks = load_knowledge_base(self._path)
        signature = _content_digest(chunks)
        with self._document_lock:
            if (
                signature != self._document_signature
                or self._document_vectors is None
            ):
                self._document_vectors = self._embed_documents(chunks, signature)
                self._document_signature = signature
            return chunks, self._document_vectors

    def score_all(self, query: str) -> list[ScoredChunk]:
        """Score every section; evaluation uses this before thresholding."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            return []

        chunks, document_vectors = self._document_snapshot()
        try:
            embedder = self._get_embedder()
            self._query_embedding_calls += 1
            raw_query_vector = embedder.embed_query(query)
        except SemanticRetrievalError:
            raise
        except Exception as exc:
            raise SemanticRetrievalError(
                "The embedding provider could not embed the query"
            ) from exc
        query_vectors = _validated_vectors(
            [raw_query_vector],
            expected_count=1,
        )
        query_vector = query_vectors[0]

        results = []
        for index, (chunk, document_vector) in enumerate(
            zip(chunks, document_vectors, strict=True)
        ):
            score = cosine_similarity(query_vector, document_vector)
            results.append(
                ScoredChunk(
                    index=index,
                    chunk=chunk,
                    score=score,
                    method="semantic",
                    extras={"cosine": score},
                )
            )
        results.sort(key=lambda result: (-result.score, result.index))
        return results

    def search(self, query: str) -> list[ScoredChunk]:
        """Return only sections meeting the calibrated cosine threshold."""
        return [
            result
            for result in self.score_all(query)
            if result.score >= self._min_cosine
        ]


__all__ = [
    "EmbeddingCacheError",
    "EmbeddingProvider",
    "InvalidEmbeddingError",
    "MissingEmbeddingCredentialsError",
    "SemanticRetrievalError",
    "SemanticRetriever",
    "cosine_similarity",
]
