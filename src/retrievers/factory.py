"""Retriever selection: one place that maps a search mode to an instance.

Build-once guarantee: every mode is built at most once per process (an
``lru_cache`` per mode), and all modes share the SAME chunk list and one
BM25 index — hybrid never builds the lexical side twice.

Laziness guarantee: ``src.retrievers.dense`` is imported inside the factory
branch, so keyword-only processes never import the OpenAI client, read the
embedding cache, or touch the network.

Failure policy: if the embedding index cannot be built (missing key, API
outage) the factory logs a clear warning and returns the keyword retriever
instead — retrieval degrades, the pipeline never crashes.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from src.config import SEARCH_MODE
from src.retrievers.base import Chunk, Retriever, load_chunks
from src.retrievers.keyword import BM25Retriever

logger = logging.getLogger(__name__)

_KNOWN_MODES = ("keyword", "semantic", "hybrid")


def get_retriever(mode: str | None = None) -> Retriever:
    """Return the cached retriever for ``mode``.

    Args:
        mode: ``"keyword"``, ``"semantic"``, or ``"hybrid"``. ``None`` (the
            default) keeps the original behaviour and uses the process-wide
            ``config.SEARCH_MODE`` — existing callers are unaffected. An
            explicit mode lets per-request callers (e.g. the Streamlit UI)
            switch strategies live without re-importing config.

    Returns:
        The retriever instance for ``mode``, built at most once per process.

    Raises:
        ValueError: If ``mode`` names an unknown implementation.
    """
    return _build_retriever(mode if mode is not None else SEARCH_MODE)


@lru_cache(maxsize=1)
def _shared_chunks() -> tuple[Chunk, ...]:
    """One chunk list per process, shared by every index."""
    return tuple(load_chunks())


@lru_cache(maxsize=1)
def _keyword_retriever() -> BM25Retriever:
    """One shared BM25 index: cheap to build and serving three roles —
    the keyword mode itself, the lexical side of hybrid, and the fallback
    when the embeddings API is unavailable."""
    return BM25Retriever(list(_shared_chunks()))


@lru_cache(maxsize=None)
def _build_retriever(mode: str) -> Retriever:
    """Build the retriever for one explicit mode (cached per mode).

    The ``lru_cache`` makes each mode a lazy singleton: the file read,
    chunking, tokenization, and index builds all happen exactly once per
    process and mode.
    """
    if mode not in _KNOWN_MODES:
        raise ValueError(
            f"Unknown search mode {mode!r}; expected one of "
            f"{_KNOWN_MODES}. (Add new modes here — the tool and agents "
            "need no changes.)"
        )
    keyword = _keyword_retriever()
    if mode == "keyword":
        return keyword

    from src.retrievers.dense import EmbeddingIndexError, OpenAIEmbeddingRetriever
    from src.retrievers.hybrid import HybridRetriever

    try:
        dense = OpenAIEmbeddingRetriever(
            list(_shared_chunks()),
            # In hybrid, BM25 already runs alongside dense — a dense-side
            # fallback would just duplicate its evidence during fusion.
            fallback=keyword if mode == "semantic" else None,
        )
    except EmbeddingIndexError as exc:
        logger.warning("%s Falling back to keyword mode.", exc)
        return keyword

    if mode == "semantic":
        return dense
    return HybridRetriever(keyword, dense)
