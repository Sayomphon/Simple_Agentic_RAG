"""Lazy, configuration-owned construction of retrieval strategies."""

from __future__ import annotations

from functools import lru_cache

from src import config
from src.retrievers.base import Retriever
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.lexical import LexicalRetriever
from src.retrievers.semantic import SemanticRetriever

SUPPORTED_SEARCH_MODES = frozenset({"lexical", "semantic", "hybrid"})


class UnsupportedSearchModeError(ValueError):
    """Raised when configuration names an unknown retrieval strategy."""


@lru_cache(maxsize=9)
def _build_retriever(
    mode: str,
    kb_path: str,
    embedding_model: str,
    min_cosine: float,
    rrf_k: int,
    cache_dir: str,
) -> Retriever:
    if mode == "lexical":
        return LexicalRetriever(kb_path)
    if mode == "semantic":
        return SemanticRetriever(
            kb_path,
            model_name=embedding_model,
            min_cosine=min_cosine,
            cache_dir=cache_dir,
        )
    if mode == "hybrid":
        lexical = _build_retriever(
            "lexical",
            kb_path,
            embedding_model,
            min_cosine,
            rrf_k,
            cache_dir,
        )
        semantic = _build_retriever(
            "semantic",
            kb_path,
            embedding_model,
            min_cosine,
            rrf_k,
            cache_dir,
        )
        return HybridRetriever(lexical, semantic, rrf_k=rrf_k)
    raise UnsupportedSearchModeError(
        f"Unsupported search mode {mode!r}; expected one of: "
        f"{', '.join(sorted(SUPPORTED_SEARCH_MODES))}"
    )


def get_retriever(mode: str | None = None) -> Retriever:
    """Return a cached retriever for an explicit or configured mode."""
    resolved_mode = (mode or config.SEARCH_MODE).strip().lower()
    return _build_retriever(
        resolved_mode,
        config.KB_PATH,
        config.EMBEDDING_MODEL_NAME,
        config.MIN_COSINE,
        config.RRF_K,
        config.EMBED_CACHE_DIR,
    )


def clear_retriever_cache() -> None:
    """Clear constructed strategies; tests use this after config patches."""
    _build_retriever.cache_clear()


__all__ = [
    "SUPPORTED_SEARCH_MODES",
    "UnsupportedSearchModeError",
    "clear_retriever_cache",
    "get_retriever",
]
