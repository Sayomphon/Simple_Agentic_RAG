"""Threshold-first reciprocal-rank fusion of lexical and semantic results."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import RRF_K
from src.retrievers.base import Retriever, ScoredChunk


@dataclass
class _FusionEntry:
    chunk: str
    index: int
    score: float = 0.0
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    lexical_score: float | None = None
    semantic_score: float | None = None

    @property
    def best_rank(self) -> int:
        ranks = [
            rank
            for rank in (self.lexical_rank, self.semantic_rank)
            if rank is not None
        ]
        return min(ranks)

    @property
    def method(self) -> str:
        if self.lexical_rank is not None and self.semantic_rank is not None:
            return "both"
        return "lexical" if self.lexical_rank is not None else "semantic"


class HybridRetriever:
    """Fuse independently gated result sets without comparing score scales."""

    def __init__(
        self,
        lexical: Retriever,
        semantic: Retriever,
        *,
        rrf_k: int = RRF_K,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        self._lexical = lexical
        self._semantic = semantic
        self._rrf_k = rrf_k

    @property
    def embedding_api_calls(self) -> int:
        return int(getattr(self._semantic, "embedding_api_calls", 0))

    def search(self, query: str) -> list[ScoredChunk]:
        """Gate each source first, then fuse ranks and deduplicate raw chunks."""
        lexical_results = self._lexical.search(query)
        semantic_results = self._semantic.search(query)
        if not lexical_results:
            return list(semantic_results)
        if not semantic_results:
            return list(lexical_results)

        entries: dict[str, _FusionEntry] = {}
        for rank, result in enumerate(lexical_results, start=1):
            entry = entries.setdefault(
                result.chunk,
                _FusionEntry(chunk=result.chunk, index=result.index),
            )
            if entry.lexical_rank is not None:
                continue
            entry.lexical_rank = rank
            entry.lexical_score = result.score
            entry.score += 1.0 / (self._rrf_k + rank)

        for rank, result in enumerate(semantic_results, start=1):
            entry = entries.setdefault(
                result.chunk,
                _FusionEntry(chunk=result.chunk, index=result.index),
            )
            if entry.semantic_rank is not None:
                continue
            entry.semantic_rank = rank
            entry.semantic_score = result.score
            entry.score += 1.0 / (self._rrf_k + rank)

        fused = [
            ScoredChunk(
                index=entry.index,
                chunk=entry.chunk,
                score=entry.score,
                method=entry.method,
                extras={
                    "lexical_rank": entry.lexical_rank,
                    "semantic_rank": entry.semantic_rank,
                    "lexical_score": entry.lexical_score,
                    "semantic_score": entry.semantic_score,
                    "rrf_k": self._rrf_k,
                },
            )
            for entry in entries.values()
        ]
        fused.sort(
            key=lambda result: (
                -result.score,
                entries[result.chunk].best_rank,
                result.index,
            )
        )
        return fused


__all__ = ["HybridRetriever"]
