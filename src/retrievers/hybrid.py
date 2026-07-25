"""Hybrid retrieval: BM25 + dense candidates merged by rank fusion.

Why Reciprocal Rank Fusion (RRF) instead of weighted score fusion:
    BM25 scores are unbounded above and grow with query length and term
    rarity, while cosine similarity lives in [-1, 1] — the two scales are
    incomparable, so weighted summing requires normalization first. On a
    corpus of a few dozen chunks, min-max normalization is unstable: the
    min/max are taken over each query's tiny candidate set, so one outlier
    candidate rescales every other score, and near-identical scores get
    stretched to opposite ends of [0, 1]. RRF sidesteps all of this by
    using only each result's *rank* within its own list — scale-free,
    parameter-light, and robust on small candidate sets. Weighted fusion
    remains available (``FUSION_METHOD=weighted``) precisely so evaluation
    can measure this trade-off instead of taking it on faith.

Relevance guardrail: fusion NEVER creates relevance. Each side retriever
applies its own gate (BM25's score/term gates, dense's ``MIN_COSINE``)
before fusion, and when both sides return nothing the hybrid returns [] —
it must not resurrect below-threshold candidates just to have output,
because unfounded snippets would invite the generator to hallucinate.
"""

from __future__ import annotations

from src.config import DENSE_WEIGHT, FUSION_METHOD, RRF_K
from src.retrievers.base import Retriever, ScoredChunk

# Fetch more than top_k from each side so fusion has real choices: a chunk
# ranked 5th by both retrievers can still beat one ranked 1st by only one.
_CANDIDATE_MULTIPLIER = 3


class HybridRetriever:
    """Composes a keyword and a dense retriever; owns only the fusion logic.

    Composition (not inheritance) keeps each side's index, gates, and
    failure handling self-contained — this class never sees raw scores'
    internals, only ranked, already-gated candidate lists.
    """

    SOURCE = "hybrid"

    def __init__(
        self,
        keyword: Retriever,
        dense: Retriever,
        rrf_k: int = RRF_K,
        fusion_method: str = FUSION_METHOD,
        dense_weight: float = DENSE_WEIGHT,
    ) -> None:
        """Wrap two already-built retrievers.

        Args:
            keyword: Lexical side (BM25), with its own relevance gates.
            dense: Semantic side, with its own ``MIN_COSINE`` gate.
            rrf_k: RRF dampening constant; 60 is the standard from the
                original RRF paper and needs no per-corpus tuning.
            fusion_method: ``"rrf"`` (default) or ``"weighted"``.
            dense_weight: Weight of the dense side under ``"weighted"``;
                the keyword side gets ``1 - dense_weight``.
        """
        if fusion_method not in ("rrf", "weighted"):
            raise ValueError(
                f"Unknown FUSION_METHOD {fusion_method!r}; "
                "expected 'rrf' or 'weighted'."
            )
        self._keyword = keyword
        self._dense = dense
        self._rrf_k = rrf_k
        self._fusion_method = fusion_method
        self._dense_weight = min(max(0.0, dense_weight), 1.0)

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Fuse gated candidates from both sides; [] when both sides say []."""
        if top_k <= 0:
            return []
        pool_size = top_k * _CANDIDATE_MULTIPLIER
        candidate_lists = [
            self._keyword.search(query, pool_size),
            self._dense.search(query, pool_size),
        ]
        if not any(candidate_lists):
            return []  # both gates said "not relevant" — do not invent results

        if self._fusion_method == "rrf":
            fused = self._fuse_rrf(candidate_lists)
        else:
            fused = self._fuse_weighted(candidate_lists)
        fused.sort(key=lambda hit: hit.score, reverse=True)
        return fused[:top_k]

    def _fuse_rrf(
        self, candidate_lists: list[list[ScoredChunk]]
    ) -> list[ScoredChunk]:
        """score(c) = sum over lists of 1 / (rrf_k + rank_in_list(c))."""
        fused_scores: dict[int, float] = {}
        merged: dict[int, ScoredChunk] = {}
        for hits in candidate_lists:
            for rank, hit in enumerate(hits, start=1):
                key = hit.chunk.index
                fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (
                    self._rrf_k + rank
                )
                merged[key] = self._merge_sources(merged.get(key), hit)
        return [
            ScoredChunk(chunk=merged[k].chunk, score=s, source=merged[k].source)
            for k, s in fused_scores.items()
        ]

    def _fuse_weighted(
        self, candidate_lists: list[list[ScoredChunk]]
    ) -> list[ScoredChunk]:
        """Min-max normalize each list to [0, 1], then weighted-sum.

        Kept for evaluation comparison; see the module docstring for why
        this is NOT the default on a small corpus.
        """
        weights = [1.0 - self._dense_weight, self._dense_weight]
        fused_scores: dict[int, float] = {}
        merged: dict[int, ScoredChunk] = {}
        for weight, hits in zip(weights, candidate_lists, strict=True):
            if not hits:
                continue
            low = min(hit.score for hit in hits)
            high = max(hit.score for hit in hits)
            span = high - low
            for hit in hits:
                # A single-candidate (or constant-score) list normalizes to
                # 1.0 — the list ranked it best, so it gets full weight.
                normalized = (hit.score - low) / span if span else 1.0
                key = hit.chunk.index
                fused_scores[key] = fused_scores.get(key, 0.0) + weight * normalized
                merged[key] = self._merge_sources(merged.get(key), hit)
        return [
            ScoredChunk(chunk=merged[k].chunk, score=s, source=merged[k].source)
            for k, s in fused_scores.items()
        ]

    @staticmethod
    def _merge_sources(existing: ScoredChunk | None, new: ScoredChunk) -> ScoredChunk:
        """Track provenance: a chunk found by both sides becomes "bm25+dense"."""
        if existing is None or existing.source == new.source:
            return new
        sources = "+".join(sorted({existing.source, new.source}))
        return ScoredChunk(chunk=new.chunk, score=new.score, source=sources)
