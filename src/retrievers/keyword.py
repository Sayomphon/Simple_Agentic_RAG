"""Phase 1 keyword retrieval: title-aware BM25 with relevance gates.

Also owns the deterministic lexical query normalization (stop words, light
stemming, aliases) and ``has_english_search_terms``, which the Retriever
Agent uses to decide whether the user's own wording is searchable.
"""

from __future__ import annotations

import heapq
import re

from rank_bm25 import BM25Okapi

from src.config import (
    MIN_MATCHED_TERMS,
    MIN_RELATIVE_SCORE,
    MIN_SCORE,
    TITLE_BOOST,
)
from src.retrievers.base import Chunk, ScoredChunk

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_QUERY_ALIASES = (
    (re.compile(r"\b(?:work|working)\s+from\s+home\b"), "remote work"),
    (re.compile(r"\bwfh\b"), "remote work"),
    (re.compile(r"\bcyber[\s-]*security\b"), "security"),
    (re.compile(r"\bvacation\b"), "annual leave"),
)

# Search-intent words add little evidence that a document is relevant. Keeping
# them in a small, local set avoids a heavyweight NLP dependency and makes the
# retrieval behaviour deterministic and easy to audit.
_STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "company",
        "could",
        "detail",
        "do",
        "does",
        "employee",
        "for",
        "from",
        "guideline",
        "have",
        "how",
        "i",
        "if",
        "in",
        "information",
        "is",
        "it",
        "may",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "please",
        "policy",
        "rule",
        # Possessive fragment: "CEO's" tokenizes to "ceo", "s". Indexed from
        # 26/54 KB sections ("company's", "month's", ...), the stray "s"
        # carries no content yet counts as a distinct matched term, letting
        # single-incidental-word chunks slip past MIN_MATCHED_TERMS.
        "s",
        "should",
        "tell",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)


def _light_stem(token: str) -> str:
    """Normalize common English suffixes without an external NLP package."""
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("sses") and len(token) > 5:
        return token[:-2]
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    """Normalize text into content-bearing English search terms."""
    tokens = (_light_stem(token) for token in _TOKEN_PATTERN.findall(text.lower()))
    return [token for token in tokens if token and token not in _STOP_WORDS]


def _normalize_query(query: str) -> list[str]:
    """Apply deterministic aliases before tokenization.

    Aliases cover common English vocabulary that differs from an HR handbook;
    the Retriever Agent handles translation only when no English search terms
    are available.
    """
    normalized = query.lower()
    for pattern, replacement in _QUERY_ALIASES:
        normalized = pattern.sub(replacement, normalized)
    return _tokenize(normalized)


def has_english_search_terms(query: str) -> bool:
    """Return whether deterministic lexical retrieval can search the query."""
    return bool(_normalize_query(query))


class BM25Retriever:
    """Title-aware BM25 retriever with a matched-term relevance gate.

    Scale note: for the current ~50-chunk KB an in-memory index with a
    linear scan per query is ideal. At ~100k chunks this design would
    change: batch-embed the corpus offline, store vectors in an external
    store (e.g. pgvector/Qdrant), and query an approximate-nearest-
    neighbour index instead of scoring every chunk.
    """

    SOURCE = "bm25"

    def __init__(
        self,
        chunks: list[Chunk],
        min_score: float = MIN_SCORE,
        min_matched_terms: int = MIN_MATCHED_TERMS,
        min_relative_score: float = MIN_RELATIVE_SCORE,
        title_boost: float = TITLE_BOOST,
    ) -> None:
        """Tokenize the corpus and build the index — once, at build time.

        Args:
            chunks: Corpus to index.
            min_score: BM25 score a chunk must exceed to be returned.
            min_matched_terms: Required distinct query-term matches for queries
                containing more than one term.
            min_relative_score: Fraction of the best candidate score a result
                must reach, preventing a long tail of weak matches.
            title_boost: Weight applied to the title-only BM25 score.
        """
        self._chunks = chunks
        self._min_score = min_score
        self._min_matched_terms = max(1, min_matched_terms)
        self._min_relative_score = min(max(0.0, min_relative_score), 1.0)
        self._title_boost = max(0.0, title_boost)
        # Corpus tokenization happens exactly once, here — never at query time.
        self._title_tokens = [_tokenize(c.title) for c in chunks]
        self._body_tokens = [_tokenize(c.text) for c in chunks]
        self._document_terms = [
            set(title_tokens) | set(body_tokens)
            for title_tokens, body_tokens in zip(
                self._title_tokens, self._body_tokens, strict=True
            )
        ]
        self._title_index = BM25Okapi(self._title_tokens)
        self._body_index = BM25Okapi(self._body_tokens)

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Return up to ``top_k`` chunks that pass all relevance gates.

        Args:
            query: Free-text query; empty or non-alphanumeric input yields [].
            top_k: Maximum number of results.

        Returns:
            Scored hits, highest score first. Empty when nothing passes the
            term-overlap and absolute/relative score gates — never a
            least-bad match, so the generator can report "not found".
        """
        tokens = _normalize_query(query)
        if not tokens or top_k <= 0:
            return []
        query_terms = set(tokens)
        required_matches = min(
            len(query_terms),
            1 if len(query_terms) == 1 else self._min_matched_terms,
        )
        body_scores = self._body_index.get_scores(tokens)
        title_scores = self._title_index.get_scores(tokens)
        candidates = [
            (i, float(body_score) + self._title_boost * float(title_scores[i]))
            for i, body_score in enumerate(body_scores)
            if len(query_terms & self._document_terms[i]) >= required_matches
        ]
        if not candidates:
            return []
        best_score = max(score for _, score in candidates)
        score_floor = max(self._min_score, best_score * self._min_relative_score)
        # Partial selection beats sorting the whole score array (O(n log k)).
        top = heapq.nlargest(top_k, candidates, key=lambda pair: pair[1])
        return [
            ScoredChunk(chunk=self._chunks[i], score=score, source=self.SOURCE)
            for i, score in top
            if score >= score_floor
        ]
