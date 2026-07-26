"""Agent-facing retrieval tool and compatibility exports.

The tool schema and ``list[str]`` result remain stable. Rich scores stay
inside the configured retriever and never enter the agent handoff.
"""

from __future__ import annotations

import math
import time
from threading import local
from typing import cast

from langchain_core.tools import tool

from src import config
from src.retrievers.base import (
    EmptyReason,
    RetrievalMethod,
    ScoredChunk,
    SearchTelemetry,
    SnippetTrace,
)
from src.retrievers.factory import get_retriever
from src.retrievers.lexical import (
    BODY_MATCH_WEIGHT,
    DEFAULT_SETTINGS,
    DOMAIN_GENERIC_TERMS,
    K_TF,
    MIN_ABSOLUTE_SCORE,
    MIN_RELATIVE_SCORE,
    PHRASE_ALIASES,
    QUERY_FRAMING_TERMS,
    SECTION_PATTERN,
    STOPWORDS,
    TITLE_MATCH_WEIGHT,
    TOKEN_ALIASES,
    TOKEN_PATTERN,
    KnowledgeBaseFormatError,
    LexicalRetriever,
    RetrievalSettings,
    discriminative_terms,
    inverse_document_frequency,
    is_candidate,
    load_knowledge_base,
    normalize_phrases,
    normalized_token_counts,
    normalized_tokens,
    score_chunk,
    search,
    search_scored,
    stem,
    tokenize,
)

_telemetry_context = local()
_MAX_DETAIL_TERMS = 32
_MAX_DETAIL_TERM_CHARS = 64


def _section_title(chunk: str) -> str:
    """Extract the validated section title without retaining the full chunk."""
    header = chunk.partition("\n")[0].strip()
    match = SECTION_PATTERN.fullmatch(header)
    return match.group("title") if match else header


def _optional_number(value: object) -> str:
    """Format one allowlisted numeric diagnostic without exposing raw extras."""
    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{value:.4f}"
    return "n/a"


def _matched_terms_detail(value: object) -> str | None:
    """Bound query-derived lexical detail before it reaches a UI consumer."""
    if not isinstance(value, tuple):
        return None
    terms = [
        str(term)[:_MAX_DETAIL_TERM_CHARS]
        for term in value[:_MAX_DETAIL_TERMS]
    ]
    suffix = ", …" if len(value) > _MAX_DETAIL_TERMS else ""
    return "matched_terms=" + ", ".join(terms) + suffix


def _trace_detail(result: ScoredChunk) -> str:
    """Render only reviewed retrieval metadata for CLI/UI diagnostics."""
    if result.method == "lexical":
        matched_terms = _matched_terms_detail(
            result.extras.get("matched_terms")
        )
        if matched_terms is not None:
            return matched_terms
        return "lexical_score=" + _optional_number(result.score)
    if result.method == "semantic":
        return "cosine=" + _optional_number(
            result.extras.get("cosine", result.score)
        )
    if result.method == "both":
        return (
            "rrf="
            + _optional_number(result.score)
            + ", lexical="
            + _optional_number(result.extras.get("lexical_score"))
            + ", semantic="
            + _optional_number(result.extras.get("semantic_score"))
        )
    return "score=" + _optional_number(result.score)


def _empty_reason(
    mode: str,
    query: str,
    results: list[ScoredChunk],
) -> EmptyReason | None:
    """Explain a legitimate empty result without changing search behavior."""
    if results:
        return None
    if mode == "lexical" and not normalized_tokens(query, is_query=True):
        return "no_query_terms"
    return "gated_out"


def consume_last_telemetry() -> SearchTelemetry | None:
    """Return and clear telemetry from the latest search in this context.

    Thread-local storage preserves values across LangChain's copied execution
    context while isolating concurrent synchronous request workers. Consumers
    must call this immediately after each tool invocation.
    """
    telemetry = getattr(_telemetry_context, "last", None)
    _telemetry_context.last = None
    return telemetry


@tool
def search_knowledge_base(query: str) -> list[str]:
    """Search the local knowledge base and return all raw relevant sections.

    The configured lexical, semantic, or hybrid strategy applies its
    relevance gate internally. Returned strings remain byte-exact source
    sections and an empty list means no section passed that gate.

    Args:
        query: Search terms describing the information needed.
    """
    _telemetry_context.last = None
    mode = config.SEARCH_MODE.strip().lower()
    started = time.perf_counter()
    results = get_retriever().search(query)
    latency_ms = (time.perf_counter() - started) * 1000
    _telemetry_context.last = SearchTelemetry(
        mode=mode,
        query=query,
        latency_ms=latency_ms,
        empty_reason=_empty_reason(mode, query, results),
        snippets=tuple(
            SnippetTrace(
                title=_section_title(result.chunk),
                score=result.score,
                method=cast(RetrievalMethod, result.method),
                detail=_trace_detail(result),
            )
            for result in results
        ),
    )
    return [result.chunk for result in results]


__all__ = [
    "BODY_MATCH_WEIGHT",
    "DEFAULT_SETTINGS",
    "DOMAIN_GENERIC_TERMS",
    "K_TF",
    "KnowledgeBaseFormatError",
    "LexicalRetriever",
    "MIN_ABSOLUTE_SCORE",
    "MIN_RELATIVE_SCORE",
    "PHRASE_ALIASES",
    "QUERY_FRAMING_TERMS",
    "RetrievalSettings",
    "SECTION_PATTERN",
    "STOPWORDS",
    "ScoredChunk",
    "TITLE_MATCH_WEIGHT",
    "TOKEN_ALIASES",
    "TOKEN_PATTERN",
    "consume_last_telemetry",
    "discriminative_terms",
    "inverse_document_frequency",
    "is_candidate",
    "load_knowledge_base",
    "normalize_phrases",
    "normalized_token_counts",
    "normalized_tokens",
    "score_chunk",
    "search",
    "search_knowledge_base",
    "search_scored",
    "stem",
    "tokenize",
]
