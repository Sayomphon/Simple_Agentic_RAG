"""Agent-facing retrieval tool and compatibility exports.

The tool schema and ``list[str]`` result remain stable. Rich scores stay
inside the configured retriever and never enter the agent handoff.
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.retrievers.base import ScoredChunk
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


@tool
def search_knowledge_base(query: str) -> list[str]:
    """Search the local knowledge base and return all raw relevant sections.

    The configured lexical, semantic, or hybrid strategy applies its
    relevance gate internally. Returned strings remain byte-exact source
    sections and an empty list means no section passed that gate.

    Args:
        query: Search terms describing the information needed.
    """
    return [result.chunk for result in get_retriever().search(query)]


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
