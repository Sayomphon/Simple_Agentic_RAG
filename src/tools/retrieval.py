"""Deterministic normalized lexical retrieval over the local knowledge base."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import tool

from src.config import KB_PATH

SECTION_PATTERN = re.compile(r"^---\s*(?P<title>.+?)\s*---\s*$", re.MULTILINE)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Longer aliases must precede shorter overlapping phrases.
PHRASE_ALIASES: tuple[tuple[str, str], ...] = (
    ("overseas business trip", "international business travel"),
    ("work from home", "remote work"),
    ("paid time off", "paid leave"),
    ("overseas trip", "international travel"),
    ("response time", "first response"),
    ("per diem", "daily allowance"),
)

TOKEN_ALIASES: dict[str, str] = {
    "accepted": "accept",
    "accepts": "accept",
    "booked": "book",
    "booking": "book",
    "books": "book",
    "entitlements": "entitlement",
    "escalate": "escalation",
    "escalated": "escalation",
    "escalates": "escalation",
    "lodging": "hotel",
    "methods": "method",
    "overseas": "international",
    "reimbursed": "reimbursement",
    "remotely": "remote",
    "staff": "employee",
    "submitted": "submit",
    "submitting": "submit",
    "vacation": "leave",
    "vacations": "leave",
}

QUERY_FRAMING_TERMS = frozenset(
    {
        "allow",
        "allowed",
        "each",
        "many",
        "much",
        "quickly",
    }
)

# These values are calibrated against tests/fixtures/retrieval_cases.json.
# Title matches remain stronger than body matches, while the relative cutoff
# rejects a broad one-title-term match when a more complete section exists.
TITLE_MATCH_WEIGHT = 1.5
BODY_MATCH_WEIGHT = 1.0
MIN_RELATIVE_SCORE = 0.60
MIN_ABSOLUTE_SCORE = 1.0

# These terms express sentence structure rather than the subject to retrieve.
STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "available",
        "be",
        "been",
        "being",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "here",
        "him",
        "his",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "may",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "please",
        "s",
        "she",
        "should",
        "tell",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "us",
        "was",
        "we",
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

# These terms recur across most enterprise handbook sections and do not
# identify which section the user needs.
DOMAIN_GENERIC_TERMS = frozenset(
    {
        "business",
        "company",
        "detail",
        "details",
        "employee",
        "employees",
        "information",
        "policy",
        "policies",
        "process",
        "request",
        "requests",
        "rule",
        "rules",
    }
)


class KnowledgeBaseFormatError(ValueError):
    """Raised when the knowledge-base file violates the section contract."""


@dataclass(frozen=True)
class ScoredChunk:
    """A scored source section and the evidence used to rank it."""

    index: int
    chunk: str
    matched_terms: frozenset[str]
    title_matches: frozenset[str]
    score: float


def tokenize(text: str) -> set[str]:
    """Return unique lowercase English/alphanumeric terms without aliases."""
    return set(TOKEN_PATTERN.findall(text.lower()))


def normalize_phrases(text: str) -> str:
    """Replace reviewed multi-word concepts while preserving word boundaries."""
    normalized = text.lower()
    for source, target in PHRASE_ALIASES:
        normalized = re.sub(
            rf"\b{re.escape(source)}\b",
            target,
            normalized,
        )
    return normalized


def normalized_tokens(text: str, *, is_query: bool) -> frozenset[str]:
    """Return canonical tokens for query or document-side matching."""
    phrase_normalized = normalize_phrases(text)
    raw_tokens = TOKEN_PATTERN.findall(phrase_normalized)
    canonical = {
        TOKEN_ALIASES.get(token, token)
        for token in raw_tokens
    }

    if is_query:
        canonical -= STOPWORDS
        canonical -= DOMAIN_GENERIC_TERMS
        canonical -= QUERY_FRAMING_TERMS

    return frozenset(canonical)


def discriminative_terms(query: str) -> set[str]:
    """Return canonical topic terms from a natural-language user query."""
    return set(normalized_tokens(query, is_query=True))


def inverse_document_frequency(
    term: str,
    document_terms: list[frozenset[str]],
) -> float:
    """Return smoothed IDF so rarer corpus terms receive higher weight."""
    if not document_terms:
        raise ValueError("document_terms must contain at least one document")

    document_count = len(document_terms)
    matching_documents = sum(term in terms for terms in document_terms)
    return math.log(
        (document_count + 1) / (matching_documents + 0.5)
    ) + 1.0


def score_chunk(
    query_terms: frozenset[str],
    title_terms: frozenset[str],
    body_terms: frozenset[str],
    idf_by_term: dict[str, float],
) -> tuple[float, frozenset[str], frozenset[str]]:
    """Score distinct title/body term matches without double-counting."""
    title_matches = query_terms & title_terms
    body_only_matches = (query_terms & body_terms) - title_matches
    matched_terms = title_matches | body_only_matches

    title_score = sum(
        idf_by_term[term] * TITLE_MATCH_WEIGHT
        for term in title_matches
    )
    body_score = sum(
        idf_by_term[term] * BODY_MATCH_WEIGHT
        for term in body_only_matches
    )
    return title_score + body_score, matched_terms, title_matches


def is_candidate(
    matched_terms: frozenset[str],
    title_matches: frozenset[str],
) -> bool:
    """Require a title anchor or at least two matching evidence terms."""
    return bool(title_matches) or len(matched_terms) >= 2


def load_knowledge_base(path: str | Path | None = None) -> list[str]:
    """Read a section-formatted text file and return its raw chunks.

    Each section must begin with ``--- Section title ---``. The returned
    strings preserve the title header and body exactly apart from surrounding
    whitespace, making the tool output an auditable slice of the source file.
    """
    kb_path = Path(path if path is not None else KB_PATH)
    text = kb_path.read_text(encoding="utf-8")
    if not text.strip():
        raise KnowledgeBaseFormatError(f"Knowledge base is empty: {kb_path}")

    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        raise KnowledgeBaseFormatError(
            f"No valid section headers found in: {kb_path}"
        )

    chunks: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end]
        if not body.strip():
            title = match.group("title").strip()
            raise KnowledgeBaseFormatError(
                f"Knowledge base section has no body ({title!r}): {kb_path}"
            )

        chunk = text[match.start() : end].strip()
        chunks.append(chunk)
    return chunks


def search(query: str, path: str | Path | None = None) -> list[str]:
    """Return normalized lexical matches in deterministic relevance order."""
    chunks = load_knowledge_base(path)
    query_terms = normalized_tokens(query, is_query=True)
    if not query_terms:
        return []

    parsed_chunks: list[
        tuple[int, str, frozenset[str], frozenset[str]]
    ] = []
    document_terms: list[frozenset[str]] = []
    for index, chunk in enumerate(chunks):
        title, _, body = chunk.partition("\n")
        title_terms = normalized_tokens(title, is_query=False)
        body_terms = normalized_tokens(body, is_query=False)
        parsed_chunks.append((index, chunk, title_terms, body_terms))
        document_terms.append(title_terms | body_terms)

    idf_by_term = {
        term: inverse_document_frequency(term, document_terms)
        for term in query_terms
    }

    candidates: list[ScoredChunk] = []
    for index, chunk, title_terms, body_terms in parsed_chunks:
        score, matched_terms, title_matches = score_chunk(
            query_terms,
            title_terms,
            body_terms,
            idf_by_term,
        )
        if is_candidate(matched_terms, title_matches):
            candidates.append(
                ScoredChunk(
                    index=index,
                    chunk=chunk,
                    matched_terms=matched_terms,
                    title_matches=title_matches,
                    score=score,
                )
            )

    if not candidates:
        return []

    best_score = max(candidate.score for candidate in candidates)
    cutoff = max(MIN_ABSOLUTE_SCORE, best_score * MIN_RELATIVE_SCORE)
    selected = [
        candidate
        for candidate in candidates
        if candidate.score >= cutoff
    ]

    # A focused two-term topic may have a lower-scoring sibling section that
    # shares a title anchor with a full-coverage result. This generic relation
    # recovers complementary evidence such as Hybrid Work for "work remotely"
    # without admitting Travel sections for "international card": the latter
    # has no full-coverage candidate with a matching title anchor.
    if len(query_terms) == 2:
        linked_title_anchors: set[str] = set()
        for candidate in selected:
            if candidate.matched_terms == query_terms:
                linked_title_anchors.update(candidate.title_matches)

        selected_indices = {candidate.index for candidate in selected}
        selected.extend(
            candidate
            for candidate in candidates
            if candidate.index not in selected_indices
            and bool(candidate.title_matches & linked_title_anchors)
        )

    selected.sort(key=lambda item: (-item.score, item.index))
    return [candidate.chunk for candidate in selected]


@tool
def search_knowledge_base(query: str) -> list[str]:
    """Search the local knowledge base and return all raw relevant sections.

    The tool normalizes reviewed domain phrases and terms, then applies
    deterministic IDF-weighted title/body scoring. It returns an empty list
    when no section passes the relevance gate.

    Args:
        query: Search terms describing the information needed.
    """
    return search(query)


__all__ = [
    "BODY_MATCH_WEIGHT",
    "KnowledgeBaseFormatError",
    "MIN_ABSOLUTE_SCORE",
    "MIN_RELATIVE_SCORE",
    "PHRASE_ALIASES",
    "QUERY_FRAMING_TERMS",
    "ScoredChunk",
    "TITLE_MATCH_WEIGHT",
    "TOKEN_ALIASES",
    "discriminative_terms",
    "inverse_document_frequency",
    "is_candidate",
    "load_knowledge_base",
    "normalize_phrases",
    "normalized_tokens",
    "search",
    "search_knowledge_base",
    "score_chunk",
    "tokenize",
]
