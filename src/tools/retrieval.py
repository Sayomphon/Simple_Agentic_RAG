"""Deterministic keyword retrieval over the local text knowledge base."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.tools import tool

from src.config import KB_PATH

SECTION_PATTERN = re.compile(r"^---\s*(?P<title>.+?)\s*---\s*$", re.MULTILINE)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
MIN_TITLE_TOPIC_MATCHES = 2

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


def tokenize(text: str) -> set[str]:
    """Return unique normalized English/alphanumeric terms."""
    return set(TOKEN_PATTERN.findall(text.lower()))


def discriminative_terms(query: str) -> set[str]:
    """Remove structural and domain-generic terms from a user query."""
    return tokenize(query) - STOPWORDS - DOMAIN_GENERIC_TERMS


def load_knowledge_base(path: str | Path | None = None) -> list[str]:
    """Read a section-formatted text file and return its raw chunks.

    Each section must begin with ``--- Section title ---``. The returned
    strings preserve the title header and body exactly apart from surrounding
    whitespace, making the tool output an auditable slice of the source file.
    """
    kb_path = Path(path if path is not None else KB_PATH)
    text = kb_path.read_text(encoding="utf-8")
    matches = list(SECTION_PATTERN.finditer(text))

    chunks: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunk = text[match.start() : end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def search(query: str, path: str | Path | None = None) -> list[str]:
    """Return every keyword-relevant chunk in deterministic rank order."""
    query_terms = discriminative_terms(query)
    if not query_terms:
        return []

    candidates: list[tuple[int, int, frozenset[str], str]] = []
    for index, chunk in enumerate(load_knowledge_base(path)):
        title, _, body = chunk.partition("\n")
        title_terms = tokenize(title)
        body_terms = tokenize(body)
        chunk_terms = title_terms | body_terms
        matched_terms = query_terms & chunk_terms
        if matched_terms:
            candidates.append(
                (
                    len(matched_terms),
                    index,
                    frozenset(query_terms & title_terms),
                    chunk,
                )
            )

    # A strict majority rejects incidental one-word matches while allowing
    # complementary sections to contribute different parts of a longer query.
    required_matches = len(query_terms) // 2 + 1

    # A verbose multi-intent query can distribute its terms across several
    # sections, so no individual section may satisfy the strict-majority rule.
    # Retain every section tied for the strongest title coverage when that
    # coverage contains at least two query terms. This treats a multi-term title
    # match as a reliable topic anchor while still rejecting weak matches such
    # as the single word "international" in an international-card query.
    strongest_title_coverage = max(
        (len(title_matches) for _, _, title_matches, _ in candidates),
        default=0,
    )

    # For a focused two-term topic, retain sibling sections that share a title
    # anchor with a full-coverage section. For example, "work remotely" first
    # identifies Remote Work Policy and then keeps Hybrid Work Guidelines via
    # their shared "work" title anchor. The exception is deliberately limited
    # to two-term queries so one broad word cannot admit unrelated sections for
    # a more specific request such as "international card fee".
    linked_title_anchors: set[str] = set()
    if len(query_terms) == 2:
        for score, _, title_matches, _ in candidates:
            if score == len(query_terms):
                linked_title_anchors.update(title_matches)

    ranked: list[tuple[int, int, str]] = []
    for score, index, title_matches, chunk in candidates:
        has_majority_coverage = score >= required_matches
        has_strongest_title_coverage = (
            strongest_title_coverage >= MIN_TITLE_TOPIC_MATCHES
            and len(title_matches) == strongest_title_coverage
        )
        is_title_linked_sibling = bool(
            linked_title_anchors and title_matches & linked_title_anchors
        )
        if (
            has_majority_coverage
            or has_strongest_title_coverage
            or is_title_linked_sibling
        ):
            ranked.append((score, index, chunk))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [chunk for _, _, chunk in ranked]


@tool
def search_knowledge_base(query: str) -> list[str]:
    """Search the local knowledge base and return all raw relevant sections.

    Use specific English subject terms found in the user's request. The tool
    returns an empty list when no section satisfies the deterministic term-
    coverage rule.

    Args:
        query: Search terms describing the information needed.
    """
    return search(query)


__all__ = [
    "discriminative_terms",
    "load_knowledge_base",
    "search",
    "search_knowledge_base",
    "tokenize",
]
