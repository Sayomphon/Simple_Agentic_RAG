"""Deterministic keyword retrieval over the local text knowledge base."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.tools import tool

from src.config import KB_PATH

SECTION_PATTERN = re.compile(r"^---\s*(?P<title>.+?)\s*---\s*$", re.MULTILINE)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

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

    ranked: list[tuple[int, int, str]] = []
    for index, chunk in enumerate(load_knowledge_base(path)):
        title, _, body = chunk.partition("\n")
        title_terms = tokenize(title)
        body_terms = tokenize(body)
        chunk_terms = title_terms | body_terms
        score = len(query_terms & chunk_terms)

        # A title match anchors the query to the section's declared topic.
        # Content-only matches are accepted when every discriminative query
        # term is present, preventing one incidental word from surfacing an
        # unrelated section.
        is_relevant = bool(query_terms & title_terms) or query_terms.issubset(
            body_terms
        )
        if score > 0 and is_relevant:
            ranked.append((score, index, chunk))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [chunk for _, _, chunk in ranked]


@tool
def search_knowledge_base(query: str) -> list[str]:
    """Search the local knowledge base and return all raw relevant sections.

    Use specific English subject terms found in the user's request. The tool
    returns an empty list when no discriminative keyword matches a section.

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
