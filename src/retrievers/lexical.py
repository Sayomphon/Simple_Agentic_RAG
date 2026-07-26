"""Deterministic normalized lexical retrieval over the local knowledge base.

The scoring code in this module is intentionally kept byte-for-byte
behavior-compatible with the original tool implementation. The
``LexicalRetriever`` adapter only exposes its existing ranking evidence
through the shared retriever protocol.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.config import KB_PATH
from src.retrievers.base import ScoredChunk

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

# Derivational families and synonyms only. Pure inflections (books,
# submitted, methods, ...) are intentionally absent: the stemmer already
# maps them onto the same canonical form, and every remaining entry fails
# stem(surface) == stem(target) — e.g. escalated stems to "escalat", not
# "escalation", and vacations stems to "vacation", never to "leave".
TOKEN_ALIASES: dict[str, str] = {
    "escalate": "escalation",
    "escalated": "escalation",
    "escalates": "escalation",
    "escalating": "escalation",
    "lodging": "hotel",
    "overseas": "international",
    "reimburse": "reimbursement",
    "reimbursed": "reimbursement",
    "reimbursing": "reimbursement",
    "remotely": "remote",
    "staff": "employee",
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
# TF saturation is off by default: the ablation study measured no metric
# gain over stemming alone anywhere inside the safe band. The calibration
# sweep showed exact match holds at 100% up to K_TF=0.08 and drops at
# 0.09, where title-only matches start outranking multi-term body evidence
# in these short sections; 0.05 keeps margin inside that band. See
# docs/DESIGN_NOTES.md ("Term frequency: measured, then dropped"). The
# scoring path stays available to evals via
# RetrievalSettings(use_tf_saturation=True).
K_TF = 0.05

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
class RetrievalSettings:
    """Feature switches for the scoring pipeline, used by the ablation study.

    The defaults reproduce the production pipeline exactly; every eval
    variant is expressed as a non-default instance (src/evaluation/ablation.py).
    """

    use_query_filters: bool = True
    use_aliases: bool = True
    use_idf: bool = True
    title_weight: float = TITLE_MATCH_WEIGHT
    use_relevance_gate: bool = True
    use_sibling_expansion: bool = True
    use_stemming: bool = True
    use_tf_saturation: bool = False
    k_tf: float = K_TF


DEFAULT_SETTINGS = RetrievalSettings()


@dataclass(frozen=True)
class LexicalCandidate:
    """A scored source section and the evidence used to rank it."""

    index: int
    chunk: str
    matched_terms: frozenset[str]
    title_matches: frozenset[str]
    score: float


def tokenize(text: str) -> set[str]:
    """Return unique lowercase English/alphanumeric terms without aliases."""
    return set(TOKEN_PATTERN.findall(text.lower()))


def _undouble(token: str) -> str:
    """Drop one trailing doubled consonant (submitt -> submit), keeping
    legitimate ll/ss endings (fall, miss)."""
    if (
        len(token) >= 2
        and token[-1] == token[-2]
        and token[-1] not in ("l", "s")
    ):
        return token[:-1]
    return token


def _stem_once(token: str) -> str:
    """Apply one pass of plural, verbal-suffix, and final-e rules."""
    if len(token) < 4:
        return token  # us, is, its — too short to strip safely

    if token.endswith("ies") and len(token) >= 5:
        token = token[:-3] + "y"  # policies -> policy
    elif token.endswith("sses"):
        token = token[:-2]  # processes -> process
    elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]  # fees -> fee, meetings -> meeting

    if token.endswith("ing") and len(token) >= 6:
        token = _undouble(token[:-3])  # booking -> book, meeting -> meet
    elif token.endswith("ed") and len(token) >= 5:
        token = _undouble(token[:-2])  # submitted -> submit

    if token.endswith("e") and len(token) >= 5:
        token = token[:-1]  # approve -> approv, matching approved
    return token


def stem(token: str) -> str:
    """Light inflectional stemmer: -s/-es/-ies/-ed/-ing + final-e elision.

    Runs the single-pass rules to a fixpoint so stemming is idempotent —
    e.g. ``expenses -> expen`` directly rather than via an intermediate
    ``expense`` that a second call would shorten further. Derivational
    forms and synonyms remain the alias tables' responsibility.
    """
    while True:
        stemmed = _stem_once(token)
        if stemmed == token:
            return token
        token = stemmed


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


def normalized_token_counts(
    text: str,
    *,
    is_query: bool,
    settings: RetrievalSettings = DEFAULT_SETTINGS,
) -> Counter[str]:
    """Count canonical-token occurrences for query or document text.

    Order matters: reviewed aliases win over the automatic stemmer, and
    the filter sets are defined on surface forms, so both apply before
    stemming. Query and document sides share the same final stem step,
    which keeps their canonical spaces aligned by construction.
    """
    source = normalize_phrases(text) if settings.use_aliases else text.lower()

    counts: Counter[str] = Counter()
    for token in TOKEN_PATTERN.findall(source):
        if settings.use_aliases:
            token = TOKEN_ALIASES.get(token, token)
        if is_query and settings.use_query_filters and (
            token in STOPWORDS
            or token in DOMAIN_GENERIC_TERMS
            or token in QUERY_FRAMING_TERMS
        ):
            continue
        if settings.use_stemming:
            token = stem(token)
        counts[token] += 1
    return counts


def normalized_tokens(
    text: str,
    *,
    is_query: bool,
    settings: RetrievalSettings = DEFAULT_SETTINGS,
) -> frozenset[str]:
    """Return canonical tokens for query or document-side matching."""
    return frozenset(
        normalized_token_counts(text, is_query=is_query, settings=settings)
    )


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
    body_counts: Mapping[str, int],
    idf_by_term: dict[str, float],
    settings: RetrievalSettings = DEFAULT_SETTINGS,
) -> tuple[float, frozenset[str], frozenset[str]]:
    """Score distinct title/body term matches without double-counting.

    Title matches stay binary — titles are too short for frequency to
    mean anything. Body-only matches optionally saturate with term
    frequency (BM25-style ``tf/(tf+k)``), which affects ranking margins
    while candidacy and cutoffs keep operating on term presence.
    """
    title_matches = query_terms & title_terms
    body_only_matches = (
        query_terms & frozenset(body_counts)
    ) - title_matches
    matched_terms = title_matches | body_only_matches

    title_score = sum(
        idf_by_term[term] * settings.title_weight
        for term in title_matches
    )
    body_score = 0.0
    for term in body_only_matches:
        term_weight = idf_by_term[term] * BODY_MATCH_WEIGHT
        if settings.use_tf_saturation:
            term_frequency = body_counts[term]
            term_weight *= term_frequency / (term_frequency + settings.k_tf)
        body_score += term_weight
    return title_score + body_score, matched_terms, title_matches


def is_candidate(
    matched_terms: frozenset[str],
    title_matches: frozenset[str],
) -> bool:
    """Require a title anchor or at least two matching evidence terms."""
    return bool(title_matches) or len(matched_terms) >= 2


@dataclass(frozen=True, eq=False)
class ParsedSection:
    """One knowledge-base section with its normalized match structures."""

    chunk: str
    title_terms: frozenset[str]
    body_counts: Counter[str]
    all_terms: frozenset[str]


def _split_sections(text: str, kb_path: Path) -> tuple[str, ...]:
    """Split file text into raw section chunks, enforcing the format contract."""
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

        chunks.append(text[match.start() : end].strip())
    return tuple(chunks)


# The (mtime_ns, size) cache key invalidates whenever the file changes;
# nanosecond mtime resolution on modern filesystems makes a same-key
# rewrite vanishingly unlikely. Exceptions are never cached by lru_cache,
# so a malformed file keeps failing loudly on every call.
@lru_cache(maxsize=8)
def _cached_chunks(
    path_str: str,
    mtime_ns: int,
    size: int,
) -> tuple[str, ...]:
    kb_path = Path(path_str)
    return _split_sections(kb_path.read_text(encoding="utf-8"), kb_path)


@lru_cache(maxsize=32)
def _cached_parsed_sections(
    path_str: str,
    mtime_ns: int,
    size: int,
    settings: RetrievalSettings,
) -> tuple[ParsedSection, ...]:
    sections = []
    for chunk in _cached_chunks(path_str, mtime_ns, size):
        title, _, body = chunk.partition("\n")
        title_terms = normalized_tokens(title, is_query=False, settings=settings)
        body_counts = normalized_token_counts(
            body, is_query=False, settings=settings
        )
        sections.append(
            ParsedSection(
                chunk=chunk,
                title_terms=title_terms,
                body_counts=body_counts,
                all_terms=title_terms | frozenset(body_counts),
            )
        )
    return tuple(sections)


def load_knowledge_base(path: str | Path | None = None) -> list[str]:
    """Read a section-formatted text file and return its raw chunks.

    Each section must begin with ``--- Section title ---``. The returned
    strings preserve the title header and body exactly apart from surrounding
    whitespace, making the tool output an auditable slice of the source file.
    """
    kb_path = Path(path if path is not None else KB_PATH)
    stat = kb_path.stat()  # FileNotFoundError propagates unchanged.
    return list(_cached_chunks(str(kb_path), stat.st_mtime_ns, stat.st_size))


def search_scored(
    query: str,
    path: str | Path | None = None,
    *,
    settings: RetrievalSettings = DEFAULT_SETTINGS,
) -> list[ScoredChunk]:
    """Return normalized lexical matches with their ranking evidence."""
    kb_path = Path(path if path is not None else KB_PATH)
    stat = kb_path.stat()  # FileNotFoundError propagates unchanged.
    sections = _cached_parsed_sections(
        str(kb_path), stat.st_mtime_ns, stat.st_size, settings
    )

    query_terms = normalized_tokens(query, is_query=True, settings=settings)
    if not query_terms:
        return []

    document_terms = [section.all_terms for section in sections]
    idf_by_term = {
        term: (
            inverse_document_frequency(term, document_terms)
            if settings.use_idf
            else 1.0
        )
        for term in query_terms
    }

    candidates: list[LexicalCandidate] = []
    for index, section in enumerate(sections):
        score, matched_terms, title_matches = score_chunk(
            query_terms,
            section.title_terms,
            section.body_counts,
            idf_by_term,
            settings,
        )
        if not matched_terms:
            continue
        if settings.use_relevance_gate and not is_candidate(
            matched_terms, title_matches
        ):
            continue
        candidates.append(
            LexicalCandidate(
                index=index,
                chunk=section.chunk,
                matched_terms=matched_terms,
                title_matches=title_matches,
                score=score,
            )
        )

    if not candidates:
        return []

    if settings.use_relevance_gate:
        best_score = max(candidate.score for candidate in candidates)
        cutoff = max(MIN_ABSOLUTE_SCORE, best_score * MIN_RELATIVE_SCORE)
        selected = [
            candidate
            for candidate in candidates
            if candidate.score >= cutoff
        ]
    else:
        selected = list(candidates)

    # A focused two-term topic may have a lower-scoring sibling section that
    # shares a title anchor with a full-coverage result. This generic relation
    # recovers complementary evidence such as Hybrid Work for "work remotely"
    # without admitting Travel sections for "international card": the latter
    # has no full-coverage candidate with a matching title anchor.
    if settings.use_sibling_expansion and len(query_terms) == 2:
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
    return [
        ScoredChunk(
            index=candidate.index,
            chunk=candidate.chunk,
            score=candidate.score,
            method="lexical",
            extras={
                "matched_terms": tuple(sorted(candidate.matched_terms)),
                "title_matches": tuple(sorted(candidate.title_matches)),
            },
        )
        for candidate in selected
    ]


def search(
    query: str,
    path: str | Path | None = None,
    *,
    settings: RetrievalSettings = DEFAULT_SETTINGS,
) -> list[str]:
    """Return normalized lexical matches in deterministic relevance order."""
    return [
        result.chunk
        for result in search_scored(query, path=path, settings=settings)
    ]


class LexicalRetriever:
    """Protocol adapter around the measured lexical scoring pipeline."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        settings: RetrievalSettings = DEFAULT_SETTINGS,
    ) -> None:
        self._path = path
        self._settings = settings

    def search(self, query: str) -> list[ScoredChunk]:
        """Return threshold-gated lexical results with raw source chunks."""
        return search_scored(
            query,
            path=self._path,
            settings=self._settings,
        )


__all__ = [
    "BODY_MATCH_WEIGHT",
    "DEFAULT_SETTINGS",
    "K_TF",
    "KnowledgeBaseFormatError",
    "LexicalRetriever",
    "MIN_ABSOLUTE_SCORE",
    "MIN_RELATIVE_SCORE",
    "PHRASE_ALIASES",
    "QUERY_FRAMING_TERMS",
    "RetrievalSettings",
    "TITLE_MATCH_WEIGHT",
    "TOKEN_ALIASES",
    "discriminative_terms",
    "inverse_document_frequency",
    "is_candidate",
    "load_knowledge_base",
    "normalize_phrases",
    "normalized_token_counts",
    "normalized_tokens",
    "search",
    "search_scored",
    "score_chunk",
    "stem",
    "tokenize",
]
