"""Shared contracts for retrieval strategies.

The public tool still returns ``list[str]``. These richer internal results
remain behind that boundary so ranking evidence can be evaluated without
changing the two-agent handoff.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

RetrievalMethod = Literal["lexical", "semantic", "both"]
EmptyReason = Literal["no_query_terms", "gated_out"]


@dataclass(frozen=True)
class Chunk:
    """One raw knowledge-base section with its stable source position."""

    index: int
    text: str

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Chunk index must be non-negative")
        if not self.text.strip():
            raise ValueError("Chunk text must be non-empty")


@dataclass(frozen=True)
class ScoredChunk:
    """A raw source section plus strategy-specific ranking evidence."""

    index: int
    chunk: str
    score: float
    method: str
    extras: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("ScoredChunk index must be non-negative")
        if not self.chunk.strip():
            raise ValueError("ScoredChunk text must be non-empty")
        if not math.isfinite(self.score):
            raise ValueError("ScoredChunk score must be finite")
        if not self.method.strip():
            raise ValueError("ScoredChunk method must be non-empty")
        object.__setattr__(
            self,
            "extras",
            MappingProxyType(dict(self.extras)),
        )


@dataclass(frozen=True)
class SnippetTrace:
    """Safe, display-only retrieval evidence for one returned section."""

    title: str
    score: float
    method: RetrievalMethod
    detail: str

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("SnippetTrace title must be non-empty")
        if not math.isfinite(self.score):
            raise ValueError("SnippetTrace score must be finite")
        if self.method not in {"lexical", "semantic", "both"}:
            raise ValueError(
                "SnippetTrace method must be lexical, semantic, or both"
            )


@dataclass(frozen=True)
class SearchTelemetry:
    """Diagnostics for one configured-retriever invocation.

    Telemetry is carried beside the raw-snippet handoff and must never be
    included in the Report Generator prompt.
    """

    mode: str
    query: str
    latency_ms: float
    empty_reason: EmptyReason | None
    snippets: tuple[SnippetTrace, ...]

    def __post_init__(self) -> None:
        if not self.mode.strip():
            raise ValueError("SearchTelemetry mode must be non-empty")
        if not isinstance(self.query, str):
            raise TypeError("SearchTelemetry query must be a string")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError(
                "SearchTelemetry latency_ms must be finite and non-negative"
            )
        if self.empty_reason not in {None, "no_query_terms", "gated_out"}:
            raise ValueError(
                "SearchTelemetry empty_reason must be no_query_terms, "
                "gated_out, or None"
            )
        if self.snippets and self.empty_reason is not None:
            raise ValueError(
                "SearchTelemetry with snippets cannot have an empty_reason"
            )


@runtime_checkable
class Retriever(Protocol):
    """Minimal contract shared by every threshold-gated retriever."""

    def search(self, query: str) -> list[ScoredChunk]:
        """Return relevant raw sections in deterministic rank order."""


__all__ = [
    "Chunk",
    "EmptyReason",
    "RetrievalMethod",
    "Retriever",
    "ScoredChunk",
    "SearchTelemetry",
    "SnippetTrace",
]
