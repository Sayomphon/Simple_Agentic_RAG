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
from typing import Protocol, runtime_checkable


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


@runtime_checkable
class Retriever(Protocol):
    """Minimal contract shared by every threshold-gated retriever."""

    def search(self, query: str) -> list[ScoredChunk]:
        """Return relevant raw sections in deterministic rank order."""


__all__ = ["Chunk", "Retriever", "ScoredChunk"]
