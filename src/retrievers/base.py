"""Shared retrieval contracts: chunking, scored results, and the protocol.

Everything downstream (tools, evaluation, UI) depends only on the types in
this module, so retrieval strategies can be added or swapped without
touching any other layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.config import KB_PATH

_SECTION_PATTERN = re.compile(r"^--- (?P<title>.+?) ---$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    """One knowledge-base section.

    Attributes:
        title: Section title taken from the ``--- Title ---`` delimiter.
        text: Body text of the section.
        index: Global position of the section across the whole corpus —
            unique over every source file (hybrid fusion uses it as its
            dedup key), assigned in sorted-filename ingestion order.
        source_file: Name of the file the section came from (provenance
            for UI layers; empty for ad-hoc chunks built in tests).
    """

    title: str
    text: str
    index: int = 0
    source_file: str = ""

    def as_snippet(self) -> str:
        """Render the chunk as a self-describing snippet string."""
        return f"[{self.title}]\n{self.text}"


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieval hit: the chunk plus ranking metadata.

    The metadata exists for evaluation and UI layers — ``score`` explains
    ranking decisions, ``source`` records which retriever produced the hit
    (``"bm25"``, ``"dense"``, or ``"bm25+dense"`` after hybrid fusion).
    """

    chunk: Chunk
    score: float
    source: str

    @property
    def title(self) -> str:
        """Section title (pass-through so callers need not unwrap ``chunk``)."""
        return self.chunk.title

    @property
    def text(self) -> str:
        """Section body (pass-through convenience)."""
        return self.chunk.text

    @property
    def source_file(self) -> str:
        """Originating knowledge-base file (pass-through convenience)."""
        return self.chunk.source_file

    def as_snippet(self) -> str:
        """Render the underlying chunk; metadata is deliberately excluded so
        the agent-facing snippet format stays identical across retrievers."""
        return self.chunk.as_snippet()


class Retriever(Protocol):
    """Contract for every retrieval implementation (keyword, dense, hybrid)."""

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Return up to ``top_k`` relevant chunks with scores, best first."""
        ...


def load_chunks(path: str = KB_PATH) -> list[Chunk]:
    """Read the knowledge base and split it into titled chunks.

    Splitting strategy lives here and only here: retrieval classes receive
    ready-made ``Chunk`` objects, so the chunking rules can change without
    touching any ranking code.

    Args:
        path: Knowledge-base location — either a directory of ``.txt``
            files (ingested in sorted-filename order, so chunk indexes
            and the embedding-cache fingerprint stay deterministic) or a
            single text file.

    Returns:
        All sections, with a corpus-wide running ``index``.

    Raises:
        FileNotFoundError: If the path (or a directory's ``*.txt``
            contents) does not exist.
        ValueError: If no file contains a ``--- Title ---`` section.
    """
    kb_path = Path(path)
    if kb_path.is_dir():
        kb_files = sorted(kb_path.glob("*.txt"))
        if not kb_files:
            raise FileNotFoundError(
                f"Knowledge-base directory '{kb_path.resolve()}' contains "
                "no .txt files. Set KB_PATH in .env or src/config.py to a "
                "directory of knowledge files or a single text file."
            )
    elif kb_path.is_file():
        kb_files = [kb_path]
    else:
        raise FileNotFoundError(
            f"Knowledge base not found at '{kb_path.resolve()}'. "
            "Set KB_PATH in .env or src/config.py to a directory of "
            "knowledge files or a single text file."
        )

    chunks: list[Chunk] = []
    for kb_file in kb_files:
        raw = kb_file.read_text(encoding="utf-8")
        # re.split with one capture group yields [preamble, title1, body1, ...].
        parts = _SECTION_PATTERN.split(raw)
        for title, body in zip(parts[1::2], parts[2::2]):
            if not body.strip():
                continue
            # len(chunks) keeps the index a corpus-wide running number —
            # it must never repeat across files (hybrid fusion dedup key).
            chunks.append(
                Chunk(
                    title=title.strip(),
                    text=body.strip(),
                    index=len(chunks),
                    source_file=kb_file.name,
                )
            )
    if not chunks:
        raise ValueError(
            f"No '--- Section Title ---' sections found in {kb_path}; "
            "check the file format."
        )
    return chunks
