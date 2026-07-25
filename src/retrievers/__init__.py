"""Retrieval strategies behind one ``Retriever`` protocol.

Public API — everything other layers may import lives here:
    - ``Chunk`` / ``ScoredChunk`` / ``Retriever`` / ``load_chunks`` (contracts)
    - ``has_english_search_terms`` (query-language probe for the agent layer)
    - ``get_retriever`` (factory; selects by ``config.SEARCH_MODE``)

Implementation modules (``keyword``, ``dense``, ``hybrid``) are internal;
``dense`` is imported lazily by the factory so keyword-only processes never
touch the OpenAI client.
"""

from src.retrievers.base import Chunk, Retriever, ScoredChunk, load_chunks
from src.retrievers.factory import get_retriever
from src.retrievers.keyword import BM25Retriever, has_english_search_terms

__all__ = [
    "BM25Retriever",
    "Chunk",
    "Retriever",
    "ScoredChunk",
    "get_retriever",
    "has_english_search_terms",
    "load_chunks",
]
