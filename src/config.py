"""Centralized configuration for the Agentic RAG system.

Intended responsibility:
    Hold every tunable value in one place. Each setting has a sensible
    default and can be overridden via an environment variable of the
    same name (loaded from ``.env`` by python-dotenv). Secrets such as
    OPENAI_API_KEY live ONLY in the environment, never in code.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# LLM
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-5-mini")
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0"))

# Knowledge base / retrieval — KB_PATH may point at a directory of .txt
# files (ingested in sorted-filename order) or a single text file.
KB_PATH: str = os.getenv("KB_PATH", "data")
TOP_K: int = int(os.getenv("TOP_K", "4"))
# BM25 is only the ranking layer. A minimum matched-term gate in the retriever
# rejects documents that score from one incidental word in a longer query.
MIN_SCORE: float = float(os.getenv("MIN_SCORE", "2.0"))
MIN_MATCHED_TERMS: int = int(os.getenv("MIN_MATCHED_TERMS", "2"))
MIN_RELATIVE_SCORE: float = float(os.getenv("MIN_RELATIVE_SCORE", "0.55"))
TITLE_BOOST: float = float(os.getenv("TITLE_BOOST", "1.5"))

# Retrieval mode: "keyword" (BM25), "semantic" (embeddings), or "hybrid" (both)
SEARCH_MODE: str = os.getenv("SEARCH_MODE", "keyword")

# Agentic retry loop: total search attempts per query (first attempt included).
# When an attempt yields zero snippets, the query rewriter proposes a new
# search query and the retriever tries again, up to this bound.
MAX_SEARCH_ATTEMPTS: int = int(os.getenv("MAX_SEARCH_ATTEMPTS", "3"))

# Dense retrieval (used by "semantic" and "hybrid" modes only)
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
# Cosine relevance gate. Cosine similarity is never zero, so without this
# gate unanswerable queries would always surface the least-unrelated chunk.
# Tuned on measured data (July 2026, text-embedding-3-small, 54-chunk KB):
# full-sentence answerable queries scored >= 0.426 against their target
# section; unanswerable queries peaked at 0.369 ("employee home addresses").
# 0.38 sits just above that negative ceiling. Ultra-terse positives
# ("quit my job" = 0.345) fall below the gate and degrade to "not found" —
# a deliberate trade: a missed answer is recoverable, a fabricated one is not.
MIN_COSINE: float = float(os.getenv("MIN_COSINE", "0.38"))
EMBEDDING_CACHE_DIR: str = os.getenv("EMBEDDING_CACHE_DIR", ".cache")

# Hybrid fusion ("rrf" is rank-based and scale-free; "weighted" kept for
# evaluation comparison — see src/retrievers/hybrid.py for the rationale)
FUSION_METHOD: str = os.getenv("FUSION_METHOD", "rrf")
RRF_K: int = int(os.getenv("RRF_K", "60"))
DENSE_WEIGHT: float = float(os.getenv("DENSE_WEIGHT", "0.5"))
