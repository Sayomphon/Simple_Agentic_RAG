"""Minimal environment-based configuration for the submission pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-5-mini")
# Empty-string env values fall back to the shared default so that a blank
# line in .env cannot silently configure a nameless model.
RETRIEVER_MODEL_NAME: str = os.getenv("RETRIEVER_MODEL_NAME") or MODEL_NAME
REPORTER_MODEL_NAME: str = os.getenv("REPORTER_MODEL_NAME") or MODEL_NAME
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0"))
LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
# A relative KB_PATH (including the .env.example default) is anchored to the
# project root so the pipeline works from any working directory; an absolute
# KB_PATH replaces the anchor entirely per pathlib joining rules.
KB_PATH: str = str(_PROJECT_ROOT / os.getenv("KB_PATH", "knowledge_base.txt"))

# Retrieval mode is configuration-owned so evaluations remain reproducible;
# the LLM is never allowed to choose a strategy at runtime.
SEARCH_MODE: str = (os.getenv("SEARCH_MODE") or "lexical").strip().lower()
EMBEDDING_MODEL_NAME: str = (
    os.getenv("EMBEDDING_MODEL_NAME") or "text-embedding-3-small"
).strip()
# Derived from the precision-weighted threshold sweep documented in
# threshold_calibration.md; an environment override remains available for
# controlled experiments.
MIN_COSINE: float = float(os.getenv("MIN_COSINE", "0.392817"))
RRF_K: int = int(os.getenv("RRF_K", "60"))
EMBED_CACHE_DIR: str = str(
    _PROJECT_ROOT
    / (os.getenv("EMBED_CACHE_DIR") or ".cache/embeddings")
)

if SEARCH_MODE not in {"lexical", "semantic", "hybrid"}:
    raise ValueError(
        "SEARCH_MODE must be one of: lexical, semantic, hybrid"
    )
if not EMBEDDING_MODEL_NAME:
    raise ValueError("EMBEDDING_MODEL_NAME must be non-empty")
if not -1.0 <= MIN_COSINE <= 1.0:
    raise ValueError("MIN_COSINE must be between -1.0 and 1.0")
if RRF_K <= 0:
    raise ValueError("RRF_K must be greater than zero")
