"""Minimal environment-based configuration for the submission pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_str(name: str, default: str) -> str:
    """Read one setting, treating unset or blank values as the default.

    A blank line such as ``TEMPERATURE=`` in .env must select the shared
    default rather than an empty value, so every setting reads through
    this single rule.
    """
    return (os.getenv(name) or "").strip() or default


def _env_float(name: str, default: str) -> float:
    """Read one float setting, naming the variable on malformed input."""
    raw = _env_str(name, default)
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


def _env_int(name: str, default: str) -> int:
    """Read one integer setting, naming the variable on malformed input."""
    raw = _env_str(name, default)
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


MODEL_NAME: str = _env_str("MODEL_NAME", "gpt-5-mini")
RETRIEVER_MODEL_NAME: str = _env_str("RETRIEVER_MODEL_NAME", MODEL_NAME)
REPORTER_MODEL_NAME: str = _env_str("REPORTER_MODEL_NAME", MODEL_NAME)
TEMPERATURE: float = _env_float("TEMPERATURE", "0")
LLM_TIMEOUT_SECONDS: float = _env_float("LLM_TIMEOUT_SECONDS", "30")
LLM_MAX_RETRIES: int = _env_int("LLM_MAX_RETRIES", "2")
# A relative KB_PATH (including the .env.example default) is anchored to the
# project root so the pipeline works from any working directory; an absolute
# KB_PATH replaces the anchor entirely per pathlib joining rules.
KB_PATH: str = str(_PROJECT_ROOT / _env_str("KB_PATH", "knowledge_base.txt"))

# Retrieval mode is configuration-owned so evaluations remain reproducible;
# the LLM is never allowed to choose a strategy at runtime.
SEARCH_MODE: str = _env_str("SEARCH_MODE", "lexical").lower()
EMBEDDING_MODEL_NAME: str = _env_str(
    "EMBEDDING_MODEL_NAME", "text-embedding-3-small"
)
# Derived from the precision-weighted threshold sweep documented in
# threshold_calibration.md; an environment override remains available for
# controlled experiments.
MIN_COSINE: float = _env_float("MIN_COSINE", "0.392817")
RRF_K: int = _env_int("RRF_K", "60")
EMBED_CACHE_DIR: str = str(
    _PROJECT_ROOT / _env_str("EMBED_CACHE_DIR", ".cache/embeddings")
)

if SEARCH_MODE not in {"lexical", "semantic", "hybrid"}:
    raise ValueError(
        "SEARCH_MODE must be one of: lexical, semantic, hybrid"
    )
if LLM_TIMEOUT_SECONDS <= 0:
    raise ValueError("LLM_TIMEOUT_SECONDS must be greater than zero")
if LLM_MAX_RETRIES < 0:
    raise ValueError("LLM_MAX_RETRIES must not be negative")
if not -1.0 <= MIN_COSINE <= 1.0:
    raise ValueError("MIN_COSINE must be between -1.0 and 1.0")
if RRF_K <= 0:
    raise ValueError("RRF_K must be greater than zero")
