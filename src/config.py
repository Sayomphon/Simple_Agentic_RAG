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
