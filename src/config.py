"""Minimal environment-based configuration for the submission pipeline."""

import os

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-5-mini")
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0"))
KB_PATH: str = os.getenv("KB_PATH", "knowledge_base.txt")
