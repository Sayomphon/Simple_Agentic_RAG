"""Shared LLM construction for the two submission agents."""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from src.config import MODEL_NAME, TEMPERATURE


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Build the shared ChatOpenAI client (one instance per process).

    gpt-5 family models accept only the default temperature, so the
    parameter is passed only to models that support setting it.
    """
    if MODEL_NAME.startswith("gpt-5"):
        return ChatOpenAI(model=MODEL_NAME)
    return ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
