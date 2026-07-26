"""Shared LLM construction for the two submission agents."""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from src.config import (
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    MODEL_NAME,
    TEMPERATURE,
)


@lru_cache(maxsize=8)
def get_llm(model_name: str | None = None) -> ChatOpenAI:
    """Build one ChatOpenAI client per distinct model name.

    Every client carries an explicit request timeout and a bounded retry
    budget so a stalled provider cannot hang the CLI indefinitely; the SDK
    retries only transient provider errors. gpt-5 family models accept only
    the default temperature, so the parameter is passed to other models only.
    """
    resolved_name = model_name or MODEL_NAME
    if resolved_name.startswith("gpt-5"):
        return ChatOpenAI(
            model=resolved_name,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )
    return ChatOpenAI(
        model=resolved_name,
        temperature=TEMPERATURE,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )
