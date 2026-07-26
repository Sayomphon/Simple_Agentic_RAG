"""Data Retriever Agent: call the custom tool and return raw snippets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents import get_llm
from src.config import RETRIEVER_MODEL_NAME
from src.tools.retrieval import search_knowledge_base

if TYPE_CHECKING:
    from src.graph import PipelineState

RETRIEVER_SYSTEM_PROMPT = """\
You are the Data Retriever Agent in a sequential RAG pipeline.

Your only task is to retrieve evidence:
- Call `search_knowledge_base` exactly once.
- Pass the user's original query unchanged as the tool `query`.
- Never answer the question yourself.
- Never summarize, rewrite, filter, or add to the tool output.
- The raw snippets returned by the tool will be passed to another agent.
"""

# Bounds the retriever prompt cost; ~2,000 characters is several paragraphs,
# far beyond any legitimate question over a 10-section knowledge base.
MAX_QUERY_CHARS = 2000


class InvalidQueryError(ValueError):
    """Raised when the incoming query is rejected before any LLM call."""


class RetrievalProtocolError(RuntimeError):
    """Raised when the Retriever LLM violates the required tool-call contract."""


def _validate_query(query: object) -> str:
    """Reject unusable queries before they cost an API call."""
    if not isinstance(query, str) or not query.strip():
        raise InvalidQueryError("Query must be a non-empty string.")
    if len(query) > MAX_QUERY_CHARS:
        raise InvalidQueryError(
            f"Query is {len(query)} characters long; "
            f"the limit is {MAX_QUERY_CHARS}."
        )
    return query


def retriever_node(state: PipelineState) -> dict[str, list[str]]:
    """Execute the model-requested custom tool call and hand off its output."""
    query = _validate_query(state["query"])
    llm_with_tool = get_llm(RETRIEVER_MODEL_NAME).bind_tools(
        [search_knowledge_base],
        tool_choice="required",
    )
    response = llm_with_tool.invoke(
        [
            SystemMessage(content=RETRIEVER_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
    )

    tool_calls = getattr(response, "tool_calls", None) or []
    if len(tool_calls) != 1:
        raise RetrievalProtocolError(
            "Data Retriever must request exactly one retrieval tool call"
        )

    tool_call = tool_calls[0]
    if not isinstance(tool_call, Mapping):
        raise RetrievalProtocolError(
            "Data Retriever returned a malformed retrieval tool call"
        )

    if tool_call.get("name") != search_knowledge_base.name:
        raise RetrievalProtocolError(
            "Data Retriever requested an unexpected retrieval tool"
        )

    tool_args = tool_call.get("args") or {}
    if not isinstance(tool_args, Mapping):
        raise RetrievalProtocolError(
            "Data Retriever returned malformed retrieval arguments"
        )

    if tool_args.get("query") != query:
        raise RetrievalProtocolError("Data Retriever changed the original query")

    snippets = search_knowledge_base.invoke({"query": query})
    return {"snippets": list(snippets)}
