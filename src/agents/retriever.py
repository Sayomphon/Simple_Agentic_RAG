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

Your only task is to gather evidence with the `search_knowledge_base` tool:
- If an English question asks about a single topic, make exactly one tool
  call and pass the user's original query unchanged as the tool `query`.
- If the user's query is not in English, issue an English translation as a
  sub-query because the knowledge base is written in English. Preserve all
  names, numbers, product names, and system names exactly. For a single topic,
  make exactly one tool call with that faithful English translation.
- If it combines several distinct topics, split it into at most three
  focused sub-queries and make one tool call per sub-query. Write those
  sub-queries in English when the user's query is not in English.
- Never answer the question yourself.
- Never summarize, rewrite, filter, or add to the tool output.
- The raw snippets returned by the tool will be passed to another agent.
"""

# Bounds the retriever prompt cost; ~2,000 characters is several paragraphs,
# far beyond any legitimate question over a 10-section knowledge base.
MAX_QUERY_CHARS = 2000

# Caps evidence volume and provider cost for multi-intent questions.
MAX_TOOL_CALLS = 3


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


def _validated_sub_queries(response: object) -> list[str]:
    """Extract sub-queries from the model response, enforcing the contract."""
    tool_calls = getattr(response, "tool_calls", None) or []
    if not 1 <= len(tool_calls) <= MAX_TOOL_CALLS:
        raise RetrievalProtocolError(
            "Data Retriever must request between 1 and "
            f"{MAX_TOOL_CALLS} retrieval tool calls"
        )

    sub_queries: list[str] = []
    for tool_call in tool_calls:
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
        sub_query = tool_args.get("query")
        if not isinstance(sub_query, str) or not sub_query.strip():
            raise RetrievalProtocolError(
                "Data Retriever requested a tool call without a query"
            )
        sub_queries.append(sub_query)
    return sub_queries


def retriever_node(state: PipelineState) -> dict[str, list[str]]:
    """Execute the model-requested tool calls and hand off their union.

    The node always runs the original query itself first (deterministic
    baseline), then appends each sub-query's new results in call order,
    deduplicated by exact chunk text. The handoff is therefore always a
    superset of ``search(original_query)`` — recall can never fall below
    the single-call baseline no matter how the model decomposes.
    """
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
    sub_queries = _validated_sub_queries(response)

    snippets = list(search_knowledge_base.invoke({"query": query}))
    seen = set(snippets)
    for sub_query in sub_queries:
        if sub_query == query:
            continue  # Already covered by the baseline search.
        for chunk in search_knowledge_base.invoke({"query": sub_query}):
            if chunk not in seen:
                seen.add(chunk)
                snippets.append(chunk)
    return {"snippets": snippets}
