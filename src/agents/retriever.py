"""Data Retriever Agent: call the custom tool and return raw snippets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents import get_llm
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


def retriever_node(state: PipelineState) -> dict[str, list[str]]:
    """Execute the model-requested custom tool call and hand off its output."""
    llm_with_tool = get_llm().bind_tools(
        [search_knowledge_base],
        tool_choice="required",
    )
    response = llm_with_tool.invoke(
        [
            SystemMessage(content=RETRIEVER_SYSTEM_PROMPT),
            HumanMessage(content=state["query"]),
        ]
    )

    if not response.tool_calls:
        # Fail closed: without an auditable tool call there is no evidence.
        return {"snippets": []}

    tool_call = response.tool_calls[0]
    snippets = search_knowledge_base.invoke(tool_call["args"])
    return {"snippets": list(snippets)}
