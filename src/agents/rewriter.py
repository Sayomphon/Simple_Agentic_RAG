"""Query Rewriter agent: proposes a fresh search query after a failed attempt.

Runs only when the Data Retriever hands off zero snippets and the attempt
budget (``MAX_SEARCH_ATTEMPTS``) is not exhausted. The same structural
guardrail as the retriever applies: ``tool_choice="required"`` forces the
model into a tool call whose single argument IS the rewritten query, so
the output is parsed deterministically — never free text, never an answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.agents import get_llm

if TYPE_CHECKING:
    from src.graph import PipelineState

REWRITER_SYSTEM_PROMPT = """\
You are the Query Rewriter agent in a retrieval pipeline. A knowledge-base
search returned ZERO results for every query tried so far. Your ONLY job
is to propose ONE new search query for the next attempt.

Rules:
- Call the `propose_search_query` tool exactly ONCE with the new query.
- The new query MUST differ meaningfully from every previously tried
  query: use employee-handbook vocabulary (policy, process, procedure,
  official HR terms), synonyms, or a wider/narrower phrasing of the SAME
  topic.
- PRESERVE the user's original intent exactly. Never change the subject,
  never generalize so far that the query could match unrelated sections.
  If the user asks about X, the new query must still be about X.
- Rewrite the WORDING, not the REQUEST. If the user asks for a specific
  piece of data or a fact, the new query must still ask for that same
  data — never turn it into a query about policies, procedures, or
  guidelines ABOUT that data.
- If the user's question is not in English, write the new query in
  concise English handbook vocabulary.
- NEVER answer the user's question. NEVER add commentary. The tool call
  is your only output.
"""


@tool
def propose_search_query(query: str) -> str:
    """Propose one new knowledge-base search query for the next attempt.

    Args:
        query: The rewritten search query — concise English handbook
            vocabulary, same intent as the user's question, different
            wording from every query already tried.
    """
    return query


def rewriter_node(state: PipelineState) -> dict[str, str]:
    """Produce a rewritten search query from the user query + failed attempts.

    Args:
        state: Pipeline state with the user ``query`` and the
            ``search_attempts`` that all returned zero snippets.

    Returns:
        Partial state update with ``rewritten_query`` for the retriever's
        next attempt. Falls back to the original user query when the
        model misbehaves — the attempt counter still advances, so the
        loop stays bounded either way.
    """
    attempts = state.get("search_attempts", [])
    tried = "\n".join(f"- {attempt}" for attempt in attempts) or "- (none)"
    llm_with_tool = get_llm().bind_tools(
        [propose_search_query],
        tool_choice="required",  # structural guardrail: output must be a query
    )
    ai_msg = llm_with_tool.invoke(
        [
            SystemMessage(content=REWRITER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"User question: {state['query']}\n\n"
                    f"Queries already tried (all returned zero results):\n{tried}"
                )
            ),
        ]
    )
    if not ai_msg.tool_calls:
        return {"rewritten_query": state["query"]}
    rewritten = str(ai_msg.tool_calls[0]["args"].get("query", "")).strip()
    return {"rewritten_query": rewritten or state["query"]}
