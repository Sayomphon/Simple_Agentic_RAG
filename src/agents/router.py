"""Router agent: decides whether a query needs the knowledge base at all.

Two routes:
    - ``kb_query`` — anything that asks for information (policy, product,
      company facts, HR questions, in any language) -> full retrieval path.
    - ``direct``   — pure small talk / greetings / meta questions about
      the assistant itself -> a short canned-style reply, no retrieval.

Fail-safe rule: when in doubt, route to ``kb_query``. The retrieval path
has a deterministic not-found guardrail behind it; the direct path has
only a prompt, so it must never see factual questions. The classifier is
a forced tool call (same structural pattern as the other agents), so the
route is parsed deterministically — and any parse failure also falls
back to ``kb_query``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from src.agents import get_llm

if TYPE_CHECKING:
    from src.graph import PipelineState

ROUTE_KB = "kb_query"
ROUTE_DIRECT = "direct"

ROUTER_SYSTEM_PROMPT = f"""\
You are the Router agent for a company knowledge-base assistant. Classify
the user's message into exactly one route by calling the `choose_route`
tool once.

Routes:
- "{ROUTE_KB}": ANY request for information — company policies, HR topics,
  products, benefits, processes, facts, or anything that could be answered
  from documents. Questions in any language. This includes questions the
  knowledge base probably cannot answer — retrieval will handle that.
- "{ROUTE_DIRECT}": ONLY pure small talk (greetings, thanks, goodbyes) or
  meta questions about the assistant itself ("who are you", "what can you
  do"). Nothing that asks for real-world or company information.

When in doubt, ALWAYS choose "{ROUTE_KB}".
"""

DIRECT_RESPONDER_SYSTEM_PROMPT = """\
You are the front desk of a company knowledge-base assistant for the Siam
Innovate employee handbook. The router classified the user's message as
small talk or a meta question — NOT an information request.

Rules:
- Reply in 1-2 short, friendly sentences.
- Invite the user to ask about handbook topics (leave, travel, expenses,
  benefits, IT security, HR processes, products).
- NEVER answer factual or company questions from your own knowledge. If
  the message actually asks for information, reply exactly:
  "I could not find this information in the knowledge base."
- Match the user's language (Thai in -> Thai out).
"""


@tool
def choose_route(route: str) -> str:
    """Select the processing route for the user's message.

    Args:
        route: Either "kb_query" (any information request) or "direct"
            (pure small talk / meta questions about the assistant).
    """
    return route


def router_node(state: PipelineState) -> dict[str, str]:
    """Classify the query into a route, failing safe to retrieval.

    Args:
        state: Pipeline state containing the user ``query``.

    Returns:
        Partial state update with ``route``.
    """
    llm_with_tool = get_llm().bind_tools(
        [choose_route],
        tool_choice="required",  # structural guardrail: output must be a route
    )
    ai_msg = llm_with_tool.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=state["query"]),
        ]
    )
    if not ai_msg.tool_calls:
        return {"route": ROUTE_KB}
    route = str(ai_msg.tool_calls[0]["args"].get("route", "")).strip().lower()
    # Fail safe: anything that is not exactly "direct" takes the
    # retrieval path, which has the not-found guardrail behind it.
    return {"route": ROUTE_DIRECT if route == ROUTE_DIRECT else ROUTE_KB}


def direct_responder_node(state: PipelineState) -> dict[str, str]:
    """Answer small talk directly — no retrieval, no knowledge claims.

    Args:
        state: Pipeline state containing the user ``query``.

    Returns:
        Partial state update with the final ``report``.
    """
    msg = get_llm().invoke(
        [
            SystemMessage(content=DIRECT_RESPONDER_SYSTEM_PROMPT),
            HumanMessage(content=state["query"]),
        ]
    )
    return {"report": str(msg.content)}
