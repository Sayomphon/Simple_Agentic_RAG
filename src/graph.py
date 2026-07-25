"""LangGraph wiring for the agentic retrieval pipeline.

Orchestration pattern: handoff through shared state, with two agentic
decision points. A Router first classifies the query — pure small talk
is answered directly; anything informational takes the retrieval path.
There, the Data Retriever writes ``snippets`` into ``PipelineState``;
when snippets were found (or the attempt budget is spent) the state
flows to the Report Generator, which writes the final ``report``. When
an attempt finds nothing, the Query Rewriter proposes a new search
query and the retriever tries again — up to ``MAX_SEARCH_ATTEMPTS``
total attempts.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import NotRequired, TypedDict

from src.agents.reporter import generator_node
from src.agents.retriever import retriever_node
from src.agents.rewriter import rewriter_node
from src.agents.router import ROUTE_DIRECT, direct_responder_node, router_node
from src.config import MAX_SEARCH_ATTEMPTS
from src.retrievers import ScoredChunk


class PipelineState(TypedDict):
    """Shared state carried across the pipeline — the whole data flow.

    The three required fields are the core contract (CLI and agents use
    only these). The ``NotRequired`` fields are optional per-run overrides
    and retrieval metadata consumed by presentation layers (Streamlit UI);
    they change nothing when absent.
    """

    query: str            # user question (input)
    snippets: list[str]   # Data Retriever output -> handoff to the Generator
    report: str           # Report Generator output (final answer)

    search_mode: NotRequired[str]   # per-run retrieval mode; config default if absent
    top_k: NotRequired[int]         # per-run result cap; config default if absent
    search_query: NotRequired[str]  # query the retriever agent actually searched
    hits: NotRequired[list[ScoredChunk]]  # scored snippets (title/score/source) for UIs

    # Agentic decision fields (written by router/retriever/rewriter, read
    # by the edge functions below and by presentation layers).
    route: NotRequired[str]                  # router verdict: kb_query | direct
    search_attempts: NotRequired[list[str]]  # every query tried, in order
    rewritten_query: NotRequired[str]        # rewriter's query for the next attempt


def _after_routing(state: PipelineState) -> str:
    """Send small talk to the direct responder; everything else retrieves.

    The fail-safe default is retrieval: only an explicit ``direct``
    verdict skips the knowledge base.
    """
    if state.get("route") == ROUTE_DIRECT:
        return "direct_responder"
    return "data_retriever"


def _after_retrieval(state: PipelineState) -> str:
    """Decide the next node after a retrieval attempt.

    Snippets found -> synthesize. Attempt budget spent -> synthesize
    (the generator's deterministic not-found fallback fires on empty
    snippets). Otherwise -> rewrite the query and search again.
    """
    if state["snippets"]:
        return "report_generator"
    if len(state.get("search_attempts", [])) >= MAX_SEARCH_ATTEMPTS:
        return "report_generator"
    return "query_rewriter"


def build_graph() -> CompiledStateGraph:
    """Compile the pipeline: Router -> Retriever -> (retry loop) -> Generator."""
    builder = StateGraph(PipelineState)
    builder.add_node("router", router_node)
    builder.add_node("direct_responder", direct_responder_node)
    builder.add_node("data_retriever", retriever_node)
    builder.add_node("query_rewriter", rewriter_node)
    builder.add_node("report_generator", generator_node)
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        _after_routing,
        ["data_retriever", "direct_responder"],
    )
    # <-- agentic handoff: found snippets travel to the generator via
    #     shared state; an empty attempt loops back through the rewriter
    #     until the MAX_SEARCH_ATTEMPTS budget is spent.
    builder.add_conditional_edges(
        "data_retriever",
        _after_retrieval,
        ["report_generator", "query_rewriter"],
    )
    builder.add_edge("query_rewriter", "data_retriever")
    builder.add_edge("report_generator", END)
    builder.add_edge("direct_responder", END)
    return builder.compile()
