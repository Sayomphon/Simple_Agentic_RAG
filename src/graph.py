"""Two-node sequential LangGraph orchestration for the submission."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import NotRequired, TypedDict

from src.agents.reporter import generator_node
from src.agents.retriever import retriever_node
from src.retrievers.base import SearchTelemetry


class PipelineState(TypedDict):
    """The complete and intentionally minimal agent handoff contract."""

    query: str
    snippets: list[str]
    report: str
    retrieval_telemetry: NotRequired[list[SearchTelemetry]]


def build_graph() -> CompiledStateGraph:
    """Compile the fixed ``Retriever -> Report Generator`` workflow."""
    builder = StateGraph(PipelineState)
    builder.add_node("data_retriever", retriever_node)
    builder.add_node("report_generator", generator_node)
    builder.add_edge(START, "data_retriever")
    builder.add_edge("data_retriever", "report_generator")
    builder.add_edge("report_generator", END)
    return builder.compile()
