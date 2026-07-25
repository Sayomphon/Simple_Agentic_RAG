"""Report Generator Agent: grounded synthesis from retrieved snippets only.

Guardrails:
    - Prompt layer: answer ONLY from snippets, merge duplicates, and use a
      fixed not-found sentence when the snippets are insufficient.
    - Deterministic layer: when the retriever hands off zero snippets, the
      node returns the not-found sentence directly — no LLM call, so the
      fallback text is guaranteed byte-exact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents import get_llm

if TYPE_CHECKING:
    from src.graph import PipelineState

NOT_FOUND_SENTENCE = "I could not find this information in the knowledge base."

REPORTER_SYSTEM_PROMPT = f"""\
You are the Report Generator Agent, an expert writer and synthesizer.
Write a clear, well-structured answer to the user's query using ONLY the
provided snippets from the company knowledge base.

Rules:
- Use only facts stated in the supplied snippets.
- Never add outside knowledge, assumptions, or invented details.
- Combine complementary facts and state duplicate facts only once.
- Each snippet starts with `--- Section Title ---`. When useful, cite that
  exact title as `[Section Title]`; never invent a source name.
- If the snippets do not contain the information needed to answer the
  query, reply with exactly this sentence and nothing else — no
  citation: "{NOT_FOUND_SENTENCE}"
- Snippets that merely relate to the query's topic do not count as an
  answer. If the user asks for specific data or a specific fact and the
  snippets only describe rules or processes about that topic without
  stating the requested information itself, use the not-found sentence.
- Keep the answer concise, cohesive, non-redundant, and easy to scan.
"""


def generator_node(state: PipelineState) -> dict[str, str]:
    """Synthesize the final grounded answer from the handed-off snippets.

    Args:
        state: Pipeline state containing ``query`` and ``snippets``.

    Returns:
        Partial state update with the final ``report``.
    """
    if not state["snippets"]:
        # Deterministic fallback: nothing retrieved, nothing to synthesize.
        return {"report": NOT_FOUND_SENTENCE}

    snippets_text = "\n\n".join(state["snippets"])
    msg = get_llm().invoke(
        [
            SystemMessage(content=REPORTER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"User query: {state['query']}\n\nSnippets:\n{snippets_text}"
            ),
        ]
    )
    return {"report": str(msg.content)}
