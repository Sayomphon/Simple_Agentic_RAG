"""Streamlit UI for the Agentic RAG pipeline — presentation layer only.

This file renders what the core already produces; it contains no retrieval
or generation logic. It touches exactly two ``src`` entry points:

    - ``build_graph()``   -> the same agentic LangGraph the CLI runs
    - ``get_retriever()`` -> the same factory the tool layer uses, warmed
                             per mode so switching modes stays instant

Both are wrapped in ``st.cache_resource`` (the retriever keyed by mode) so
Streamlit's rerun-everything model never rebuilds an index or recompiles
the graph. Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import os
import time

import streamlit as st

from src.agents.reporter import NOT_FOUND_SENTENCE
from src.config import MODEL_NAME, SEARCH_MODE, TOP_K
from src.graph import build_graph
from src.retrievers import get_retriever

st.set_page_config(
    page_title="Agentic RAG Explorer",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- chrome ---
# One spacing scale (4/8/16/24/32px) drives every gap and margin; hierarchy
# comes from weight + space, not colour; raw evidence reads as data (mono,
# muted, hairline-ruled), the answer reads as prose. currentColor auto-adapts
# every derived tone to the active theme, so no dark-mode block is needed.
_CSS = """
<style>
:root {
  --s1:4px; --s2:8px; --s3:16px; --s4:24px; --s5:32px;
  --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
  --line:color-mix(in srgb, currentColor 15%, transparent);
  --muted:color-mix(in srgb, currentColor 58%, transparent);
  --faint:color-mix(in srgb, currentColor 40%, transparent);
}
.block-container, [data-testid="stMainBlockContainer"] { max-width:60rem; padding-top:var(--s5); }
#MainMenu, footer, [data-testid="stDecoration"] { visibility:hidden; }

.app-title { font-size:1.55rem; font-weight:700; letter-spacing:-.02em; margin:0; }
.app-sub { font-size:.92rem; line-height:1.5; color:var(--muted); max-width:58ch; margin:var(--s2) 0 var(--s5); }
.query-echo { font-size:1.15rem; font-weight:600; letter-spacing:-.01em; margin:0 0 var(--s3); }

.telemetry { display:flex; flex-wrap:wrap; gap:var(--s4); padding:var(--s3) 0; margin-bottom:var(--s2); border-top:1px solid var(--line); border-bottom:1px solid var(--line); }
.telemetry .t { display:flex; flex-direction:column; gap:var(--s1); }
.telemetry .k { font-size:.62rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }
.telemetry .v { font-family:var(--mono); font-size:.9rem; font-weight:500; }

.stage-head { display:flex; align-items:baseline; gap:var(--s2); flex-wrap:wrap; margin:var(--s5) 0 var(--s3); }
.stage-kicker { font-family:var(--mono); font-size:.68rem; font-weight:600; letter-spacing:.1em; color:var(--faint); }
.stage-title { font-size:1.05rem; font-weight:650; letter-spacing:-.01em; }
.stage-sub { font-size:.78rem; color:var(--muted); }

details.snip { border-left:2px solid var(--line); padding-left:var(--s3); }
details.snip + details.snip { margin-top:var(--s2); }
details.snip[open] { border-left-color:var(--muted); }
details.snip summary { display:flex; align-items:center; gap:var(--s2); padding:var(--s1) 0; cursor:pointer; list-style:none; }
details.snip summary::-webkit-details-marker { display:none; }
details.snip summary::after { content:"+"; font-family:var(--mono); color:var(--faint); }
details.snip[open] summary::after { content:"−"; }
.snip-rank { font-family:var(--mono); font-size:.72rem; color:var(--faint); }
.snip-title { font-size:.9rem; font-weight:550; }
.snip-file { font-family:var(--mono); font-size:.68rem; color:var(--faint); }
.snip-score { margin-left:auto; font-family:var(--mono); font-size:.72rem; color:var(--muted); }
.badge { font-family:var(--mono); font-size:.62rem; font-weight:600; letter-spacing:.03em; color:var(--muted); }
.badge-both { font-weight:700; color:inherit; }
details.snip pre { margin:var(--s2) 0 0; padding:0; font-family:var(--mono); font-size:.78rem; line-height:1.7; white-space:pre-wrap; color:var(--muted); }

.empty { padding:var(--s4) var(--s3); border:1px dashed var(--line); border-radius:8px; color:var(--muted); font-size:.9rem; }
.empty b { display:block; font-weight:650; margin-bottom:var(--s1); color:inherit; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------ static copy ---
_MODES = ("keyword", "semantic", "hybrid")
_MODE_CAPTIONS = (
    "BM25 lexical ranking — exact-term matching with a title boost. "
    "Deterministic, offline, no API cost.",
    "OpenAI embeddings + cosine similarity — matches meaning even when the "
    "wording differs from the handbook.",
    "Runs both retrievers and merges ranks with Reciprocal Rank Fusion. "
    "Badges show which side found each snippet.",
)
# Expected ScoredChunk/retriever ``source`` labels per mode, used only to
# detect the factory's documented keyword fallback and surface it in the UI.
_MODE_SOURCE = {"keyword": "bm25", "semantic": "dense", "hybrid": "hybrid"}

_EXAMPLES = (
    ("International travel", "What is the policy on international travel?"),
    ("Work from home", "Can I work from home every day?"),
    ("Mileage claim", "How much can I claim when I use my own car for a client visit?"),
    ("Thai query", "ลาบวชได้กี่วัน และต้องแจ้งล่วงหน้าอย่างไร?"),
    ("Not in KB", "What is the CEO's salary?"),
    ("Greeting", "Hello! What can you do?"),
)

# ------------------------------------------------------- cached core hooks ---
@st.cache_resource(show_spinner=False)
def load_graph():
    """Compile the agentic LangGraph once per server process."""
    return build_graph()


@st.cache_resource(show_spinner=False)
def load_retriever(mode: str):
    """Warm the retriever for ``mode`` once — cache key includes the mode,
    so switching modes builds each index at most once and reuses it after."""
    return get_retriever(mode)


# ------------------------------------------------------------- renderers ---
def _stage_head(number: int, title: str, sub: str) -> None:
    st.markdown(
        f'<div class="stage-head"><span class="stage-kicker">STAGE {number}</span>'
        f'<span class="stage-title">{title}</span>'
        f'<span class="stage-sub">{sub}</span></div>',
        unsafe_allow_html=True,
    )


def _badge(source: str) -> str:
    if "+" in source:
        return '<span class="badge badge-both">BM25 + EMBEDDINGS</span>'
    label = {"bm25": "BM25", "dense": "EMBEDDINGS"}.get(source, html.escape(source).upper())
    return f'<span class="badge">{label}</span>'


def _snippet_cards(hits) -> str:
    cards = []
    for rank, hit in enumerate(hits, start=1):
        source_file = getattr(hit, "source_file", "")
        file_tag = (
            f'<span class="snip-file">{html.escape(source_file)}</span>'
            if source_file
            else ""
        )
        cards.append(
            f'<details class="snip"{" open" if rank == 1 else ""}>'
            f'<summary><span class="snip-rank">{rank:02d}</span>'
            f'<span class="snip-title">{html.escape(hit.title)}</span>'
            f"{file_tag}"
            f"{_badge(hit.source)}"
            f'<span class="snip-score">score {hit.score:.4f}</span></summary>'
            f"<pre>{html.escape(hit.text)}</pre></details>"
        )
    return "".join(cards)


def _telemetry(run: dict) -> str:
    timings = run["timings"]
    total = sum(timings.values())
    if run.get("route") == "direct":
        items = (
            ("route", "direct"),
            ("model", run["model"]),
            ("respond", f"{timings.get('synthesis', 0):.2f}s"),
            ("total", f"{total:.2f}s"),
        )
    else:
        items = (
            ("route", run.get("route", "kb_query")),
            ("mode", run["mode"]),
            ("model", run["model"]),
            ("top-k", str(run["top_k"])),
            ("attempts", str(len(run.get("search_attempts") or []) or 1)),
            ("snippets", str(len(run["snippets"]))),
            ("stage 1 · retrieve", f"{timings.get('retrieval', 0):.2f}s"),
            ("stage 2 · synthesize", f"{timings.get('synthesis', 0):.2f}s"),
            ("total", f"{total:.2f}s"),
        )
    spans = "".join(
        f'<div class="t"><span class="k">{k}</span><span class="v">{html.escape(v)}</span></div>'
        for k, v in items
    )
    return f'<div class="telemetry">{spans}</div>'


def render_run(run: dict) -> None:
    """Display one completed pipeline run: telemetry, evidence, answer."""
    st.markdown(f'<p class="query-echo">“{html.escape(run["query"])}”</p>', unsafe_allow_html=True)
    st.markdown(_telemetry(run), unsafe_allow_html=True)

    if run.get("route") == "direct":
        _stage_head(1, "Direct Responder", "router verdict: small talk / meta — the knowledge base was never touched")
        st.markdown(run["report"])
        return

    _stage_head(1, "Data Retriever Agent", "forced tool call → ranked evidence from the knowledge base")
    attempts = run.get("search_attempts") or []
    if len(attempts) > 1:
        # Every attempt before the last found nothing — the rewriter then
        # proposed the next query. Show the whole agentic trail.
        trail = "".join(
            f'<div>attempt {i}: “{html.escape(attempt)}” → '
            f'{len(run["hits"]) if i == len(attempts) else 0} result(s)</div>'
            for i, attempt in enumerate(attempts, start=1)
        )
        st.caption(f"Search attempts (query rewritten on empty results):{trail}", unsafe_allow_html=True)
    elif run["search_query"] != run["query"]:
        st.caption(f'Agent reformulated the search as: “{run["search_query"]}”')
    if run["hits"]:
        st.markdown(_snippet_cards(run["hits"]), unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="empty"><b>No snippets cleared the relevance gates</b>'
            f"{run['mode']} mode found nothing relevant to this query, so the "
            "evidence set handed to Stage 2 was empty — by design, the "
            "generator then falls back deterministically instead of guessing.</div>",
            unsafe_allow_html=True,
        )

    _stage_head(2, "Report Generator Agent", "grounded synthesis — uses the snippets above and nothing else")
    if run["report"].strip() == NOT_FOUND_SENTENCE:
        st.markdown(f"*{run['report']}*")
        st.caption(
            "Deterministic fallback: zero snippets were handed off, so this "
            "fixed sentence was returned without an LLM call."
            if not run["hits"]
            else "Prompt guardrail: the generator judged the retrieved "
            "snippets insufficient to answer this query."
        )
    else:
        st.markdown(run["report"])


# ------------------------------------------------------------ run driver ---
def execute(query: str, mode: str, top_k: int) -> dict | None:
    """Stream one query through the graph, narrating each stage live."""
    graph = load_graph()
    with st.spinner(f"Preparing the {mode} index…"):
        retriever = load_retriever(mode)
    if getattr(retriever, "SOURCE", "") != _MODE_SOURCE[mode]:
        st.warning(
            f"The **{mode}** index could not be built (embeddings unavailable), "
            "so retrieval fell back to **keyword / BM25** for this session. "
            "Check `OPENAI_API_KEY` and network access, then restart the app."
        )

    run = {
        "query": query, "mode": mode, "top_k": top_k, "model": MODEL_NAME,
        "snippets": [], "hits": [], "report": "", "search_query": query,
        "route": "kb_query", "search_attempts": [], "timings": {},
    }
    stage_router = st.status("**Router** — does this query need the knowledge base?", state="running")
    stage1 = None
    stage2 = None
    started = time.perf_counter()
    try:
        for update in graph.stream(
            {"query": query, "snippets": [], "report": "",
             "search_mode": mode, "top_k": top_k},
            stream_mode="updates",
        ):
            # The retry loop may fire data_retriever / query_rewriter
            # several times; retrieval time accumulates across attempts.
            if "router" in update:
                run["timings"]["routing"] = time.perf_counter() - started
                run.update(update["router"] or {})
                started = time.perf_counter()
                if run["route"] == "direct":
                    stage_router.update(
                        label="**Router** — direct: small talk, skipping the knowledge base",
                        state="complete",
                    )
                else:
                    stage_router.update(
                        label="**Router** — kb_query: retrieval required",
                        state="complete",
                    )
                    stage1 = st.status("**Stage 1 · Data Retriever** — choosing a search query and retrieving…", state="running")
            elif "direct_responder" in update:
                run["timings"]["synthesis"] = time.perf_counter() - started
                run.update(update["direct_responder"] or {})
            elif "data_retriever" in update:
                if stage1 is None:
                    stage1 = st.status("**Stage 1 · Data Retriever**", state="running")
                run["timings"]["retrieval"] = (
                    run["timings"].get("retrieval", 0) + time.perf_counter() - started
                )
                run.update(update["data_retriever"] or {})
                started = time.perf_counter()
                attempts = run.get("search_attempts") or []
                if run["snippets"]:
                    stage1.update(
                        label=(f"**Stage 1 · Data Retriever** — {len(run['snippets'])} "
                               f"snippet(s) in {len(attempts)} attempt(s), "
                               f"{run['timings']['retrieval']:.1f}s"),
                        state="complete",
                    )
                    stage2 = st.status("**Stage 2 · Report Generator** — synthesizing the grounded answer…", state="running")
                else:
                    stage1.update(
                        label=(f"**Stage 1 · Data Retriever** — attempt "
                               f"{len(attempts)} found nothing, rewriting the query…"),
                        state="running",
                    )
            elif "query_rewriter" in update:
                run.update(update["query_rewriter"] or {})
            elif "report_generator" in update:
                run["timings"]["synthesis"] = time.perf_counter() - started
                attempts = run.get("search_attempts") or []
                if not run["snippets"]:
                    # All attempts exhausted: close stage 1 as a clean miss.
                    stage1.update(
                        label=(f"**Stage 1 · Data Retriever** — 0 snippets after "
                               f"{len(attempts)} attempt(s), "
                               f"{run['timings'].get('retrieval', 0):.1f}s"),
                        state="complete",
                    )
                run.update(update["report_generator"] or {})
                if stage2 is None:
                    stage2 = st.status("**Stage 2 · Report Generator**", state="running")
                stage2.update(
                    label=(f"**Stage 2 · Report Generator** — answered in "
                           f"{run['timings']['synthesis']:.1f}s"),
                    state="complete",
                )
    except Exception as exc:  # noqa: BLE001 — surface any provider error readably
        for status in (stage_router, stage1, stage2):
            if status is not None:
                status.update(state="error")
        st.error(f"**Pipeline failed** ({type(exc).__name__}): {str(exc)[:400]}")
        return None
    return run


# ------------------------------------------------------------------ page ---
state = st.session_state
state.setdefault("runs", [])
state.setdefault("active_run", None)

with st.sidebar:
    st.markdown("#### Retrieval settings")
    mode = st.radio(
        "Search mode",
        _MODES,
        index=_MODES.index(SEARCH_MODE) if SEARCH_MODE in _MODES else 0,
        format_func=str.capitalize,
        captions=_MODE_CAPTIONS,
    )
    top_k = st.slider("Top-k snippets", min_value=1, max_value=8, value=TOP_K,
                      help="Maximum snippets the Data Retriever may hand to the generator.")
    st.caption(
        "Settings apply to the next query. Indexes are cached per mode — the "
        "first semantic/hybrid query builds (or loads) the embedding index; "
        "after that, switching modes is instant."
    )

st.markdown('<p class="app-title">Agentic RAG Explorer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-sub">Agentic LangGraph pipeline over the Siam Innovate employee '
    'handbook — a <b>Router</b> decides whether the knowledge base is needed, a '
    '<b>Data Retriever</b> is forced through a search tool (rewriting the query and '
    'retrying when a search comes back empty), and a <b>Report Generator</b> answers '
    'from the retrieved evidence only.</p>',
    unsafe_allow_html=True,
)

if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "**`OPENAI_API_KEY` is not set — both agents need the OpenAI API.**\n\n"
        "1. `cp .env.example .env`\n"
        "2. Put your key in `.env` (`OPENAI_API_KEY=sk-...`)\n"
        "3. Restart: `streamlit run app.py`"
    )
    st.stop()

with st.form("query_form", border=False):
    col_input, col_button = st.columns([5, 1], vertical_alignment="bottom")
    typed = col_input.text_input(
        "Question", placeholder="Ask the employee handbook anything…",
        label_visibility="collapsed",
    )
    submitted = col_button.form_submit_button("Search", type="primary", width="stretch")

query_to_run = typed.strip() if submitted and typed.strip() else None
example_cols = st.columns(len(_EXAMPLES))
for col, (label, example_query) in zip(example_cols, _EXAMPLES):
    if col.button(label, key=f"ex_{label}", help=example_query, width="stretch"):
        query_to_run = example_query

if submitted and not typed.strip() and query_to_run is None:
    st.warning("Type a question or pick an example.")

if query_to_run:
    finished = execute(query_to_run, mode, top_k)
    if finished is not None:
        state.runs.append(finished)
        state.active_run = len(state.runs) - 1

with st.sidebar:
    if state.runs:
        st.divider()
        st.markdown("#### History")
        for i in range(len(state.runs) - 1, -1, -1):
            past = state.runs[i]
            if st.button(
                past["query"][:48] + ("…" if len(past["query"]) > 48 else ""),
                key=f"hist_{i}", width="stretch",
                help=f"{past['query']}  ·  {past['mode']}, top_k={past['top_k']}",
            ):
                state.active_run = i

if state.runs and state.active_run is not None:
    render_run(state.runs[state.active_run])
else:
    st.markdown(
        '<div class="empty"><b>No queries yet</b>Ask a question or click an example '
        "above — each run shows the full pipeline: retrieved evidence with scores "
        "and provenance first, then the grounded answer.</div>",
        unsafe_allow_html=True,
    )
