# Simple Agentic RAG — Web UI

A single-page front end for the two-agent pipeline. It renders the workflow as
five sequential stages so the agent handoff is visible:

```text
User Query → Data Retriever Agent → Retrieved Evidence → Report Generator Agent → Final Answer
```

The retrieved evidence is shown **raw and unmodified**, separately from the
synthesised answer — that separation is the point of the demo, since it lets a
reviewer check the Generator's input against `knowledge_base.txt` directly.

## Run it

No build step, no dependencies, no install.

```bash
open web/index.html          # macOS — double-clicking works too
```

The UI starts on bundled mock data, so every state is demonstrable offline. For
live mode (which needs `fetch`), serve the folder over HTTP:

```bash
python3 -m http.server 8000  # then open http://localhost:8000/web/
```

## Files

| File | Responsibility |
|---|---|
| `index.html` | Markup for the five stages, query form, and empty/error states |
| `styles.css` | Design tokens, layout, badges, skeletons, light + dark themes |
| `api.js` | **The only backend seam.** Validates result and telemetry payloads |
| `mock-data.js` | Offline fixtures: the 10 real KB sections + a matching gate |
| `app.js` | One state object, one render pass per change |

State is a single plain object in `app.js`; there is no framework, store, or
router. Adding a stage means adding an `<li class="step">` and a renderer.

## Connect the Python backend

Everything the UI needs is behind `RagApi.runWorkflow`, so wiring it up is two
steps:

**1. Expose the graph over HTTP.** The response mirrors `PipelineState` exactly:

```python
# server.py
from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel
from src.graph import build_graph

app = FastAPI()
graph = build_graph()

class Query(BaseModel):
    query: str

@app.post("/api/query")
def run(payload: Query) -> dict:
    result = graph.invoke({"query": payload.query, "snippets": [], "report": ""})
    return {
        "query": payload.query,
        "snippets": result["snippets"],   # raw, exactly as the tool returned them
        "report": result["report"],
        "retrieval_telemetry": [
            asdict(item)
            for item in result.get("retrieval_telemetry", [])
        ],
    }
```

**2. Flip the source pill** in the header to **Live backend** (persisted in
`localStorage`), or change the default in `api.js`:

```js
var CONFIG = {
  endpoint: "/api/query",   // absolute URL if the API is on another origin
  model: "gpt-5-mini",      // metadata display only
  timeoutMs: 60000
};
```

Serving the UI from a different origin than the API needs CORS enabled on the
Python side.

### Contract notes

- `snippets` must stay **byte-identical** to `search_knowledge_base` output. The
  evidence panel exists to prove the Retriever did not rewrite anything.
- `retrieval_telemetry` is optional and additive. Each attempt contains
  `mode`, `query`, `latency_ms`, `empty_reason`, and per-snippet
  `title`/`score`/`method`/`detail`. `api.js` validates and bounds these fields
  before rendering them.
- Telemetry belongs in the API/UI response only. Never concatenate scores,
  methods, latency, or diagnostic detail into the Report Generator prompt.
- An empty `snippets` array is a **valid result, not an error**. The UI shows the
  *No evidence found* badge and the exact not-found sentence, matching the
  Reporter's deterministic short-circuit.
- `NOT_FOUND_SENTENCE` in `api.js` must match `src/agents/reporter.py`.
- Failures (protocol violation, malformed corpus, provider error) should return a
  non-2xx with `{"error": "..."}`; the UI surfaces that message and marks the
  unfinished stages as *Failed*.

### Per-node progress

The pipeline answers in one round trip, so live-mode stage transitions are
derived from that single response. For true per-node progress, expose an SSE
endpoint over LangGraph's `graph.stream()` and emit from `runLive` in `api.js` as
each node event arrives — no other file changes.

## States

Each sample chip demonstrates a different path:

| Chip | State |
|---|---|
| *What is the policy on international travel?* | Multi-section retrieval → grounded synthesis with citations |
| *Can I work remotely?* | Two complementary sections merged into one answer |
| *What is the refund policy?* | Not-found — no evidence, deterministic fallback, no LLM call |

Empty state shows before the first query. Loading shows skeletons with per-stage
`Running` badges. Errors surface inline with a retry.

## Interaction details

- **Cancel.** The Run button stays enabled while busy and relabels to *Cancel*
  (Enter/⌘+Enter follow the same rule); the sticky compact bar also gets a
  Cancel action once the query card scrolls out of view. Cancelling aborts the
  in-flight `fetch` in live mode and rejects the mock's promise chain with
  `error.name === "CancelledError"`, which `app.js` treats as a silent return
  to the empty state rather than an error. Pass `signal` (an `AbortSignal`) to
  `runWorkflow` to support this from a different UI.
- **Collapsible steps.** Each stage header is a `.step-toggle` button
  (`aria-expanded` + `aria-controls`) that hides its `.step-body`. Purely
  presentational — it never touches pipeline state.
- **Evidence clamp.** `#evidence-clamp` caps the evidence panel at 480px and
  fades out once a query's combined snippets exceed that height (the
  international-travel sample query is the built-in example); `#evidence-expand`
  removes the cap. Small result sets never clamp.
- **Compact bar.** An `IntersectionObserver` on `#query-card` shows `#compact-bar`
  once the card scrolls away and a run has happened, mirroring the run's status,
  query, and elapsed timer.

## About the mock

`mock-data.js` embeds the 10 sections from `knowledge_base.txt` verbatim and
mirrors the real retrieval gate (title weight 1.5, body 1.0, keep ≥ 60% of the
best score, plus the two-term sibling rule) closely enough to return the same
sections as the Python tool for the golden queries. It also emits the same
telemetry shape so score, method, latency, and empty-reason states remain
demonstrable offline. It is a fixture for demoing the UI, not a second
implementation — delete it once the backend is wired up.
