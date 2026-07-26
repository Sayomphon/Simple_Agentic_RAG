# Simple Agentic RAG

> An auditable two-agent RAG pipeline built with LangGraph and OpenAI.
> It retrieves raw evidence from a local knowledge base, hands that evidence
> between agents through typed shared state, and produces a grounded answer—or
> a deterministic not-found response when the knowledge base has no answer.

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-1C3C3C)
![Tests](https://img.shields.io/badge/tests-99%20passing%20%7C%204%20live%20skipped-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

This repository is a deliberately small implementation of the
AI Engineer Programming Test. It focuses
on the assignment's core engineering concerns: clear agent responsibilities, a
custom Retrieval-Augmented Generation (RAG) tool, explicit orchestration,
inspectable evidence handoff, grounded generation, and reproducible offline
tests.

## What the system does

A user asks a question about the fictional **Siam Innovate** knowledge base.
The system then:

1. sends the question to a **Data Retriever Agent**;
2. forces that agent to request the custom `search_knowledge_base` tool;
3. retrieves relevant sections from `knowledge_base.txt`;
4. passes the raw sections to a **Report Generator Agent** through LangGraph
   state; and
5. produces a concise answer based only on those sections.

Both front ends — the CLI and the bundled web UI — display the query, retrieved
evidence, and final answer as separate stages, making the RAG handoff easy to
audit.

### Key properties

- **Two specialized agents:** retrieval and answer synthesis have separate
  responsibilities.
- **Custom local RAG tool:** retrieval reads a plain UTF-8 text file and needs
  no vector database.
- **Deterministic retrieval:** the same query and knowledge base produce the
  same ordered snippets.
- **Raw evidence handoff:** the Retriever does not summarize or rewrite the
  sections it returns; a mixed question may be split into up to three
  sub-queries, but the handoff always contains at least everything a single
  search over the original query returns.
- **Grounded generation:** the Generator is instructed to use only the
  retrieved evidence, snippets are passed inside a declared `<evidence>`
  data boundary, and every `[Section Title]` citation is validated at
  runtime against the sections actually handed off.
- **Explicit failure semantics:** a valid empty search returns an exact
  not-found sentence, while protocol, corpus, and model-output failures surface
  as errors.
- **Streaming CLI:** retrieved evidence prints as soon as retrieval
  finishes and answer tokens render as they arrive, while the final state
  report remains the source of truth.
- **Measured quality:** retrieval and answer quality are numbers produced
  by runners in this repository, reported for both a calibration set and a
  held-out set (see [Evaluation](#evaluation)).
- **Resilient CLI boundary:** one failed interactive query is reported safely
  without ending the session.
- **Inspectable web UI:** a dependency-free single-page front end renders the
  same five stages, keeping raw evidence visually separate from the synthesised
  answer.
- **Offline test suite:** retrieval and orchestration tests require neither an
  API key nor network access.

## Architecture

```mermaid
flowchart LR
    U["User query"] --> R

    subgraph LG["LangGraph — fixed sequential workflow"]
        R["Data Retriever Agent<br/>forced tool request"]
        H["Shared state<br/>snippets: list[str]"]
        G["Report Generator Agent<br/>grounded synthesis"]
        R -->|"raw snippets"| H
        H --> G
    end

    R -->|"search_knowledge_base(query)"| T["Custom retrieval tool"]
    T -->|"read and score sections"| KB[("knowledge_base.txt<br/>10 sections")]
    KB -->|"raw matching sections"| T
    T --> R

    G -->|"evidence available"| A["Grounded answer"]
    G -.->|"no evidence · no LLM call"| N["Deterministic not-found"]
```

The graph topology has no router, retry loop, or conditional retrieval path:

```text
START -> data_retriever -> report_generator -> END
```

Agent-to-agent handoff uses one explicit state contract:

```python
class PipelineState(TypedDict):
    query: str            # original user question
    snippets: list[str]   # Retriever output -> Generator input
    report: str           # final answer
```

### Agent responsibilities

#### 1. Data Retriever Agent

- rejects empty, whitespace-only, and over-length (> 2,000 characters)
  queries before any LLM call;
- binds only the `search_knowledge_base` tool;
- uses `tool_choice="required"` so the model must request a tool call;
- validates the model's plan: between 1 and 3 correctly named tool calls,
  each with a non-empty string query — a single-topic question stays one
  call with the original query, a mixed question may split into focused
  sub-queries;
- executes every search itself: the original graph-state query always runs
  first as a deterministic baseline, then each sub-query's new sections are
  appended in call order with exact-text deduplication — so the handoff is
  provably a superset of `search(original_query)` no matter how the model
  decomposes;
- writes that raw `list[str]` to `state["snippets"]`; and
- never produces the user-facing answer.

#### 2. Report Generator Agent

- receives only the user query and retrieved snippets;
- has no tools;
- reads snippets inside an `<evidence>` block its prompt declares to be
  data, never instructions (knowledge-base text is untrusted);
- combines complementary facts and removes repetition;
- is instructed not to add outside knowledge or assumptions;
- answers the supported parts of a mixed question and states plainly which
  part the knowledge base does not cover;
- has every `[Section Title]` citation checked in code against the
  handed-off snippets — an invented citation raises an error instead of
  shipping a fabricated source; and
- short-circuits empty evidence to:

```text
I could not find this information in the knowledge base.
```

## Retrieval design

`knowledge_base.txt` contains 10 clearly delimited sections across workplace
policies, international travel, expenses, a payment product, and customer
support. Each section begins with a machine-readable heading:

```text
--- Section Title ---
Section content...
```

The custom tool in `src/tools/retrieval.py` uses a transparent normalized
weighted lexical pipeline:

1. read `knowledge_base.txt` as UTF-8 (parsed sections are cached per file
   identity — path, `mtime_ns`, size — so an edited file invalidates
   naturally);
2. reject empty files, missing section headings, and sections without bodies;
3. split the document at section headings;
4. normalize reviewed phrase aliases such as `work from home` → `remote work`
   and `per diem` → `daily allowance`;
5. canonicalize reviewed derivational variants and synonyms such as
   `lodging` → `hotel`, `reimbursing` → `reimbursement`, and
   `vacation` → `leave`;
6. remove query framing, English stopwords, and broad enterprise terms from
   query topic terms only;
7. apply a light inflectional stemmer (`-s`/`-es`/`-ies`/`-ed`/`-ing` plus
   final-e elision, run to an idempotent fixpoint) to both query and section
   terms — after aliases, so reviewed mappings always win;
8. calculate smoothed inverse document frequency (IDF) across the 10 sections;
9. score each distinct query term once, with title matches weighted `1.5` and
   body-only matches weighted `1.0`;
10. admit a candidate when it has a title anchor or at least two matched terms;
11. keep candidates scoring at least `60%` of the best score and at least
    `1.0`;
12. for a focused two-term topic, retain a lower-scoring sibling only when it
    shares a title anchor with a full-coverage candidate;
13. sort by descending score and then original document order; and
14. return every section that passes the relevance gate—there is no fixed
    `TOP_K`.

The constants are calibrated against the 27-case **calibration set** in
`tests/fixtures/retrieval_cases.json` — the numbers on that set are a fit
statistic, not a generalization estimate. The gate achieves 100% exact-case
pass, section precision, section recall, and unknown-query rejection on that
tuning set while remaining fully deterministic; generalization is measured
separately on a held-out set (see [Evaluation](#evaluation)):

| Query | Retrieved sections |
|---|---|
| `international travel` | Approval Process, Daily Allowance, Insurance |
| `How many remote days are allowed?` | Remote Work |
| `overseas business trip per diem` | Daily Allowance |
| `annual vacation entitlements` | Annual Leave |
| `international card fee` | PaySiam Gateway only |
| `international travel insurance coverage` | Travel Insurance only |
| `international travel approval, allowance, and insurance requirements` | All three Travel sections |
| `escalate a P1 outage` | Support Escalation + Customer Support Levels |
| `Can I work remotely?` | Remote Work + Hybrid Work |
| `What is the CEO's salary?` | No sections |

The tool returns source sections as raw text. It does not ask an LLM to search,
summarize, enrich, or rank the evidence.

## Technology stack

| Component | Technology | Purpose |
|---|---|---|
| Runtime | Python 3.11 | Application and tests |
| Orchestration | LangGraph | Fixed two-node workflow and shared state |
| Agent/tool primitives | LangChain Core | Messages, tool schema, and invocation |
| LLM integration | LangChain OpenAI | `ChatOpenAI` client and tool binding |
| Configuration | python-dotenv | Local environment configuration |
| Knowledge store | UTF-8 text file | Small, inspectable evidence source |
| Web UI | HTML, CSS, vanilla JS | Dependency-free stage-by-stage demo front end |
| Testing | `unittest` | Offline regression and graph tests |

## Project structure

```text
.
├── main.py                       # CLI: single-query and interactive modes
├── knowledge_base.txt            # 10-section local knowledge base
├── requirements.txt              # pinned Python dependencies
├── .env.example                  # safe configuration template
├── LICENSE
├── README.md
├── screenshots/
│   ├── 01_international_travel.png       # CLI runs
│   ├── 02_remote_work.png
│   ├── 03_not_found.png
│   └── ui_01_empty.png … ui_07_dark.png  # web UI states
├── docs/
│   └── DESIGN_NOTES.md           # engineering rationale and trade-offs
├── evaluation_results.md         # generated by the offline retrieval eval
├── answer_eval_results.md        # generated by the opt-in live answer eval
├── src/
│   ├── config.py                 # models, timeouts, and KB path
│   ├── graph.py                  # PipelineState and LangGraph wiring
│   ├── agents/
│   │   ├── __init__.py           # per-model ChatOpenAI construction
│   │   ├── retriever.py          # Data Retriever node
│   │   └── reporter.py           # Report Generator node
│   ├── evaluation/
│   │   ├── dataset.py            # shared fixture loader
│   │   ├── metrics.py            # pure set-based retrieval metrics
│   │   ├── ablation.py           # V0..V5 scoring-layer ladder
│   │   ├── run_retrieval_eval.py # offline eval runner / CI gate
│   │   └── run_answer_eval.py    # opt-in live answer eval runner
│   └── tools/
│       └── retrieval.py          # parsing, scoring, and custom tool
├── tests/
│   ├── fixtures/
│   │   ├── retrieval_cases.json    # 27-case calibration set (tuning set)
│   │   ├── retrieval_heldout.json  # 14-case held-out set (never tuned on)
│   │   └── answer_cases.json       # answer-quality facts and citations
│   ├── test_retrieval.py         # retrieval, stemming, and cache tests
│   ├── test_evaluation.py        # dataset loader and metric tests
│   ├── test_graph.py             # agent behavior and graph handoff
│   ├── test_live_e2e.py          # opt-in real provider integration
│   └── test_main.py              # CLI streaming and failure recovery
└── web/                          # dependency-free single-page UI
    ├── index.html                # markup for the five pipeline stages
    ├── styles.css                # tokens, badges, light and dark themes
    ├── api.js                    # the only backend seam
    ├── mock-data.js              # offline fixtures for the demo
    ├── app.js                    # state and rendering
    └── README.md                 # run and backend-wiring guide
```

## Quick start

The project has been verified on **Python 3.11** and requires a Standard OpenAI
API key for end-to-end queries.

```bash
git clone https://github.com/Sayomphon/Simple_Agentic_RAG.git
cd Simple_Agentic_RAG

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

cp .env.example .env
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

Add your API key to `.env`:

```dotenv
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-5-mini
TEMPERATURE=0
KB_PATH=knowledge_base.txt
LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=2
RUN_LIVE_LLM_TESTS=0
LIVE_LLM_TEST_MODEL=gpt-5-mini
```

| Variable | Required | Default | Description |
|---|---:|---|---|
| `OPENAI_API_KEY` | Yes | — | Standard OpenAI API credential |
| `MODEL_NAME` | No | `gpt-5-mini` | Chat model used by both agents |
| `RETRIEVER_MODEL_NAME` | No | `MODEL_NAME` | Optional override for the Data Retriever only |
| `REPORTER_MODEL_NAME` | No | `MODEL_NAME` | Optional override for the Report Generator only |
| `TEMPERATURE` | No | `0` | Used for models that support a custom temperature |
| `LLM_TIMEOUT_SECONDS` | No | `30` | Per-request provider timeout so the CLI cannot hang |
| `LLM_MAX_RETRIES` | No | `2` | Bounded retry budget for transient provider errors |
| `KB_PATH` | No | `knowledge_base.txt` | Path to the knowledge base; a relative value is anchored to the project root |
| `RUN_LIVE_LLM_TESTS` | No | `0` | Set to `1` only when explicitly running live provider tests |
| `LIVE_LLM_TEST_MODEL` | No | `gpt-5-mini` | Model used by the opt-in integration tests |

`.env` and virtual environments are ignored by Git. Never commit a real API
key.

## Run the application

### Single query

```bash
python main.py "What is the policy on international travel?"
```

### Interactive mode

```bash
python main.py
```

Enter an empty line, `exit`, `quit`, or press `Ctrl-C` to stop.

The CLI exposes all three observable stages and streams them: the snippet
block prints as soon as retrieval finishes, and answer tokens render as
they arrive (the final text on screen is always byte-equal with the
pipeline's report state):

```text
[1] USER QUERY
[2] RETRIEVED SNIPPETS (Data Retriever -> Report Generator)
[3] FINAL ANSWER
```

Suggested queries:

```text
What is the policy on international travel?
Can I work remotely?
What is the CEO's salary?
```

- The international-travel query retrieves three complementary sections.
- The remote-work query combines Remote Work and Hybrid Work.
- The CEO salary query retrieves nothing and returns the deterministic
  not-found sentence.

### Web interface

The repository also ships a single-page UI that renders the same workflow as five
sequential stages, keeping the raw retrieved evidence visually separate from the
synthesised answer.

```bash
open web/index.html
```

There is no build step, no `npm install`, and no dependencies. The page starts on
**bundled mock data** — fixtures copied verbatim from `knowledge_base.txt` — so
every state is demonstrable offline, including the not-found guardrail.

![Web UI empty state before the first query](screenshots/ui_01_empty.png)

This repository contains no HTTP service, so live mode requires adding one.
`web/api.js` is the single seam: point `CONFIG.endpoint` at a `POST /api/query`
route that returns `PipelineState` as JSON, then switch the header pill to
**Live backend**. [`web/README.md`](web/README.md) contains the FastAPI snippet
and the full contract, including why an empty `snippets` array is a valid result
rather than an error.

## Tests

Run the complete default suite:

```bash
python -m unittest discover -v
```

The default run discovers **103 tests**: **99 pass offline** and the **4 live
tests are skipped**. It covers:

- knowledge-base loading, section splitting, and parse-cache invalidation;
- phrase/token normalization, query-framing removal, and the inflectional
  stemmer (table-driven cases, corpus-wide idempotency, stopword-collision
  guard);
- deterministic IDF and weighted title/body scoring;
- the 27-case calibration set (exact pass, precision/recall, unknown
  rejection) via the shared evaluation metrics;
- dataset-loader schema validation and hand-computed metric tests;
- the ablation ladder's equivalence with production settings;
- Retriever tool-call contract enforcement (1–3 calls, tool name,
  non-empty sub-queries) and the baseline-union superset guarantee;
- query validation (empty / whitespace / over-length) with no LLM call;
- per-role LLM construction, timeout, and retry wiring;
- citation validation, evidence-block wrapping, and injection-guard prompt
  structure;
- malformed knowledge-base rejection;
- string and structured Report Generator output;
- raw-snippet handoff through LangGraph and exact two-node topology;
- deterministic not-found behavior without a Generator LLM call;
- CLI streaming (screen text byte-equal with the state report), exception
  chaining, safe error rendering, exit status, and interactive recovery.

The suite uses mocks at the LLM boundary, so it needs no API key and makes no
network requests. The live tests remain skipped unless explicitly enabled.

Run the offline evaluation (also usable as a CI gate — it exits non-zero if
the current variant misses its calibration thresholds):

```bash
python -m src.evaluation.run_retrieval_eval
```

### Opt-in live LLM integration

The live gate verifies authentication, model/tool-call compatibility, the
actual `ChatOpenAI.bind_tools()` response shape, raw corpus handoff, and the
real Retriever → Tool → Reporter path:

```bash
RUN_LIVE_LLM_TESTS=1 \
LIVE_LLM_TEST_MODEL=gpt-5-mini \
python -m unittest tests.test_live_e2e -v
```

`OPENAI_API_KEY` must be present in the environment or `.env`; otherwise the
opted-in class is skipped with a clear reason. The four tests use only
fictional knowledge-base questions and make approximately seven provider
calls:

| Test | Retriever LLM | Reporter LLM | Total |
|---|---:|---:|---:|
| Known international-travel query | 1 | 1 | 2 |
| Unknown CEO-salary query | 1 | 0 | 1 |
| Multi-intent baseline-coverage query | 1 | 1 | 2 |
| Streamed CLI byte-equality query | 1 | 1 | 2 |

The known-query assertion checks structural contracts and verifies that every
returned snippet is byte-for-byte one of the loaded corpus sections. It does
not assert exact generative wording. The unknown path verifies the exact
deterministic not-found sentence, the multi-intent case asserts the handoff
is a superset of the deterministic baseline search, and the CLI case asserts
the streamed screen text ends byte-equal with the state report.

For a submission or release check, run:

```bash
python -m compileall -q main.py src tests
python -m pip check
python -m unittest discover -v
python -m src.evaluation.run_retrieval_eval
git diff --check
```

Run the live command separately only with an approved project-scoped key,
provider usage limits, and controlled egress. Do not publish prompts, raw
responses, environment dumps, authorization headers, or credentials as CI
artifacts.

## Evaluation

Retrieval and answer quality are measured by code in this repository, not by
hand-checked examples. Both runners are reproducible:

```bash
python -m src.evaluation.run_retrieval_eval          # offline, no API key
RUN_LIVE_LLM_TESTS=1 python -m src.evaluation.run_answer_eval
```

### Datasets

Two labeled retrieval sets plus one answer-quality set, all in
`tests/fixtures/`:

| Set | n | Purpose |
|---|---|---|
| calibration | 27 | The set every scoring constant was tuned against (including the stemmer rules and the added `morphology` cases). Numbers here are a fit statistic, not a generalization estimate. |
| held-out | 14 | Written from `knowledge_base.txt` alone after the retrieval implementation was frozen, committed before the evaluator ever ran against it, and never edited to flatter a result. |
| answer cases | 12 | Required/forbidden facts and allowed citations per query, written from the knowledge base and committed before the first answer-eval run. |

Because the retriever returns a threshold-gated set rather than a fixed-size
ranking (there is no `TOP_K`), retrieval is scored with set-based metrics.
`@k` metrics are deliberately not reported — the system has no `k`. Negative
queries are excluded from precision/recall/MRR and scored only by the
false-positive rate.

### Retrieval results

| set | exact_match | precision_macro | recall_macro | F1 | MRR | FP_rate (neg) |
|---|---|---|---|---|---|---|
| calibration (27) | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 | 0.0% |
| held-out (14) | 57.1% | 77.3% | 72.7% | 0.749 | 0.818 | 66.7% |

The held-out gap is expected and honest: unseen synonyms are out of scope
for a curated lexical system, and two of the three held-out negatives
deliberately contain corpus vocabulary (`parental leave`,
`TigerLink VPN password`) to make the false-positive rate hard to pass.
A local search takes well under a millisecond (p50 ≈ 0.02–0.6 ms depending
on cache warmth); end-to-end latency is dominated by the LLM calls.

### What each design decision buys

The pipeline is scored with each layer removed, on the held-out set:

| variant | exact_match | precision_macro | recall_macro | FP_rate (neg) |
|---|---|---|---|---|
| V0 raw token overlap | 21.4% | 43.1% | 90.9% | 100.0% |
| V1 + query-term filtering | 28.6% | 54.7% | 90.9% | 100.0% |
| V2 + phrase/token aliases | 35.7% | 63.8% | 100.0% | 100.0% |
| V3 + IDF and title weighting | 35.7% | 63.8% | 100.0% | 100.0% |
| V4 + relevance gate and sibling expansion | 50.0% | 60.6% | 63.6% | 66.7% |
| V5 current (+ light inflectional stemming) | 57.1% | 77.3% | 72.7% | 66.7% |

BM25-style term-frequency saturation was implemented and measured as a
seventh rung: at the largest constant that keeps calibration at 100% it
matched V5 on every metric, so it was dropped from the default
configuration. The measurement and reasoning are recorded in
[docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md).

### Answer-level results

All axes are scored by deterministic matching — no LLM judge, no reference
answers. The generator output itself is probabilistic: results below were
produced with `gpt-5-mini` for both agents, prompt version `a40dfbd`
(commit), and 1 run per case over all 53 labeled queries.

| axis | result | threshold |
|---|---|---|
| citation validity (runtime-enforced) | 100.0% (53/53) | 100% |
| not-found discipline | 100.0% (9/9) | 100% |
| evidence provenance | 100.0% (53/53) | 100% |
| no LLM call on empty retrieval | 100.0% (7/7) | 100% |
| baseline coverage | 100.0% (53/53) | 100% |
| required-fact coverage | 100.0% (14/14) | 100% |
| unsupported-number rate | 0.0% (0/22) | 0% |
| forbidden-fact violations | 0 | 0 |

Full per-variant tables, per-case mismatches, and run metadata are in
[evaluation_results.md](evaluation_results.md) and
[answer_eval_results.md](answer_eval_results.md).

**Evaluation limitations:** both retrieval sets are small (n = 27 and
n = 14) over a 10-section corpus, so a single case moves a percentage by
several points. The held-out set is written by the same author as the
knowledge base. Answer metrics come from one run per case of a
probabilistic model. No LLM-as-judge axis is reported — semantic
faithfulness beyond citation validity is not measured.

## Example results

### Command-line interface

The screenshots below were captured from successful live CLI runs with
`gpt-5-mini`. Each image shows the user query, the raw evidence handoff, and the
final grounded answer.

#### International travel — multi-section synthesis

The assignment's sample question retrieves Approval Process, Daily Allowance,
and Insurance sections before producing one cohesive answer.

![International travel query with three retrieved sections and a grounded answer](screenshots/01_international_travel.png)

#### Remote work — related-section synthesis

The system combines Remote Work Policy and Hybrid Work Guidelines without
duplicating overlapping information.

![Remote work query with Remote Work and Hybrid Work evidence](screenshots/02_remote_work.png)

#### Knowledge-base gap — deterministic not-found

Executive salary information does not exist in the knowledge base, so the
system returns the fixed fallback instead of inventing an answer.

![CEO salary query returning the deterministic not-found answer](screenshots/03_not_found.png)

### Web interface

These screenshots come from the bundled UI running on **mock fixtures**, not a
live provider call. The retrieved sections are byte-identical to
`knowledge_base.txt`, and the mock gate returns the same sections as the Python
tool for the queries in the table above.

#### International travel — evidence handoff made visible

Step 2 shows the forced single tool call and confirms the query reached the tool
unchanged. Step 3 shows all three sections raw. Step 5 shows the synthesised
answer with its section citations.

![Web UI with three raw sections and a grounded answer](screenshots/ui_03_international_travel.png)

#### Not-found guardrail

No section clears the relevance gate, so the evidence panel reports
*No evidence found* and the Report Generator returns the deterministic sentence
without an LLM call.

![Web UI showing the not-found state with zero snippets](screenshots/ui_04_not_found.png)

#### Loading state

Each stage reports its own status while the workflow runs, so the sequence
Retriever → Evidence → Generator → Answer stays visible in flight.

![Web UI running with per-stage status badges and skeletons](screenshots/ui_02_running.png)

#### Backend failure

A failure is not converted into a not-found. The error is surfaced with the
unfinished stages marked *Failed*, matching the CLI's failure semantics.

![Web UI error state after an unreachable backend](screenshots/ui_05_error.png)

#### Responsive and dark theme

The same five stages on a 390 px viewport and in the dark colour scheme.

| Mobile | Dark |
|---|---|
| ![Web UI on a mobile viewport](screenshots/ui_06_mobile_remote_work.png) | ![Web UI in dark theme](screenshots/ui_07_dark.png) |

## Design decisions

**Why LangGraph.** The assignment evaluates orchestration, and LangGraph makes
the execution order and handoff contract inspectable. Each agent is a node,
each transition is an explicit edge, and `snippets` is a visible state field
rather than an implicit function-call detail.

**Why a forced retrieval tool call.** Binding the Retriever with
`tool_choice="required"` asks the model to use retrieval, while runtime
validation enforces 1–3 correctly named calls with non-empty queries. Only
the custom tool reads the knowledge base; the Retriever's model output is
not treated as evidence.

**Why the baseline union.** Letting the model decompose a mixed question
improves recall for multi-intent queries, but model behavior is
probabilistic. The node therefore always runs the original query itself
first and unions in the sub-query results — the handoff is provably a
superset of the deterministic single-search baseline, so a bad
decomposition can never lose recall, only add evidence.

**Why normalized weighted lexical retrieval.** For a 10-section assignment
corpus, reviewed phrase/token aliases plus a light inflectional stemmer and
IDF-weighted title/body matching are easier to explain, audit, and test
than an embedding index. The relative relevance gate tolerates
natural-language filler without letting a broad one-term match overwhelm a
focused section. The thresholds, aliases, and stemmer rules are versioned
with the calibration set instead of being tuned from one example, and every
layer's contribution is measured in the ablation table above.

**Why raw state handoff.** Keeping source sections unchanged makes it possible
to compare the Generator's input directly with `knowledge_base.txt`. This
separates retrieval quality from answer-generation quality.

**Why short-circuit empty evidence.** When `snippets` is empty, there is nothing
safe to synthesize. Returning a fixed sentence without an LLM call reduces
hallucination risk, latency, and API cost.

**Why failures are not converted to not-found.** A missing tool call, malformed
corpus, or empty model response means the pipeline failed; it does not prove
that the requested information is absent. These errors retain their original
cause internally, while the CLI emits a concise message without raw queries,
prompts, snippets, or credentials. Single-query mode exits non-zero, and
interactive mode continues with the next query.

**Why pinned dependencies.** Exact versions reduce installation drift in a
reviewer's Python 3.11 environment.

## Limitations and production next steps

This repository intentionally optimizes for clarity and assignment alignment,
not production scale.

- **Curated lexical semantics only:** unseen synonyms and conceptual
  similarity are not understood — measured directly: held-out exact match
  is 57.1% versus 100% on the calibration set, with misses like
  "how fast do you reply" never reaching the support sections.
- **Inflectional stemming only:** the light stemmer covers
  `-s`/`-es`/`-ies`/`-ed`/`-ing` (held-out `unseen_inflection` cases pass),
  but derivational forms still need reviewed aliases, and stemming can
  bypass the surface-form generic-term filter — on the held-out set,
  `card processing rates` wrongly retrieves "…Process" sections because
  `processing` stems to the filtered word `process` after filtering.
- **Hard negatives leak:** two of three held-out negatives that
  intentionally contain corpus vocabulary (`parental leave`,
  `TigerLink VPN password`) retrieve a plausible-but-wrong section
  (66.7% FP rate on that small negative set); the Report Generator's
  insufficient-evidence guardrail then produced the correct not-found
  answer in the live answer eval, but that second layer is probabilistic.
- **English query terms:** effective retrieval requires specific English terms
  present in the knowledge base, its reviewed alias vocabulary, or a
  stemmable inflection of them.
- **Heuristic relevance gate:** the `1.5` title weight and `0.60` relative
  cutoff pass the 27-case calibration set but require recalibration against
  representative production traffic.
- **Small local corpus:** a linear scan is appropriate for 10 sections, not
  hundreds of thousands of documents.
- **Citation-level grounding only:** invented citations fail loudly at
  runtime and the answer eval checks facts and numbers against the handed-off
  evidence, but there is no per-claim semantic faithfulness check.
- **No service layer:** the pipeline runs in-process behind the CLI. There is no
  HTTP API, authentication, persistence, monitoring, or rate-limit handling.
- **Mock-first web UI:** because no service exists yet, the front end ships with
  offline fixtures. They reproduce the retrieval gate's decisions for the
  evaluated queries but do not execute the Python tool, so the UI is a
  demonstration of the workflow rather than a second implementation of it.

A production evolution would add document ingestion and lifecycle management,
a larger labeled retrieval dataset, hybrid lexical/vector retrieval, metadata
filters and access control, answer faithfulness checks, tracing, cost and
latency monitoring, provider error handling, and a deployable API layer.

## License

This project is available under the [MIT License](LICENSE).
