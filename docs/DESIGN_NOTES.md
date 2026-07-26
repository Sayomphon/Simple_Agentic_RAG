# Design Notes — Simple Agentic RAG

Engineering rationale behind the deliberately small two-agent pipeline: the
failure modes that shaped its boundaries, why retrieval is deterministic, and
what the current tests actually verify. Results here are reproducible with:

```bash
./.venv-clean/bin/python -m unittest discover -v
```

## Design goal and system boundary

The assignment requires a Data Retriever Agent, a custom RAG tool, and a
Report Generator Agent connected through an explicit handoff. The design keeps
that path visible instead of hiding it behind a general-purpose autonomous
agent:

```text
START -> data_retriever -> report_generator -> END
```

The graph has no router, retry loop, memory, or conditional search path. Each
query follows the same bounded workflow:

1. the Data Retriever requests between one and three calls to
   `search_knowledge_base` (one per focused sub-topic of the question);
2. the node executes the original query first as a deterministic baseline,
   then each sub-query, and unions the raw results;
3. raw matching sections are stored in `snippets`;
4. the Report Generator synthesizes an answer from those snippets only; and
5. the CLI streams the query, evidence handoff, and final answer as
   separate stages.

The handoff is a typed, intentionally minimal contract:

```python
class PipelineState(TypedDict):
    query: str
    snippets: list[str]
    report: str
```

This separation makes retrieval observable and independently testable. It also
prevents the Retriever from answering the user or silently summarizing source
material before the Generator receives it.

## Boundary failures that shaped the implementation

The first implementation could make distinct failures look like a legitimate
knowledge-base miss:

| Boundary condition | Earlier risk | Current behavior |
|---|---|---|
| Retriever returns no tool call | Empty evidence | `RetrievalProtocolError` |
| Retriever changes the query | Search executes altered intent | Reject before tool execution |
| Knowledge base is empty or malformed | No matching sections | `KnowledgeBaseFormatError` |
| Generator returns non-text or empty output | Invalid text reaches the CLI | `ReportGenerationError` |
| One graph invocation fails | Interactive process ends | Safe error; next query is accepted |

The key rule is that `snippets == []` has one meaning only: the corpus loaded
successfully, the tool executed correctly, and no section passed the relevance
rules. Protocol, corpus, provider, and model-output failures remain exceptions.
This prevents an operational incident from being presented to the user as
“the knowledge base has no answer.”

## Default deterministic lexical retrieval

The current corpus contains 10 small, machine-delimited sections, so a local
lexical scan is easier to inspect and reproduce than an embedding service or
vector database. `src/retrievers/lexical.py` performs the following steps:

1. read `knowledge_base.txt` as UTF-8, with parsing cached per file
   identity (path, `mtime_ns`, size);
2. require at least one `--- Section Title ---` heading and a non-empty body
   for every section;
3. tokenize English/alphanumeric text with `[a-z0-9]+` and apply reviewed
   phrase aliases (`per diem` → `daily allowance`) and token aliases for
   derivational variants and synonyms (`lodging` → `hotel`);
4. remove English stop words, query-framing words, and generic enterprise
   terms such as `policy` and `company` from query terms only;
5. stem both query and section terms with a light inflectional stemmer
   (`-s`/`-es`/`-ies`/`-ed`/`-ing`, final-e elision, run to an idempotent
   fixpoint) — after aliases, so reviewed mappings win;
6. weight each distinct matched term by smoothed IDF, with title matches
   at 1.5x body weight;
7. admit a candidate only with a title anchor or at least two matched
   terms, and keep candidates scoring at least 60% of the best score
   (minimum 1.0);
8. for a focused two-term query, retain title-linked sibling sections only
   when a full-coverage section provides a reliable anchor; and
9. rank by score, then by original source order.

The title exceptions solve two different recall problems without turning every
shared word into a match. A broad request for international-travel rules can
retrieve approval, allowance, and insurance sections, while a request for an
international card fee does not admit every section containing
`international`. Each layer's measured contribution is in
`evaluation_results.md` (V0–V5 ablation over a calibration and a held-out
set).

Version-controlled regression cases include:

| Query | Expected sections |
|---|---|
| `international travel` | Approval Process, Daily Allowance, Insurance |
| `international card fee` | PaySiam Gateway only |
| `international travel insurance coverage` | Travel Insurance only |
| `Can I work remotely?` | Remote Work and Hybrid Work |
| `How do I escalate a P1 outage?` | Support Service Levels and Escalation Process |
| `What is the CEO's salary?` | No sections |

The tool returns the original section text, including its title. It does not
use an LLM to rank, rewrite, summarize, or enrich evidence, and it has no fixed
`TOP_K`. At this corpus size, returning every section that passes the rule
preserves cross-section answers better than an arbitrary cap.

## Semantic and hybrid measured path

The agent-facing `search_knowledge_base(query: str) -> list[str]` contract is
unchanged. Internally, `Retriever.search()` returns scored raw chunks and a
configuration-owned factory selects `lexical`, `semantic`, or `hybrid`.
The LLM never selects a mode, and the graph, agent nodes, and raw-snippet
handoff are identical in all three cases.

Semantic mode is deliberately basic: `text-embedding-3-small`, an in-memory
linear cosine scan over 10 sections, and one global threshold. Document
vectors are batched and cached by a hash of embedding model plus raw chunk
content. The cache stores validated JSON rather than executable
serialization, writes atomically with restrictive file permissions, rejects
symlinked cache paths, and rebuilds malformed content. Missing credentials
or provider failures raise; they are never presented as an empty search and
never trigger a silent lexical fallback.

The hard-negative calibration set contains 12 intentional near-misses. It is
a tuning set, not held-out evidence. The measured distributions overlap:

| calibration measure | value |
|---|---:|
| minimum expected-pair cosine | 0.223143 |
| maximum hard-negative top cosine | 0.757588 |
| clean gap | -0.534445 |
| selected `MIN_COSINE` | 0.392817 |
| pair precision / recall / F0.5 | 0.875 / 0.921 / 0.884 |
| hard-negative leaks | 5/12 |

A threshold above the strongest negative would lose all 38 expected pairs.
The deployed value therefore maximizes pair-level F0.5 across six-decimal
configuration boundaries, weighting precision twice as strongly as recall.
Every score, lost positive, leaked negative, and the zero-FP counterfactual
is recorded in `threshold_calibration.md`.

Hybrid mode gates lexical and semantic results independently, then applies
Reciprocal Rank Fusion with `k=60`. Rank-only fusion avoids treating lexical
weights and cosine as comparable magnitudes. Exact-text deduplication
preserves the raw source contract. On held-out queries, hybrid recall rises
to 100%, but its false-positive rate stays at 66.7% because RRF can reorder
admitted evidence; it cannot remove lexical false positives. This is a
recall-oriented mode, not a universal quality improvement.

## Term frequency: measured, then dropped

BM25-style TF saturation (`tf/(tf+K_TF)` on body-only matches) was
implemented behind `RetrievalSettings(use_tf_saturation=True)` and evaluated
as a seventh ablation rung on the 27-case calibration set (2026-07-26):

| variant | exact | P_macro | R_macro | F1 | MRR | FP_neg |
|---|---|---|---|---|---|---|
| V5 (+stemming) | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 | 0.0% |
| V6 (+TF saturation, K_TF=0.05) | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 | 0.0% |

Two findings led to dropping it from the default configuration:

1. A `K_TF x MIN_RELATIVE_SCORE` sweep against the calibration set showed
   the constraint is binding at `K_TF <= 0.08` with the shipped `0.60`
   relative cutoff: these sections are so short (term frequencies are
   almost always 1) that any stronger TF signal shrinks body scores until
   broad title-only matches outrank multi-term body evidence
   (`international card` starts admitting the travel sections).
2. At the largest safe constant, V6 equals V5 on every metric — the layer
   buys nothing measurable.

Per the decision rule set before implementation (keep TF only if the
ablation shows it beats stemming alone on at least one metric), the
default is `use_tf_saturation=False`, the ablation ladder ends at
`V5_current`, and the scoring path remains available to evaluations via
settings.

## Bounded agent behavior

The Retriever uses an LLM because the assignment explicitly asks for an agent
configured to use a custom retrieval tool. The LLM does not control retrieval
semantics:

- only `search_knowledge_base` is bound;
- `tool_choice="required"` forces a tool request;
- between one and three tool calls with the expected tool name and
  non-empty string queries are required — anything else raises
  `RetrievalProtocolError`;
- the node executes every search itself: the graph-state query always runs
  first, so the handoff is a superset of the deterministic single-search
  baseline regardless of how the model decomposes; and
- invalid input (empty, whitespace-only, or over-length queries) is
  rejected before any LLM call.

These checks are enforced in code as well as in the prompt. A malformed
tool-call plan fails visibly instead of continuing with untrusted
arguments, and a poor decomposition can only add evidence, never lose it.

This design has a deliberate trade-off: the Retriever LLM call demonstrates
agent/tool orchestration but adds provider latency and token cost without
improving the deterministic search. In a production service where that
assignment constraint does not exist, calling the retriever directly would
remove one model call and one failure boundary.

## Grounded generation and not-found behavior

The Report Generator has no tools and receives only the original query plus
the retrieved raw sections. Its system prompt requires it to:

- use only supplied evidence;
- combine complementary facts and remove repetition;
- avoid assumptions and invented details;
- use only real section titles when citing a source; and
- distinguish topical material from evidence that actually answers the
  requested fact.

When retrieval returns no snippets, the node does not call the Generator LLM.
It returns the following byte-exact sentence in code:

```text
I could not find this information in the knowledge base.
```

When snippets exist but are insufficient, the same behavior is requested at
the prompt layer, and a mixed question whose evidence covers only part of it
is answered partially with the uncovered part named plainly. Two further
layers harden grounding in code: snippets are passed inside an
`<evidence>` block the prompt declares to be data (knowledge-base text is
untrusted), and every `[Section Title]` citation in a real answer is
validated against the handed-off snippets — an invented citation raises
`ReportGenerationError`. Beyond citations, per-claim verification remains
probabilistic: the live answer eval measures required facts, unsupported
numbers, and forbidden facts per run, but no per-sentence semantic check
exists.

LangChain responses may contain a plain string or structured content blocks.
The Generator reads the framework's normalized text accessor, trims the result,
and raises `ReportGenerationError` when no textual answer exists. Invalid model
output is never converted into a knowledge-base miss.

## Failure semantics and application behavior

Failure handling is intentionally layered:

```text
valid search with no match
  -> snippets=[]
  -> deterministic not-found; no Generator LLM call

invalid Retriever protocol
  -> RetrievalProtocolError

missing or malformed corpus
  -> FileNotFoundError or KnowledgeBaseFormatError

empty/non-text Generator response
  -> ReportGenerationError

any graph execution error at the CLI boundary
  -> QueryExecutionError, preserving the original exception as its cause
```

Single-query mode returns a non-zero status after a pipeline failure.
Interactive mode writes a concise error to `stderr` and continues accepting
queries. The user query, snippets, provider message, and credential-shaped
values are not copied into the application-level error text.

The successful CLI path intentionally prints the raw query and retrieved
sections for auditability. That is appropriate for a local demonstration, but
it is also a data-exposure boundary: production logging must not copy this
output by default.

## Verified outcomes

Using Python 3.11.15 in the repository's clean virtual environment:

- all 122 offline unit tests pass (4 live tests skipped by default);
- the knowledge base loads as exactly 10 ordered sections;
- retrieval results are deterministic for repeated queries, and the parse
  cache invalidates on file change in default lexical mode;
- semantic document caches reuse valid vectors, invalidate on knowledge-base
  changes, rebuild corruption safely, and map results to raw source chunks;
- hybrid gate-first RRF preserves one-sided results and removes duplicates;
- assignment invariants enforce two nodes, fixed sequential edges, raw
  `list[str]` tool output, a tool-free Reporter, and offline lexical default;
- generic-only, stopword-only, unknown, empty, and malformed-corpus cases are
  distinguished;
- stemming is idempotent over every corpus and fixture token and never
  collapses a content term into a stopword;
- the Retriever rejects missing, excess, unexpected, and empty tool calls,
  and its handoff is a superset of the deterministic baseline search;
- invalid queries are rejected before any LLM call;
- invented citations raise instead of shipping a fabricated source;
- raw snippet lists cross the LangGraph handoff unchanged;
- the graph contains exactly the two intended agent nodes;
- the deterministic not-found path makes no Generator LLM call;
- string and structured model text are normalized;
- the streamed CLI answer ends byte-equal with the state report;
- interactive execution recovers after a query failure;
- bytecode compilation succeeds; and
- installed dependencies pass `pip check`.

The suite mocks the LLM boundary and requires neither an API key nor network
access. Retrieval quality itself is measured by
`src/evaluation/run_retrieval_eval.py` over a calibration and a held-out set
(`evaluation_results.md`), and answer quality by the opt-in
`run_answer_eval.py` (`answer_eval_results.md`) — those reports, not this
test list, carry the quality numbers. There is still no claim for latency
SLO or cost per query over real user traffic.

## Security and operational boundary

Credentials are read from environment variables and `.env` is ignored by Git.
Application errors avoid echoing raw queries or provider exceptions. The
embedding cache contains numeric document vectors only, uses JSON rather than
pickle, validates shape and finite values, and is ignored by Git. Semantic and
hybrid modes send knowledge-base sections and user queries to OpenAI; enterprise
use therefore requires explicit data-egress approval and data classification.
The current implementation still lacks production controls such as authentication,
document-level authorization, encryption policy, prompt-injection filtering,
PII classification, rate limiting, audit retention, and redacted telemetry.

The knowledge-base sections are inserted into an LLM prompt as untrusted text.
Before connecting enterprise documents, the system should add content-source
governance, access-control filtering before retrieval, prompt-injection
evaluation, and an output policy appropriate to the business domain.

## Scaling and production path

For 10 sections, reparsing and scanning the text file per request is a
reasonable transparency-first choice. Lexical search cost is approximately
`O(B + n log n)`, where `B` is corpus size and `n` is the number of matching
candidates. Semantic mode performs `O(Sd)` cosine work for `S` sections and
embedding dimension `d`; document vectors are cached, while every query still
incurs one provider request. No ANN or external vector store is justified at
this scale.

A redesign becomes justified when measured requirements show one or more of
the following:

- full-scan latency exceeds the retrieval budget;
- corpus size risks overflowing the Generator context because there is no
  `TOP_K`;
- real Thai queries, spelling variation, or unseen vocabulary cause
  unacceptable measured recall;
- documents require metadata filtering or per-user authorization;
- concurrent traffic reaches model-provider rate limits; or
- production evaluation shows that lexical ranking no longer meets quality
  targets.

The next measured step is a Thai/cross-lingual slice against the current
semantic path. A larger corpus would justify an offline ingestion pipeline
with stable chunk IDs, metadata and ACLs, bounded candidate retrieval, and
possibly an ANN store behind the same snippet handoff. Adopt added
infrastructure only when it improves measured answer quality within agreed
latency, cost, privacy, and operational constraints.
