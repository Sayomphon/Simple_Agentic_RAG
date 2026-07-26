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

1. the Data Retriever requests exactly one call to
   `search_knowledge_base`;
2. the tool reads, validates, and searches the local knowledge base;
3. raw matching sections are stored in `snippets`;
4. the Report Generator synthesizes an answer from those snippets only; and
5. the CLI displays the query, evidence handoff, and final answer separately.

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

## Deterministic retrieval

The current corpus contains 10 small, machine-delimited sections, so a local
lexical scan is easier to inspect and reproduce than an embedding service or
vector database. `src/tools/retrieval.py` performs the following steps:

1. read `knowledge_base.txt` as UTF-8;
2. require at least one `--- Section Title ---` heading and a non-empty body
   for every section;
3. tokenize English/alphanumeric text with `[a-z0-9]+`;
4. remove English stop words and generic enterprise terms such as `policy`,
   `company`, and `information`;
5. count distinct query terms found in each section;
6. require strict-majority term coverage for the normal case;
7. retain strongest multi-term title matches for verbose, multi-intent
   questions;
8. for a focused two-term query, retain title-linked sibling sections only
   when a full-coverage section provides a reliable anchor; and
9. rank by matched-term count, then by original source order.

The title exceptions solve two different recall problems without turning every
shared word into a match. A broad request for international-travel rules can
retrieve approval, allowance, and insurance sections, while a request for an
international card fee does not admit every section containing
`international`.

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

Per the decision rule in IMPROVEMENT_PLAN §4.2, the default is
`use_tf_saturation=False`, the ablation ladder ends at `V5_current`, and
the scoring path remains available to evaluations via settings.

## Bounded agent behavior

The Retriever uses an LLM because the assignment explicitly asks for an agent
configured to use a custom retrieval tool. The LLM does not control retrieval
semantics:

- only `search_knowledge_base` is bound;
- `tool_choice="required"` forces a tool request;
- exactly one tool call and the expected tool name are required;
- the proposed tool query must exactly equal `state["query"]`; and
- the graph-state query, not arbitrary model output, is used to invoke the
  tool.

These checks are enforced in code as well as in the prompt. A malformed or
altered tool call fails visibly instead of continuing with untrusted
arguments.

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
the prompt layer. This second case is probabilistic: the current implementation
does not independently verify every generated claim against the snippets.
The design therefore reduces hallucination risk but does not prove factual
grounding.

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

- all 42 offline unit tests pass;
- the knowledge base loads as exactly 10 ordered sections;
- retrieval results are deterministic for repeated queries;
- generic-only, stopword-only, unknown, empty, and malformed-corpus cases are
  distinguished;
- the Retriever rejects missing, multiple, unexpected, and query-altering
  tool calls;
- raw snippet lists cross the LangGraph handoff unchanged;
- the graph contains exactly the two intended agent nodes;
- the deterministic not-found path makes no Generator LLM call;
- string and structured model text are normalized;
- interactive execution recovers after a query failure;
- bytecode compilation succeeds; and
- installed dependencies pass `pip check`.

The suite mocks the LLM boundary and requires neither an API key nor network
access. These tests verify known contracts and regressions; they are not a
production retrieval benchmark or an answer-quality evaluation. There is no
current claim for semantic recall, groundedness rate, latency SLO, or cost per
query over real user traffic.

## Security and operational boundary

Credentials are read from environment variables and `.env` is ignored by Git.
Application errors avoid echoing raw queries or provider exceptions. The
current implementation still lacks production controls such as authentication,
document-level authorization, encryption policy, prompt-injection filtering,
PII classification, rate limiting, audit retention, and redacted telemetry.

The knowledge-base sections are inserted into an LLM prompt as untrusted text.
Before connecting enterprise documents, the system should add content-source
governance, access-control filtering before retrieval, prompt-injection
evaluation, and an output policy appropriate to the business domain.

## Scaling and production path

For 10 sections, reparsing and scanning the text file per request is a
reasonable transparency-first choice. Approximate search cost is
`O(B + n log n)`, where `B` is corpus size and `n` is the number of matching
candidates. There is no index lifecycle, cache invalidation, or external
storage dependency to operate.

A redesign becomes justified when measured requirements show one or more of
the following:

- full-scan latency exceeds the retrieval budget;
- corpus size risks overflowing the Generator context because there is no
  `TOP_K`;
- real Thai queries, synonyms, spelling variation, or unseen vocabulary cause
  unacceptable recall;
- documents require metadata filtering or per-user authorization;
- concurrent traffic reaches model-provider rate limits; or
- production evaluation shows that lexical ranking no longer meets quality
  targets.

The next step would be an offline ingestion pipeline with stable document and
chunk IDs, metadata and ACLs, Thai-capable embeddings, and a hybrid
lexical/vector retriever behind the existing snippet handoff. Add bounded
candidate retrieval, reranking, citations, redacted tracing, and a
version-controlled evaluation set built from anonymized real questions.
Adopt the added infrastructure only when it improves measured answer quality
within agreed latency, cost, privacy, and operational constraints.
