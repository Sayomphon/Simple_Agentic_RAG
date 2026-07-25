# Design Notes — Retrieval Quality

Engineering rationale behind the retrieval layer: the measured baseline
failures, what was changed, and the verified outcomes. Numbers here are
reproducible with `python -m src.evaluation.run_eval` and the unit suite.

## Baseline problem (measured, then fixed)

The original BM25 implementation tokenized every English word and used one
absolute score threshold. Natural-language stop words could therefore outweigh
the actual intent:

| Query | Baseline result | Expected result |
|---|---|---|
| `work from home` | Equipment and Laptop Policy | Remote Work Policy |
| `What is the CEO's salary?` | Four unrelated sections | No result |
| `What are the cybersecurity incident reporting rules?` | Three unrelated sections | IT Security and Password Policy |

LLM query reformulation sometimes masked these errors, but it did not make the
retrieval mechanism itself reliable — the fix had to be in the retriever.

## What was changed

1. **Deterministic query normalization.** English stop words and generic
   search-intent words are removed, lightweight suffix normalization is
   applied, and a small alias map rewrites high-value phrases such as
   `work from home` → `remote work`.
2. **Ranking and filtering.** Section titles and bodies are scored
   separately with a title boost; multi-term queries must match multiple
   distinct terms; candidates below a percentage of the best score are
   rejected. One measured failure shaped the term gate: `"CEO's salary"`
   tokenized to a stray possessive fragment `s` (indexed from 26/54
   sections) that counted as a second matched term — the fragment is now
   stopworded.
3. **Bounded agent behaviour.** The agent preserves the original query
   when it already contains English search terms, uses the model only to
   translate non-English input, has no control over `top_k`, and executes
   exactly one tool call with the evidence list capped at `TOP_K`.
4. **Reproducible evaluation.** A version-controlled golden query set with
   expected section titles reports exact-match, macro precision, and macro
   recall, and runs inside the standard-library `unittest` suite with no
   API key.

## Verified outcomes

- 15/15 golden retrieval cases return the exact expected section set
  (100% macro precision and recall).
- Negative queries return no sections; empty and punctuation-only queries
  return no sections.
- Tool and agent outputs never exceed `TOP_K`.
- The full unit suite runs without an API key.

These metrics describe the version-controlled golden set only. They prevent
known regressions but do not replace a larger production dataset based on
real, anonymized user queries.

## A later lesson: query rewriting can shift intent

When the agentic retry loop was added (rewrite the query after an empty
search, up to `MAX_SEARCH_ATTEMPTS`), end-to-end sampling caught a new
failure class the retrieval-only eval cannot see: rewriting the negative
request *"employee home addresses and phone numbers"* into a query about
data-handling **policies** surfaced the *Data Classification and Handling*
section, and the generator answered with classification rules instead of
the not-found sentence. The fix was applied at both layers — the rewriter
must rewrite the wording, never the request itself (a query for data must
stay a query for that data), and the generator treats snippets that merely
relate to the topic as non-answers. The regression is now covered by the
end-to-end negative checks in the answer-level evaluation.

## Scaling note

The in-memory design (54 chunks, linear scan) is deliberate at this corpus
size. If real-query evaluation showed unseen vocabulary or Thai-language
recall gaps at larger scale, the path is: batch-embed the corpus offline,
store vectors in an external vector store behind the existing `Retriever`
protocol, and adopt an ANN index — selecting the new mode only when it
improves measured recall without materially reducing precision, latency,
data privacy, or reproducibility.
