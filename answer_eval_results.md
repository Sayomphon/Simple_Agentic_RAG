# Answer-Level Evaluation Results

- Run date: 2026-07-25 15:58
- Search mode: **hybrid**  ·  model: `gpt-5-mini`  ·  `TOP_K=4`
- Questions: 17 (from `src/evaluation/run_qa.py`); 14 scored on all axes, 2 negative, 0 degraded to not-found
- Judges see only (query, snippets, answer) — no golden answers. Deterministic axes use no LLM at all.

## Summary

| axis | method | result | threshold | pass |
|---|---|---|---|---|
| citation validity | deterministic | 100% of answers cite only handed-off sections | 100% | ✅ |
| negative discipline | deterministic | 100% byte-exact not-found (2 queries) | 100% | ✅ |
| faithfulness | LLM judge | 0.985 avg (claims supported/total) | ≥ 0.9 | ✅ |
| answer relevance | LLM judge | 5.00 avg (1-5) | ≥ 4.0 | ✅ |

## Per question

| # | category | citations | faithfulness | relevance |
|---|---|---|---|---|
| 1 | lexical | ✅ | 4/4 (1.00) | 5/5 |
| 2 | lexical | ✅ | 7/7 (1.00) | 5/5 |
| 3 | lexical | ✅ | 1/1 (1.00) | 5/5 |
| 4 | lexical | ✅ | 3/3 (1.00) | 5/5 |
| 5 | semantic | ✅ | 9/9 (1.00) | 5/5 |
| 6 | semantic | ✅ | 6/6 (1.00) | 5/5 |
| 7 | semantic | ✅ | 7/7 (1.00) | 5/5 |
| 8 | semantic | ✅ | 15/15 (1.00) | 5/5 |
| 9 | semantic | ✅ | 7/8 (0.88) | 5/5 |
| 10 | semantic | ✅ | 17/17 (1.00) | 5/5 |
| 11 | multi_chunk | ✅ | 10/11 (0.91) | 5/5 |
| 12 | multi_chunk | ✅ | 5/5 (1.00) | 5/5 |
| 13 | multi_chunk | ✅ | 2/2 (1.00) | 5/5 |
| 14 | negative | — | — | byte-exact ✅ |
| 15 | negative | — | — | byte-exact ✅ |
| 16 | thai | ✅ | 9/9 (1.00) | 5/5 |
| 17 | greeting | — | — | direct route, not scored |

## Unsupported claims (raw, judge verdicts)

**counseling for stress and burnout**
- You can access confidential counseling for stress and burnout through the company’s Employee Assistance Program (MindBridge).

**everything I need for an overseas business trip**
- Below is a concise checklist of what you need and what will be handled for an overseas business trip:
