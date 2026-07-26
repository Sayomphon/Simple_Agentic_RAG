# Retrieval fixture roles

- `retrieval_cases.json` is the original lexical calibration and regression
  set.
- `retrieval_heldout.json` is the untouched generalization set. It must never
  be used to tune scoring or thresholds.
- `retrieval_negatives.json` is an intentionally designed hard-negative
  **calibration** set. Every query is a near-miss whose topic or vocabulary
  overlaps the knowledge base while the requested answer is absent. It is
  used to choose the semantic cosine threshold and must not be presented as
  held-out evidence.

Once measured, fixtures are append-only unless a label is demonstrably wrong.
