"""Compare all retrieval modes on the golden test set.

Usage:
    python -m src.evaluation.run_eval

Prints an overall table (mode x metric) and a per-category breakdown, and
writes the same tables to ``evaluation_results.md`` with the run date and
the exact config values, so any number in the README can be reproduced.

Methodology notes:
    - The factory is bypassed on purpose: it is a per-process singleton
      keyed to one SEARCH_MODE, while this harness must compare all modes
      side by side in a single run.
    - Each mode gets its OWN retriever instances. Sharing the dense
      retriever between "semantic" and "hybrid" would let the second mode
      reuse the first mode's memoized query embeddings, understating its
      real latency. The corpus embedding itself still comes from the disk
      cache, so isolation costs only one extra embedding call per query.
    - Latency is wall-clock per ``search`` call and therefore includes the
      OpenAI query-embedding round-trip for dense/hybrid — that network
      cost is real and belongs in the number.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.config import (
    EMBEDDING_MODEL,
    FUSION_METHOD,
    MIN_COSINE,
    MIN_SCORE,
    RRF_K,
    TOP_K,
)
from src.evaluation.testset import TEST_SET
from src.retrievers.base import Retriever, load_chunks
from src.retrievers.dense import OpenAIEmbeddingRetriever
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.keyword import BM25Retriever

RESULTS_PATH = Path("evaluation_results.md")
CATEGORIES = ("lexical", "semantic", "multi_chunk", "negative")


@dataclass(frozen=True)
class CaseResult:
    """Retrieval outcome for one (mode, query) pair."""

    case_id: str
    category: str
    expected: tuple[str, ...]
    retrieved: tuple[str, ...]  # titles, best rank first
    latency_ms: float

    @property
    def hit(self) -> bool:
        """At least one expected section appears in the top-k."""
        return any(title in self.retrieved for title in self.expected)

    @property
    def recall(self) -> float:
        """Fraction of expected sections retrieved."""
        found = sum(title in self.retrieved for title in self.expected)
        return found / len(self.expected)

    @property
    def reciprocal_rank(self) -> float:
        """1/rank of the best-ranked expected section (0 when absent)."""
        for rank, title in enumerate(self.retrieved, start=1):
            if title in self.expected:
                return 1.0 / rank
        return 0.0

    @property
    def false_positive(self) -> bool:
        """A negative query that retrieved anything at all."""
        return not self.expected and bool(self.retrieved)


def build_retrievers() -> dict[str, Retriever]:
    """Construct per-mode retrievers with isolated instances (see module doc)."""
    chunks = load_chunks()
    return {
        "keyword": BM25Retriever(chunks),
        "semantic": OpenAIEmbeddingRetriever(chunks),
        "hybrid": HybridRetriever(
            BM25Retriever(chunks), OpenAIEmbeddingRetriever(chunks)
        ),
    }


def evaluate_mode(retriever: Retriever) -> list[CaseResult]:
    """Run every test case against one retriever."""
    results = []
    for case in TEST_SET:
        query = str(case["query"])
        started = time.perf_counter()
        hits = retriever.search(query, top_k=TOP_K)
        latency_ms = (time.perf_counter() - started) * 1000
        results.append(
            CaseResult(
                case_id=str(case["id"]),
                category=str(case["category"]),
                expected=tuple(case["expected_titles"]),  # type: ignore[arg-type]
                retrieved=tuple(hit.title for hit in hits),
                latency_ms=latency_ms,
            )
        )
    return results


def summarize(results: list[CaseResult]) -> dict[str, float | str]:
    """Aggregate one mode's results into the report metrics.

    Hit/recall/MRR average over answerable queries only; the negative
    category is scored solely by its false-positive rate — mixing the two
    would let a mode look accurate by being merely silent.
    """
    answerable = [r for r in results if r.expected]
    negatives = [r for r in results if not r.expected]
    return {
        "hit_rate": sum(r.hit for r in answerable) / len(answerable),
        "recall": sum(r.recall for r in answerable) / len(answerable),
        "mrr": sum(r.reciprocal_rank for r in answerable) / len(answerable),
        "fp_rate": (
            sum(r.false_positive for r in negatives) / len(negatives)
            if negatives
            else 0.0
        ),
        "latency_ms": sum(r.latency_ms for r in results) / len(results),
    }


def _format_table(header: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table that is also readable in a terminal."""
    widths = [
        max(len(str(cell)) for cell in column)
        for column in zip(header, *rows, strict=True)
    ]
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(
            str(c).ljust(w) for c, w in zip(cells, widths, strict=True)
        ) + " |"
    divider = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(header), divider, *(line(r) for r in rows)])


def overall_table(all_results: dict[str, list[CaseResult]]) -> str:
    header = ["mode", "hit_rate@k", "recall@k", "MRR", "FP_rate(neg)", "avg_latency"]
    rows = []
    for mode, results in all_results.items():
        m = summarize(results)
        rows.append(
            [
                mode,
                f"{m['hit_rate']:.0%}",
                f"{m['recall']:.0%}",
                f"{m['mrr']:.3f}",
                f"{m['fp_rate']:.0%}",
                f"{m['latency_ms']:.1f} ms",
            ]
        )
    return _format_table(header, rows)


def category_table(all_results: dict[str, list[CaseResult]]) -> str:
    header = ["category", "n", "metric", *all_results.keys()]
    rows = []
    for category in CATEGORIES:
        per_mode = {
            mode: [r for r in results if r.category == category]
            for mode, results in all_results.items()
        }
        n = len(next(iter(per_mode.values())))
        if category == "negative":
            metric_rows = [
                ("FP_rate", lambda rs: f"{sum(r.false_positive for r in rs) / len(rs):.0%}"),
            ]
        else:
            metric_rows = [
                ("hit_rate", lambda rs: f"{sum(r.hit for r in rs) / len(rs):.0%}"),
                ("recall", lambda rs: f"{sum(r.recall for r in rs) / len(rs):.0%}"),
                ("MRR", lambda rs: f"{sum(r.reciprocal_rank for r in rs) / len(rs):.3f}"),
            ]
        for metric_name, fn in metric_rows:
            rows.append(
                [category, str(n), metric_name]
                + [fn(per_mode[mode]) for mode in all_results]
            )
    return _format_table(header, rows)


def misses_table(all_results: dict[str, list[CaseResult]]) -> str:
    """Every non-perfect case, so failures are inspectable, not hidden."""
    header = ["mode", "case", "expected", "retrieved (top-k)"]
    rows = []
    for mode, results in all_results.items():
        for r in results:
            perfect = (r.recall == 1.0) if r.expected else not r.false_positive
            if not perfect:
                rows.append(
                    [
                        mode,
                        r.case_id,
                        ", ".join(r.expected) or "(nothing)",
                        ", ".join(r.retrieved) or "[]",
                    ]
                )
    return _format_table(header, rows) if rows else "(all cases perfect)"


def main() -> None:
    all_results = {
        mode: evaluate_mode(retriever)
        for mode, retriever in build_retrievers().items()
    }

    overall = overall_table(all_results)
    by_category = category_table(all_results)
    misses = misses_table(all_results)

    print(f"Golden test set: {len(TEST_SET)} queries, top_k={TOP_K}\n")
    print(overall)
    print("\nPer-category breakdown:\n")
    print(by_category)
    print("\nImperfect cases (for analysis):\n")
    print(misses)

    RESULTS_PATH.write_text(
        "\n".join(
            [
                "# Retrieval Evaluation Results",
                "",
                f"- Run date: {datetime.now():%Y-%m-%d %H:%M}",
                f"- Test set: {len(TEST_SET)} queries "
                "(src/evaluation/testset.py)",
                f"- Config: TOP_K={TOP_K}, MIN_SCORE={MIN_SCORE}, "
                f"MIN_COSINE={MIN_COSINE}, EMBEDDING_MODEL={EMBEDDING_MODEL}, "
                f"FUSION_METHOD={FUSION_METHOD}, RRF_K={RRF_K}",
                "",
                "## Overall",
                "",
                overall,
                "",
                "## Per category",
                "",
                by_category,
                "",
                "## Imperfect cases",
                "",
                misses,
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nWritten to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
