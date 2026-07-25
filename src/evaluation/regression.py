"""Retrieval regression suite: exact-match golden cases, no LLM, no API key.

Usage:
    python -m src.evaluation.regression      # exits non-zero on regression

This is one of the project's two evaluation harnesses, with distinct roles:
this suite pins exact expected section sets for the default (keyword)
retriever and runs inside ``unittest`` to catch regressions; its sibling
``run_eval`` is a comparative benchmark that scores all three retrieval
modes side by side on a separate golden set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.config import TOP_K
from src.retrievers import get_retriever

CASES_PATH = Path(__file__).with_name("regression_cases.json")


@dataclass(frozen=True)
class CaseResult:
    """Metrics for one golden retrieval query."""

    case_id: str
    expected: frozenset[str]
    actual: frozenset[str]

    @property
    def precision(self) -> float:
        if not self.actual:
            return 1.0 if not self.expected else 0.0
        return len(self.expected & self.actual) / len(self.actual)

    @property
    def recall(self) -> float:
        if not self.expected:
            return 1.0 if not self.actual else 0.0
        return len(self.expected & self.actual) / len(self.expected)

    @property
    def exact(self) -> bool:
        return self.actual == self.expected


def load_cases(path: Path = CASES_PATH) -> list[dict[str, object]]:
    """Load the version-controlled golden query set."""
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate() -> list[CaseResult]:
    """Evaluate every query against its exact expected section-title set."""
    retriever = get_retriever()
    results: list[CaseResult] = []
    for case in load_cases():
        query = str(case["query"])
        expected = frozenset(str(title) for title in case["expected_titles"])
        actual = frozenset(
            chunk.title for chunk in retriever.search(query, top_k=TOP_K)
        )
        results.append(CaseResult(str(case["id"]), expected, actual))
    return results


def main() -> int:
    """Print compact metrics and return non-zero when a regression is found."""
    results = evaluate()
    for result in results:
        status = "PASS" if result.exact else "FAIL"
        print(
            f"{status:4}  {result.case_id:28} "
            f"precision={result.precision:.2f} recall={result.recall:.2f}"
        )

    exact_rate = sum(result.exact for result in results) / len(results)
    macro_precision = sum(result.precision for result in results) / len(results)
    macro_recall = sum(result.recall for result in results) / len(results)
    print(
        "\nSUMMARY  "
        f"cases={len(results)} exact_match={exact_rate:.1%} "
        f"macro_precision={macro_precision:.1%} "
        f"macro_recall={macro_recall:.1%}"
    )
    return 0 if all(result.exact for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
