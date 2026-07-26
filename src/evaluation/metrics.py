"""Pure set-based retrieval metrics: no I/O, no printing, no globals.

The retriever returns a threshold-gated, ordered set of sections rather
than a fixed-size ranking, so quality is measured with set-based metrics
plus MRR over the returned order. Negative queries (empty expectation)
are excluded from precision/recall/MRR and scored only by the
false-positive rate, otherwise recall would divide by zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryOutcome:
    """Retrieved titles (in ranked order) versus expected titles for one query."""

    retrieved: tuple[str, ...]
    expected: tuple[str, ...]

    @property
    def is_negative(self) -> bool:
        return not self.expected


@dataclass(frozen=True)
class RetrievalMetrics:
    """Aggregate scores for one dataset. Ratios are 0..1, or None when the
    contributing group is empty (e.g. no negative queries in the dataset)."""

    total_queries: int
    positive_queries: int
    negative_queries: int
    exact_match: float
    set_match: float
    precision_macro: float | None
    recall_macro: float | None
    f1_macro: float | None
    precision_micro: float | None
    recall_micro: float | None
    mrr: float | None
    fp_rate_negative: float | None
    over_retrieval: float | None
    under_retrieval: float | None


def query_precision(retrieved: Sequence[str], expected: Sequence[str]) -> float:
    """Fraction of retrieved titles that are expected; empty retrieval is
    perfectly precise only when nothing was expected."""
    if not retrieved:
        return 1.0 if not expected else 0.0
    relevant = len(set(retrieved) & set(expected))
    return relevant / len(retrieved)


def query_recall(retrieved: Sequence[str], expected: Sequence[str]) -> float:
    """Fraction of expected titles that were retrieved."""
    if not expected:
        raise ValueError(
            "recall is undefined for a negative query; "
            "score it with the false-positive rate instead"
        )
    relevant = len(set(retrieved) & set(expected))
    return relevant / len(expected)


def reciprocal_rank(retrieved: Sequence[str], expected: Sequence[str]) -> float:
    """1/rank of the first relevant title, or 0.0 when none was retrieved."""
    expected_set = set(expected)
    for rank, title in enumerate(retrieved, start=1):
        if title in expected_set:
            return 1.0 / rank
    return 0.0


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _harmonic_mean(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    if first + second == 0:
        return 0.0
    return 2 * first * second / (first + second)


def evaluate(outcomes: Iterable[QueryOutcome]) -> RetrievalMetrics:
    """Aggregate one dataset of per-query outcomes into RetrievalMetrics."""
    outcomes = list(outcomes)
    positives = [outcome for outcome in outcomes if not outcome.is_negative]
    negatives = [outcome for outcome in outcomes if outcome.is_negative]

    exact_matches = sum(
        outcome.retrieved == outcome.expected for outcome in outcomes
    )
    set_matches = sum(
        set(outcome.retrieved) == set(outcome.expected) for outcome in outcomes
    )

    precisions = [
        query_precision(outcome.retrieved, outcome.expected)
        for outcome in positives
    ]
    recalls = [
        query_recall(outcome.retrieved, outcome.expected)
        for outcome in positives
    ]
    precision_macro = _mean(precisions)
    recall_macro = _mean(recalls)

    retrieved_total = sum(len(outcome.retrieved) for outcome in positives)
    expected_total = sum(len(outcome.expected) for outcome in positives)
    retrieved_relevant = sum(
        len(set(outcome.retrieved) & set(outcome.expected))
        for outcome in positives
    )

    over_retrieved = sum(
        bool(set(outcome.retrieved) - set(outcome.expected))
        for outcome in positives
    )
    under_retrieved = sum(
        bool(set(outcome.expected) - set(outcome.retrieved))
        for outcome in positives
    )

    return RetrievalMetrics(
        total_queries=len(outcomes),
        positive_queries=len(positives),
        negative_queries=len(negatives),
        exact_match=exact_matches / len(outcomes) if outcomes else 0.0,
        set_match=set_matches / len(outcomes) if outcomes else 0.0,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=_harmonic_mean(precision_macro, recall_macro),
        precision_micro=(
            retrieved_relevant / retrieved_total if retrieved_total else None
        ),
        recall_micro=(
            retrieved_relevant / expected_total if expected_total else None
        ),
        mrr=_mean(
            [
                reciprocal_rank(outcome.retrieved, outcome.expected)
                for outcome in positives
            ]
        ),
        fp_rate_negative=(
            sum(bool(outcome.retrieved) for outcome in negatives)
            / len(negatives)
            if negatives
            else None
        ),
        over_retrieval=(
            over_retrieved / len(positives) if positives else None
        ),
        under_retrieval=(
            under_retrieved / len(positives) if positives else None
        ),
    )


__all__ = [
    "QueryOutcome",
    "RetrievalMetrics",
    "evaluate",
    "query_precision",
    "query_recall",
    "reciprocal_rank",
]
