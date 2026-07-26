"""Derive the semantic cosine gate from positives and hard negatives.

Usage:
    python -m src.evaluation.calibrate_threshold

The original calibration positives establish the weakest supported pair.
The intentionally designed hard-negative set establishes the strongest
unsupported nearest neighbor. The held-out dataset is never read here.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from src.config import (
    EMBED_CACHE_DIR,
    EMBEDDING_MODEL_NAME,
    KB_PATH,
    MIN_COSINE,
)
from src.evaluation.dataset import EvalCase, load_cases
from src.retrievers.base import ScoredChunk
from src.retrievers.semantic import SemanticRetrievalError, SemanticRetriever

RESULTS_PATH = Path(__file__).resolve().parents[2] / "threshold_calibration.md"
_CONFIG_PRECISION = 6


class SemanticScorer(Protocol):
    """Evaluation-only surface that exposes scores before thresholding."""

    @property
    def embedding_api_calls(self) -> int:
        """Return provider method calls made by this scorer."""

    def score_all(self, query: str) -> list[ScoredChunk]:
        """Return every source section in descending cosine order."""


@dataclass(frozen=True)
class CalibrationPair:
    """One measured query-to-section cosine."""

    case_id: str
    query: str
    title: str
    score: float


@dataclass(frozen=True)
class ThresholdDecision:
    """Measured separation and the resulting configuration choice."""

    min_positive: float
    max_negative: float
    gap: float
    recommended: float
    strategy: str
    pair_precision: float
    pair_recall: float
    pair_f_beta: float
    beta: float
    zero_fp_threshold: float
    lost_positive_pairs: tuple[CalibrationPair, ...]
    leaked_negative_pairs: tuple[CalibrationPair, ...]


def _title(chunk: str) -> str:
    return chunk.partition("\n")[0]


def _ceil_for_config(value: float) -> float:
    scale = 10**_CONFIG_PRECISION
    return min(1.0, math.ceil(value * scale) / scale)


def _floor_for_config(value: float) -> float:
    scale = 10**_CONFIG_PRECISION
    return max(-1.0, math.floor(value * scale) / scale)


def _pair_scores(
    positives: list[CalibrationPair],
    negatives: list[CalibrationPair],
    threshold: float,
    *,
    beta: float,
) -> tuple[float, float, float]:
    true_positives = sum(pair.score >= threshold for pair in positives)
    false_positives = sum(pair.score >= threshold for pair in negatives)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = true_positives / len(positives)
    beta_squared = beta * beta
    denominator = (beta_squared * precision) + recall
    f_beta = (
        (1.0 + beta_squared) * precision * recall / denominator
        if denominator
        else 0.0
    )
    return precision, recall, f_beta


def _precision_weighted_threshold(
    positives: list[CalibrationPair],
    negatives: list[CalibrationPair],
    *,
    beta: float,
) -> tuple[float, float, float, float]:
    candidates = {-1.0, 1.0}
    candidates.update(_floor_for_config(pair.score) for pair in positives)
    candidates.update(
        _ceil_for_config(math.nextafter(pair.score, math.inf))
        for pair in negatives
    )
    measured = [
        (
            *_pair_scores(
                positives,
                negatives,
                threshold,
                beta=beta,
            ),
            threshold,
        )
        for threshold in candidates
    ]
    precision, recall, f_beta, threshold = max(
        measured,
        key=lambda row: (
            row[2],
            row[0],
            row[1],
            row[3],
        ),
    )
    return threshold, precision, recall, f_beta


def choose_threshold(
    positives: list[CalibrationPair],
    negatives: list[CalibrationPair],
) -> ThresholdDecision:
    """Choose midpoint for a clean gap, otherwise protect precision."""
    if not positives:
        raise ValueError("At least one positive pair is required")
    if not negatives:
        raise ValueError("At least one negative pair is required")

    min_positive = min(pair.score for pair in positives)
    max_negative = max(pair.score for pair in negatives)
    gap = min_positive - max_negative
    beta = 0.5
    zero_fp_threshold = _ceil_for_config(
        math.nextafter(max_negative, math.inf)
    )
    if gap > 0.0:
        recommended = round(
            max_negative + (gap / 2.0),
            _CONFIG_PRECISION,
        )
        strategy = "clean-gap midpoint"
        precision, recall, f_beta = _pair_scores(
            positives,
            negatives,
            recommended,
            beta=beta,
        )
    else:
        recommended, precision, recall, f_beta = (
            _precision_weighted_threshold(
                positives,
                negatives,
                beta=beta,
            )
        )
        strategy = "precision-weighted F0.5 sweep (overlap)"

    lost_positives = tuple(
        pair for pair in positives if pair.score < recommended
    )
    leaked_negatives = tuple(
        pair for pair in negatives if pair.score >= recommended
    )
    return ThresholdDecision(
        min_positive=min_positive,
        max_negative=max_negative,
        gap=gap,
        recommended=recommended,
        strategy=strategy,
        pair_precision=precision,
        pair_recall=recall,
        pair_f_beta=f_beta,
        beta=beta,
        zero_fp_threshold=zero_fp_threshold,
        lost_positive_pairs=lost_positives,
        leaked_negative_pairs=leaked_negatives,
    )


def measure_pairs(
    scorer: SemanticScorer,
    positives: list[EvalCase],
    negatives: list[EvalCase],
) -> tuple[list[CalibrationPair], list[CalibrationPair]]:
    """Measure expected positive pairs and each negative query's top hit."""
    positive_pairs: list[CalibrationPair] = []
    for case in positives:
        if case.is_negative:
            continue
        by_title = {
            _title(result.chunk): result
            for result in scorer.score_all(case.query)
        }
        for expected_title in case.expected_titles:
            try:
                result = by_title[expected_title]
            except KeyError:
                raise ValueError(
                    f"Expected title {expected_title!r} is absent from the "
                    "knowledge base"
                ) from None
            positive_pairs.append(
                CalibrationPair(
                    case_id=case.id,
                    query=case.query,
                    title=expected_title,
                    score=result.score,
                )
            )

    negative_pairs: list[CalibrationPair] = []
    for case in negatives:
        if not case.is_negative:
            raise ValueError(
                f"Hard-negative case {case.id!r} must have no expected titles"
            )
        results = scorer.score_all(case.query)
        if not results:
            raise ValueError(
                "score_all must return the complete corpus for calibration"
            )
        top_result = results[0]
        negative_pairs.append(
            CalibrationPair(
                case_id=case.id,
                query=case.query,
                title=_title(top_result.chunk),
                score=top_result.score,
            )
        )
    return positive_pairs, negative_pairs


def _pair_table(title: str, pairs: list[CalibrationPair]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| case | score | section | query |",
        "|---|---:|---|---|",
    ]
    for pair in pairs:
        safe_query = pair.query.replace("|", "\\|")
        lines.append(
            f"| `{pair.case_id}` | {pair.score:.6f} | "
            f"{pair.title} | {safe_query} |"
        )
    lines.append("")
    return lines


def build_report(
    positives: list[CalibrationPair],
    negatives: list[CalibrationPair],
    decision: ThresholdDecision,
    *,
    embedding_api_calls: int,
) -> str:
    """Render complete threshold provenance as reviewable Markdown."""
    positive_margin = decision.min_positive - decision.recommended
    negative_margin = decision.recommended - decision.max_negative
    lines = [
        "# Semantic Threshold Calibration",
        "",
        f"- Run date: {date.today().isoformat()}",
        f"- Embedding model: `{EMBEDDING_MODEL_NAME}`",
        f"- Knowledge base: `{Path(KB_PATH).name}`",
        "- Positive source: `tests/fixtures/retrieval_cases.json` "
        "(answerable cases only)",
        "- Negative source: `tests/fixtures/retrieval_negatives.json` "
        "(intentional calibration set, not held-out)",
        "- Held-out source was not loaded or used for tuning.",
        f"- Embedding provider calls: {embedding_api_calls}",
        "",
        "## Decision",
        "",
        "| measure | value |",
        "|---|---:|",
        f"| min positive | {decision.min_positive:.6f} |",
        f"| max negative | {decision.max_negative:.6f} |",
        f"| gap (`min_positive - max_negative`) | {decision.gap:.6f} |",
        f"| recommended `MIN_COSINE` | **{decision.recommended:.6f}** |",
        f"| pair precision at recommendation | {decision.pair_precision:.3f} |",
        f"| pair recall at recommendation | {decision.pair_recall:.3f} |",
        f"| pair F0.5 at recommendation | {decision.pair_f_beta:.3f} |",
        f"| zero-FP boundary | {decision.zero_fp_threshold:.6f} |",
        f"| positive margin | {positive_margin:.6f} |",
        f"| negative margin | {negative_margin:.6f} |",
        "",
        f"Strategy: **{decision.strategy}**.",
        "",
    ]
    if decision.strategy == "precision-weighted F0.5 sweep (overlap)":
        zero_fp_lost = sum(
            pair.score < decision.zero_fp_threshold
            for pair in positives
        )
        lines += [
            "There is no clean positive/negative separation. A global cosine "
            "gate cannot reject every near-miss while preserving useful "
            "recall. The selected threshold maximizes pair-level F0.5 over "
            "six-decimal deployable boundaries, weighting precision twice as "
            "strongly as recall.",
            "",
            f"The zero-FP boundary `{decision.zero_fp_threshold:.6f}` would "
            f"lose {zero_fp_lost}/{len(positives)} measured positive pairs, "
            "so it is reported as a counterfactual rather than deployed.",
            "",
        ]
    else:
        lines += [
            "A clean separation exists, so the threshold is placed at the "
            "midpoint rather than fitted to either boundary.",
            "",
        ]

    lines += [
        f"Positive pairs lost at this threshold: "
        f"**{len(decision.lost_positive_pairs)}/{len(positives)}**.",
        "",
        f"Hard negatives leaked at this threshold: "
        f"**{len(decision.leaked_negative_pairs)}/{len(negatives)}**.",
        "",
    ]
    if decision.lost_positive_pairs:
        lines += _pair_table(
            "Positive pairs below the recommended threshold",
            sorted(
                decision.lost_positive_pairs,
                key=lambda pair: pair.score,
            ),
        )
    if decision.leaked_negative_pairs:
        lines += _pair_table(
            "Hard negatives still above the recommended threshold",
            sorted(
                decision.leaked_negative_pairs,
                key=lambda pair: pair.score,
                reverse=True,
            ),
        )

    lines += _pair_table(
        "All positive pairs (weakest first)",
        sorted(positives, key=lambda pair: pair.score),
    )
    lines += _pair_table(
        "All hard-negative top hits (strongest first)",
        sorted(negatives, key=lambda pair: pair.score, reverse=True),
    )
    lines += [
        "## Configuration check",
        "",
        f"Configured `MIN_COSINE` at run time: `{MIN_COSINE:.6f}`.",
        "",
        "After changing configuration, rerun both calibration and retrieval "
        "evaluation. The original held-out fixture must remain untouched.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    scorer = SemanticRetriever(
        KB_PATH,
        model_name=EMBEDDING_MODEL_NAME,
        min_cosine=-1.0,
        cache_dir=EMBED_CACHE_DIR,
    )
    try:
        positives, negatives = measure_pairs(
            scorer,
            load_cases("calibration"),
            load_cases("negatives"),
        )
    except (SemanticRetrievalError, ValueError) as exc:
        print(f"ERROR: semantic threshold calibration failed: {exc}", file=sys.stderr)
        return 2

    decision = choose_threshold(positives, negatives)
    RESULTS_PATH.write_text(
        build_report(
            positives,
            negatives,
            decision,
            embedding_api_calls=scorer.embedding_api_calls,
        ),
        encoding="utf-8",
    )
    print(
        f"Wrote {RESULTS_PATH} "
        f"(recommended MIN_COSINE={decision.recommended:.6f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CalibrationPair",
    "RESULTS_PATH",
    "ThresholdDecision",
    "build_report",
    "choose_threshold",
    "measure_pairs",
]
