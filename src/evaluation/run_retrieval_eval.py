"""Offline retrieval evaluation: all ablation variants over all datasets.

Usage:
    python -m src.evaluation.run_retrieval_eval

Writes evaluation_results.md at the project root and exits non-zero when
the current variant misses its calibration thresholds, so the command can
serve as a CI gate. Requires no API key and makes no network calls.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

from src.evaluation.ablation import build_variants
from src.evaluation.dataset import EvalCase, load_all
from src.evaluation.metrics import QueryOutcome, RetrievalMetrics, evaluate
from src.tools.retrieval import DEFAULT_SETTINGS, RetrievalSettings, search

RESULTS_PATH = Path(__file__).resolve().parents[2] / "evaluation_results.md"

# The ablation ladder's last rung must equal production behavior; it is
# also the variant the CI thresholds below are checked against.
CURRENT_VARIANT = "V5_current_+stemming"

CALIBRATION_THRESHOLDS = {
    "exact_match": 1.0,
    "fp_rate_negative": 0.0,
}


def _titles(snippets: list[str]) -> tuple[str, ...]:
    return tuple(snippet.splitlines()[0] for snippet in snippets)


def _run_variant(
    cases: list[EvalCase],
    settings: RetrievalSettings,
) -> tuple[RetrievalMetrics, list[tuple[EvalCase, tuple[str, ...]]], list[float]]:
    """Score one variant over one dataset, returning failures and latencies."""
    outcomes: list[QueryOutcome] = []
    failures: list[tuple[EvalCase, tuple[str, ...]]] = []
    latencies_ms: list[float] = []
    for case in cases:
        started = time.perf_counter()
        retrieved = _titles(search(case.query, settings=settings))
        latencies_ms.append((time.perf_counter() - started) * 1000)
        outcomes.append(
            QueryOutcome(retrieved=retrieved, expected=case.expected_titles)
        )
        if retrieved != case.expected_titles:
            failures.append((case, retrieved))
    return evaluate(outcomes), failures, latencies_ms


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def _dataset_section(
    name: str,
    cases: list[EvalCase],
    variants: dict[str, RetrievalSettings],
) -> tuple[list[str], RetrievalMetrics]:
    lines = [f"## Dataset: {name} (n={len(cases)})", ""]

    lines += [
        "| variant | exact | set | P_macro | R_macro | F1 | P_micro "
        "| R_micro | MRR | FP_neg | over | under |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    current_metrics: RetrievalMetrics | None = None
    current_failures: list[tuple[EvalCase, tuple[str, ...]]] = []
    current_latencies: list[float] = []
    for variant_name, settings in variants.items():
        metrics, failures, latencies = _run_variant(cases, settings)
        if variant_name == CURRENT_VARIANT:
            current_metrics = metrics
            current_failures = failures
            current_latencies = latencies
        lines.append(
            f"| {variant_name} | {_percent(metrics.exact_match)} "
            f"| {_percent(metrics.set_match)} "
            f"| {_percent(metrics.precision_macro)} "
            f"| {_percent(metrics.recall_macro)} "
            f"| {_ratio(metrics.f1_macro)} "
            f"| {_percent(metrics.precision_micro)} "
            f"| {_percent(metrics.recall_micro)} "
            f"| {_ratio(metrics.mrr)} "
            f"| {_percent(metrics.fp_rate_negative)} "
            f"| {_percent(metrics.over_retrieval)} "
            f"| {_percent(metrics.under_retrieval)} |"
        )
    assert current_metrics is not None, (
        f"ablation ladder is missing the current variant {CURRENT_VARIANT!r}"
    )

    lines += [
        "",
        f"Current variant `{CURRENT_VARIANT}` latency: "
        f"p50 {_percentile(current_latencies, 0.50):.2f} ms, "
        f"p95 {_percentile(current_latencies, 0.95):.2f} ms "
        "(single process, local file scan).",
        "",
    ]

    if current_failures:
        lines += [f"### `{CURRENT_VARIANT}` mismatches on {name}", ""]
        for case, retrieved in current_failures:
            lines += [
                f"- `{case.id}` — query: {case.query!r}",
                f"  - expected: {list(case.expected_titles)}",
                f"  - retrieved: {list(retrieved)}",
            ]
        lines.append("")
    else:
        lines += [
            f"`{CURRENT_VARIANT}` matches every expected title list on "
            f"{name} exactly.",
            "",
        ]
    return lines, current_metrics


def main() -> int:
    variants = build_variants()
    if CURRENT_VARIANT in variants:
        mismatch = variants[CURRENT_VARIANT] != DEFAULT_SETTINGS
    else:
        mismatch = True
    if mismatch:
        print(
            f"ERROR: variant {CURRENT_VARIANT!r} must exist and equal "
            "DEFAULT_SETTINGS; the ablation ladder has drifted from "
            "production behavior.",
            file=sys.stderr,
        )
        return 1

    datasets = load_all()
    lines = [
        "# Retrieval Evaluation Results",
        "",
        "Generated by `python -m src.evaluation.run_retrieval_eval` "
        "(offline, deterministic, no API key).",
        "",
        "Negative queries (empty expectation) are excluded from "
        "precision/recall/MRR and scored only by the negative "
        "false-positive rate. Latency varies run to run; every other "
        "number is reproducible bit for bit.",
        "",
    ]

    gate_failed = False
    for name, cases in datasets.items():
        section_lines, current_metrics = _dataset_section(
            name, cases, variants
        )
        lines += section_lines
        if name == "calibration":
            for metric_name, threshold in CALIBRATION_THRESHOLDS.items():
                value = getattr(current_metrics, metric_name)
                meets = (
                    value >= threshold
                    if metric_name == "exact_match"
                    else value <= threshold
                )
                if not meets:
                    gate_failed = True
                    print(
                        f"GATE FAILURE: calibration {metric_name} = {value} "
                        f"(threshold {threshold})",
                        file=sys.stderr,
                    )

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    return 1 if gate_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
