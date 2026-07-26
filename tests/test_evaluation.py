"""Offline tests for the evaluation dataset loader and pure metrics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation.dataset import (
    DatasetValidationError,
    EvalCase,
    load_all,
    load_cases,
)
from src.evaluation.metrics import (
    QueryOutcome,
    evaluate,
    query_precision,
    query_recall,
    reciprocal_rank,
)


class DatasetLoaderTests(unittest.TestCase):
    def test_calibration_set_loads_with_unique_ids(self) -> None:
        cases = load_cases("calibration")

        self.assertGreaterEqual(len(cases), 23)
        self.assertEqual(len({case.id for case in cases}), len(cases))
        self.assertTrue(all(isinstance(case, EvalCase) for case in cases))

    def test_negative_cases_are_flagged(self) -> None:
        cases = load_cases("calibration")
        negatives = [case for case in cases if case.is_negative]

        self.assertGreaterEqual(len(negatives), 3)
        self.assertTrue(
            all(case.expected_titles == () for case in negatives)
        )

    def test_unknown_dataset_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(DatasetValidationError, "Unknown dataset"):
            load_cases("production")

    def test_load_all_includes_calibration(self) -> None:
        datasets = load_all()

        self.assertIn("calibration", datasets)
        self.assertEqual(
            [case.id for case in datasets["calibration"]],
            [case.id for case in load_cases("calibration")],
        )

    def _load_fixture(self, raw_cases: object) -> list[EvalCase]:
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "broken.json"
            fixture.write_text(json.dumps(raw_cases), encoding="utf-8")
            with (
                patch("src.evaluation.dataset.FIXTURES_DIR", Path(temp_dir)),
                patch.dict(
                    "src.evaluation.dataset.DATASET_FILES",
                    {"broken": "broken.json"},
                ),
            ):
                return load_cases("broken")

    def test_missing_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(DatasetValidationError, "'query'"):
            self._load_fixture(
                [
                    {
                        "id": "x",
                        "category": "c",
                        "expected_titles": [],
                        "forbidden_titles": [],
                    }
                ]
            )

    def test_duplicate_ids_are_rejected(self) -> None:
        case = {
            "id": "same",
            "category": "c",
            "query": "q",
            "expected_titles": [],
            "forbidden_titles": [],
        }
        with self.assertRaisesRegex(DatasetValidationError, "unique"):
            self._load_fixture([case, case])

    def test_expected_and_forbidden_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(DatasetValidationError, "both expected"):
            self._load_fixture(
                [
                    {
                        "id": "x",
                        "category": "c",
                        "query": "q",
                        "expected_titles": ["--- A ---"],
                        "forbidden_titles": ["--- A ---"],
                    }
                ]
            )

    def test_empty_fixture_is_rejected(self) -> None:
        with self.assertRaisesRegex(DatasetValidationError, "non-empty"):
            self._load_fixture([])


class QueryMetricTests(unittest.TestCase):
    def test_precision_counts_only_expected_titles(self) -> None:
        self.assertEqual(query_precision(["a", "b"], ["a"]), 0.5)
        self.assertEqual(query_precision(["a"], ["a", "b"]), 1.0)

    def test_precision_of_empty_retrieval(self) -> None:
        self.assertEqual(query_precision([], []), 1.0)
        self.assertEqual(query_precision([], ["a"]), 0.0)

    def test_recall_counts_expected_coverage(self) -> None:
        self.assertEqual(query_recall(["a"], ["a", "b"]), 0.5)
        self.assertEqual(query_recall(["a", "b"], ["a", "b"]), 1.0)
        self.assertEqual(query_recall([], ["a"]), 0.0)

    def test_recall_rejects_negative_queries(self) -> None:
        with self.assertRaisesRegex(ValueError, "negative query"):
            query_recall(["a"], [])

    def test_reciprocal_rank_uses_first_relevant_position(self) -> None:
        self.assertEqual(reciprocal_rank(["a", "b"], ["a"]), 1.0)
        self.assertEqual(reciprocal_rank(["x", "a"], ["a"]), 0.5)
        self.assertEqual(reciprocal_rank(["x", "y"], ["a"]), 0.0)
        self.assertEqual(reciprocal_rank([], ["a"]), 0.0)


class EvaluateTests(unittest.TestCase):
    def test_hand_computed_mixed_dataset(self) -> None:
        metrics = evaluate(
            [
                # Perfect: exact order and content.
                QueryOutcome(retrieved=("a", "b"), expected=("a", "b")),
                # Wrong order only: set match but not exact match.
                QueryOutcome(retrieved=("d", "c"), expected=("c", "d")),
                # One correct of two retrieved, one of two expected missed.
                QueryOutcome(retrieved=("a", "x"), expected=("a", "e")),
                # Negative query that wrongly returned a section.
                QueryOutcome(retrieved=("x",), expected=()),
                # Negative query correctly rejected.
                QueryOutcome(retrieved=(), expected=()),
            ]
        )

        self.assertEqual(metrics.total_queries, 5)
        self.assertEqual(metrics.positive_queries, 3)
        self.assertEqual(metrics.negative_queries, 2)
        # Exact: cases 1 and 5. Set: cases 1, 2, and 5.
        self.assertEqual(metrics.exact_match, 2 / 5)
        self.assertEqual(metrics.set_match, 3 / 5)
        # Positive precisions: 1.0, 1.0, 0.5 -> macro 2.5/3.
        self.assertAlmostEqual(metrics.precision_macro, 2.5 / 3)
        # Positive recalls: 1.0, 1.0, 0.5 -> macro 2.5/3.
        self.assertAlmostEqual(metrics.recall_macro, 2.5 / 3)
        self.assertAlmostEqual(metrics.f1_macro, 2.5 / 3)
        # Micro: 5 relevant of 6 retrieved and of 6 expected.
        self.assertAlmostEqual(metrics.precision_micro, 5 / 6)
        self.assertAlmostEqual(metrics.recall_micro, 5 / 6)
        # Reciprocal ranks: 1.0, 1.0, 1.0.
        self.assertEqual(metrics.mrr, 1.0)
        # One of two negatives leaked a section.
        self.assertEqual(metrics.fp_rate_negative, 0.5)
        # Case 3 both over-retrieves ("x") and under-retrieves ("e").
        self.assertAlmostEqual(metrics.over_retrieval, 1 / 3)
        self.assertAlmostEqual(metrics.under_retrieval, 1 / 3)

    def test_empty_groups_report_none_not_zero(self) -> None:
        positives_only = evaluate(
            [QueryOutcome(retrieved=("a",), expected=("a",))]
        )
        negatives_only = evaluate([QueryOutcome(retrieved=(), expected=())])

        self.assertIsNone(positives_only.fp_rate_negative)
        self.assertIsNone(negatives_only.precision_macro)
        self.assertIsNone(negatives_only.recall_macro)
        self.assertIsNone(negatives_only.f1_macro)
        self.assertIsNone(negatives_only.mrr)
        self.assertEqual(negatives_only.exact_match, 1.0)

    def test_empty_dataset(self) -> None:
        metrics = evaluate([])

        self.assertEqual(metrics.total_queries, 0)
        self.assertEqual(metrics.exact_match, 0.0)
        self.assertIsNone(metrics.precision_macro)
        self.assertIsNone(metrics.fp_rate_negative)

    def test_no_intersection_scores_zero(self) -> None:
        metrics = evaluate(
            [QueryOutcome(retrieved=("x", "y"), expected=("a", "b"))]
        )

        self.assertEqual(metrics.precision_macro, 0.0)
        self.assertEqual(metrics.recall_macro, 0.0)
        self.assertEqual(metrics.f1_macro, 0.0)
        self.assertEqual(metrics.mrr, 0.0)


if __name__ == "__main__":
    unittest.main()
