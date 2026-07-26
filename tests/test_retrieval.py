"""Offline tests for the deterministic custom retrieval tool."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.evaluation.dataset import load_cases
from src.evaluation.metrics import QueryOutcome, evaluate
from src.tools.retrieval import (
    BODY_MATCH_WEIGHT,
    KnowledgeBaseFormatError,
    TITLE_MATCH_WEIGHT,
    inverse_document_frequency,
    is_candidate,
    load_knowledge_base,
    normalize_phrases,
    normalized_tokens,
    search,
    search_knowledge_base,
    score_chunk,
)


def _titles(snippets: list[str]) -> list[str]:
    return [snippet.splitlines()[0] for snippet in snippets]


class RetrievalTests(unittest.TestCase):
    def test_load_knowledge_base(self) -> None:
        chunks = load_knowledge_base()

        self.assertEqual(len(chunks), 10)
        self.assertTrue(chunks[0].startswith("--- Remote Work Policy ---"))
        self.assertTrue(chunks[-1].startswith("--- Support Escalation Process ---"))

    def test_keyword_search_returns_relevant_snippets(self) -> None:
        snippets = search_knowledge_base.invoke(
            {"query": "What is the policy on international travel?"}
        )

        self.assertGreaterEqual(len(snippets), 1)
        self.assertTrue(
            all("International Travel" in title for title in _titles(snippets))
        )

    def test_keyword_search_returns_empty_for_unknown_query(self) -> None:
        snippets = search_knowledge_base.invoke(
            {"query": "What is the CEO's salary?"}
        )

        self.assertEqual(snippets, [])

    def test_stopwords_do_not_create_false_positive(self) -> None:
        snippets = search_knowledge_base.invoke(
            {"query": "What is the a an of on can I?"}
        )

        self.assertEqual(snippets, [])

    def test_generic_domain_terms_do_not_return_unrelated_sections(self) -> None:
        generic_only = search_knowledge_base.invoke(
            {"query": "What company policy information is available?"}
        )
        focused = search_knowledge_base.invoke(
            {"query": "international travel policy"}
        )

        self.assertEqual(generic_only, [])
        self.assertTrue(focused)
        self.assertTrue(
            all("International Travel" in title for title in _titles(focused))
        )

    def test_keyword_search_returns_all_relevant_snippets(self) -> None:
        snippets = search_knowledge_base.invoke({"query": "international travel"})

        self.assertEqual(
            _titles(snippets),
            [
                "--- International Travel Approval Process ---",
                "--- International Travel Daily Allowance ---",
                "--- International Travel Insurance ---",
            ],
        )

    def test_result_order_is_deterministic(self) -> None:
        first = search_knowledge_base.invoke({"query": "international travel"})
        second = search_knowledge_base.invoke({"query": "international travel"})

        self.assertEqual(first, second)

    def test_multi_term_query_rejects_single_term_title_matches(self) -> None:
        snippets = search_knowledge_base.invoke(
            {"query": "What is the international card fee?"}
        )

        self.assertEqual(
            _titles(snippets),
            ["--- PaySiam Gateway Product Overview ---"],
        )

    def test_specific_query_requires_stronger_term_coverage(self) -> None:
        snippets = search_knowledge_base.invoke(
            {"query": "What is the international travel insurance coverage?"}
        )

        self.assertEqual(
            _titles(snippets),
            ["--- International Travel Insurance ---"],
        )

    def test_verbose_multi_intent_query_returns_all_title_aligned_sections(
        self,
    ) -> None:
        snippets = search_knowledge_base.invoke(
            {
                "query": (
                    "Summarize all international travel rules including "
                    "approval, booking, allowance, lodging, insurance, "
                    "and claims."
                )
            }
        )

        self.assertEqual(
            _titles(snippets),
            [
                "--- International Travel Insurance ---",
                "--- International Travel Daily Allowance ---",
                "--- International Travel Approval Process ---",
            ],
        )

    def test_partial_matches_retrieve_cross_section_evidence(self) -> None:
        snippets = search_knowledge_base.invoke(
            {"query": "How do I escalate a P1 outage?"}
        )

        self.assertEqual(
            _titles(snippets),
            [
                "--- Support Escalation Process ---",
                "--- Customer Support Service Levels ---",
            ],
        )

    def test_two_term_query_keeps_title_linked_sections(self) -> None:
        snippets = search_knowledge_base.invoke({"query": "Can I work remotely?"})

        self.assertEqual(
            _titles(snippets),
            [
                "--- Remote Work Policy ---",
                "--- Hybrid Work Guidelines ---",
            ],
        )

    def test_two_term_query_does_not_expand_without_full_title_anchor(self) -> None:
        snippets = search_knowledge_base.invoke({"query": "international card"})

        self.assertEqual(
            _titles(snippets),
            ["--- PaySiam Gateway Product Overview ---"],
        )

    def test_empty_knowledge_base_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.txt"
            path.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(
                KnowledgeBaseFormatError,
                "Knowledge base is empty",
            ) as context:
                load_knowledge_base(path)

            self.assertIn(str(path), str(context.exception))

    def test_whitespace_only_knowledge_base_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "whitespace.txt"
            path.write_text("  \n\t\n", encoding="utf-8")

            with self.assertRaises(KnowledgeBaseFormatError) as context:
                load_knowledge_base(path)

            self.assertIn(str(path), str(context.exception))

    def test_nonempty_file_without_section_header_is_rejected(self) -> None:
        sensitive_content = "confidential corpus payload"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "malformed.txt"
            path.write_text(sensitive_content, encoding="utf-8")

            with self.assertRaises(KnowledgeBaseFormatError) as context:
                load_knowledge_base(path)

            message = str(context.exception)
            self.assertIn(str(path), message)
            self.assertNotIn(sensitive_content, message)

    def test_section_without_body_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing-body.txt"
            path.write_text("--- Empty Section ---\n  ", encoding="utf-8")

            with self.assertRaises(KnowledgeBaseFormatError) as context:
                load_knowledge_base(path)

            message = str(context.exception)
            self.assertIn(str(path), message)
            self.assertIn("Empty Section", message)

    def test_valid_one_section_file_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "one-section.txt"
            path.write_text(
                "--- One Section ---\nOne section body.\n",
                encoding="utf-8",
            )

            self.assertEqual(
                load_knowledge_base(path),
                ["--- One Section ---\nOne section body."],
            )

    def test_valid_multi_section_file_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "multi-section.txt"
            path.write_text(
                "--- First ---\nFirst body.\n\n"
                "--- Second ---\nSecond body.\n",
                encoding="utf-8",
            )

            self.assertEqual(
                _titles(load_knowledge_base(path)),
                ["--- First ---", "--- Second ---"],
            )

    def test_search_validates_corpus_before_returning_no_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "malformed.txt"
            path.write_text("not a section", encoding="utf-8")

            with self.assertRaises(KnowledgeBaseFormatError):
                search("the company policy", path)

    def test_missing_knowledge_base_preserves_file_not_found_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing.txt"

            with self.assertRaises(FileNotFoundError) as context:
                load_knowledge_base(path)

            self.assertIn(str(path), str(context.exception))

    def test_default_kb_path_is_independent_of_working_directory(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)
                chunks = load_knowledge_base()
                results = search("international travel")
            finally:
                os.chdir(original_cwd)

        self.assertEqual(len(chunks), 10)
        self.assertTrue(results)

    def test_kb_path_env_override_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            override = Path(temp_dir) / "override.txt"
            override.write_text(
                "--- Override Section ---\nOverride body.\n",
                encoding="utf-8",
            )
            script = (
                "from src.tools.retrieval import load_knowledge_base; "
                "print(load_knowledge_base()[0].splitlines()[0])"
            )
            completed = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "KB_PATH": str(override)},
                check=True,
            )

        self.assertEqual(completed.stdout.strip(), "--- Override Section ---")


class RetrievalNormalizationTests(unittest.TestCase):
    def test_phrase_aliases_preserve_reviewed_multi_word_concepts(self) -> None:
        cases = {
            "work from home": {"remote", "work"},
            "per diem": {"daily", "allowance"},
            "paid time off": {"paid", "leave"},
            "overseas business trip": {
                "international",
                "travel",
            },
        }

        for text, expected_terms in cases.items():
            with self.subTest(text=text):
                self.assertTrue(
                    expected_terms
                    <= normalized_tokens(text, is_query=True)
                )

    def test_longer_phrase_alias_is_applied_before_overlapping_alias(self) -> None:
        self.assertEqual(
            normalize_phrases("Overseas business trip"),
            "international business travel",
        )

    def test_token_aliases_canonicalize_domain_variants(self) -> None:
        self.assertEqual(
            normalized_tokens(
                "remotely vacation entitlements methods accepted",
                is_query=True,
            ),
            frozenset(
                {
                    "remote",
                    "leave",
                    "entitlement",
                    "method",
                    "accept",
                }
            ),
        )

    def test_query_framing_terms_are_removed_only_from_queries(self) -> None:
        text = "How many remote days are allowed each week?"

        query_terms = normalized_tokens(text, is_query=True)
        document_terms = normalized_tokens(text, is_query=False)

        self.assertEqual(query_terms, frozenset({"remote", "days", "week"}))
        self.assertTrue(
            {"many", "allowed", "each"} <= document_terms
        )

    def test_inverse_document_frequency_rewards_rare_terms(self) -> None:
        documents = [
            frozenset({"international", "insurance"}),
            frozenset({"international", "allowance"}),
            frozenset({"international", "approval"}),
        ]

        rare = inverse_document_frequency("insurance", documents)
        common = inverse_document_frequency("international", documents)

        self.assertGreater(rare, common)
        self.assertEqual(
            rare,
            inverse_document_frequency("insurance", documents),
        )

    def test_inverse_document_frequency_rejects_empty_corpus(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one document"):
            inverse_document_frequency("term", [])

    def test_score_chunk_counts_title_match_only_once(self) -> None:
        score, matched_terms, title_matches = score_chunk(
            frozenset({"remote", "days"}),
            frozenset({"remote"}),
            frozenset({"remote", "days"}),
            {"remote": 2.0, "days": 3.0},
        )

        self.assertEqual(matched_terms, frozenset({"remote", "days"}))
        self.assertEqual(title_matches, frozenset({"remote"}))
        self.assertEqual(
            score,
            (2.0 * TITLE_MATCH_WEIGHT) + (3.0 * BODY_MATCH_WEIGHT),
        )

    def test_candidate_requires_title_anchor_or_two_matches(self) -> None:
        self.assertTrue(
            is_candidate(frozenset({"remote"}), frozenset({"remote"}))
        )
        self.assertTrue(
            is_candidate(frozenset({"p1", "outage"}), frozenset())
        )
        self.assertFalse(
            is_candidate(frozenset({"international"}), frozenset())
        )


class RetrievalEvaluationTests(unittest.TestCase):
    def test_calibration_retrieval_cases_and_metrics(self) -> None:
        cases = load_cases("calibration")
        self.assertGreaterEqual(len(cases), 20)

        category_counts = Counter(case.category for case in cases)
        minimum_category_counts = {
            "existing_regression": 7,
            "natural_paraphrase": 6,
            "cross_domain_precision": 3,
            "multi_section_recall": 2,
            "unknown_or_generic": 2,
        }
        for category, minimum in minimum_category_counts.items():
            self.assertGreaterEqual(category_counts[category], minimum)

        outcomes: list[QueryOutcome] = []
        for case in cases:
            with self.subTest(case_id=case.id):
                titles = _titles(search(case.query))
                self.assertEqual(tuple(titles), case.expected_titles)
                for forbidden_title in case.forbidden_titles:
                    self.assertNotIn(forbidden_title, titles)

            outcomes.append(
                QueryOutcome(
                    retrieved=tuple(titles),
                    expected=case.expected_titles,
                )
            )

        metrics = evaluate(outcomes)
        self.assertEqual(metrics.exact_match, 1.0)
        self.assertGreaterEqual(metrics.precision_micro, 0.95)
        self.assertGreaterEqual(metrics.recall_micro, 0.95)
        self.assertEqual(metrics.fp_rate_negative, 0.0)


if __name__ == "__main__":
    unittest.main()
