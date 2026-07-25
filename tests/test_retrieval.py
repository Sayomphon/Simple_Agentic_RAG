"""Offline tests for the deterministic custom retrieval tool."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

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


def _load_retrieval_cases() -> list[dict[str, object]]:
    fixture_path = Path(__file__).parent / "fixtures" / "retrieval_cases.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


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
    def test_golden_retrieval_cases_and_metrics(self) -> None:
        cases = _load_retrieval_cases()
        self.assertGreaterEqual(len(cases), 20)
        self.assertEqual(
            len({str(case["id"]) for case in cases}),
            len(cases),
        )

        category_counts = Counter(
            str(case["category"])
            for case in cases
        )
        minimum_category_counts = {
            "existing_regression": 7,
            "natural_paraphrase": 6,
            "cross_domain_precision": 3,
            "multi_section_recall": 2,
            "unknown_or_generic": 2,
        }
        for category, minimum in minimum_category_counts.items():
            self.assertGreaterEqual(category_counts[category], minimum)

        retrieved_relevant = 0
        retrieved_total = 0
        expected_total = 0
        exact_passes = 0
        rejected_unknown = 0
        unknown_total = 0

        for case in cases:
            case_id = str(case["id"])
            query = str(case["query"])
            expected_titles = list(case["expected_titles"])
            forbidden_titles = list(case["forbidden_titles"])

            with self.subTest(case_id=case_id):
                titles = _titles(search(query))
                self.assertEqual(titles, expected_titles)
                for forbidden_title in forbidden_titles:
                    self.assertNotIn(forbidden_title, titles)

            exact_passes += titles == expected_titles
            expected_set = set(expected_titles)
            retrieved_relevant += sum(
                title in expected_set
                for title in titles
            )
            retrieved_total += len(titles)
            expected_total += len(expected_titles)

            if not expected_titles:
                unknown_total += 1
                rejected_unknown += not titles

        exact_pass_rate = exact_passes / len(cases)
        precision = retrieved_relevant / retrieved_total
        recall = retrieved_relevant / expected_total
        unknown_rejection_rate = rejected_unknown / unknown_total

        self.assertEqual(exact_pass_rate, 1.0)
        self.assertGreaterEqual(precision, 0.95)
        self.assertGreaterEqual(recall, 0.95)
        self.assertEqual(unknown_rejection_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
