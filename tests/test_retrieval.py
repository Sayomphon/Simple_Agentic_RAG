"""Offline tests for the deterministic custom retrieval tool."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.tools.retrieval import (
    KnowledgeBaseFormatError,
    load_knowledge_base,
    search,
    search_knowledge_base,
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
                "--- International Travel Approval Process ---",
                "--- International Travel Daily Allowance ---",
            ],
        )

    def test_partial_matches_retrieve_cross_section_evidence(self) -> None:
        snippets = search_knowledge_base.invoke(
            {"query": "How do I escalate a P1 outage?"}
        )

        self.assertEqual(
            _titles(snippets),
            [
                "--- Customer Support Service Levels ---",
                "--- Support Escalation Process ---",
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


if __name__ == "__main__":
    unittest.main()
