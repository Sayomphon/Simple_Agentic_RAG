"""Offline tests for the deterministic custom retrieval tool."""

from __future__ import annotations

import unittest

from src.tools.retrieval import load_knowledge_base, search_knowledge_base


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


if __name__ == "__main__":
    unittest.main()
