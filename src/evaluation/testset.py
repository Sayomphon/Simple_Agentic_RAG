"""Golden retrieval test set, derived from the KB expansion design.

Each case declares what a correct retrieval looks like BEFORE any mode is
run — the set must never be edited to flatter a result. Categories map to
the retrieval behaviours the KB was designed to probe:

    - "lexical":     exact identifiers (form codes, plan names, extensions);
                     keyword search is expected to win.
    - "semantic":    the user's vocabulary differs from the handbook's
                     official wording; dense retrieval is expected to win.
    - "multi_chunk": the answer is spread across several sections; recall
                     across chunks matters more than any single rank.
    - "negative":    nothing in the KB answers this; the ONLY correct
                     output is an empty result (hallucination guard).
"""

TEST_SET: list[dict[str, object]] = [
    # --- lexical: exact-match identifiers -------------------------------
    {
        "id": "lex_form_hr204",
        "category": "lexical",
        "query": "What is Form HR-204 used for?",
        "expected_titles": ["Employee Referral Program"],
    },
    {
        "id": "lex_safejourney",
        "category": "lexical",
        "query": "SafeJourney Plan B coverage",
        "expected_titles": ["International Travel Insurance"],
    },
    {
        "id": "lex_ap_helpdesk",
        "category": "lexical",
        "query": "Accounts Payable helpdesk extension",
        "expected_titles": ["Vendor Invoice and Payment Terms"],
    },
    {
        "id": "lex_stocksense_price",
        "category": "lexical",
        "query": "StockSense Growth plan price",
        "expected_titles": ["StockSense Product Overview"],
    },
    # --- semantic: vocabulary mismatch ----------------------------------
    {
        "id": "sem_get_paid",
        "category": "semantic",
        "query": "When do I get paid each month?",
        "expected_titles": ["Compensation Disbursement Schedule"],
    },
    {
        "id": "sem_funeral",
        "category": "semantic",
        "query": "funeral leave for my father",
        "expected_titles": ["Bereavement and Compassionate Absence"],
    },
    {
        "id": "sem_dress_code",
        "category": "semantic",
        "query": "what is the dress code",
        "expected_titles": ["Workplace Attire Standards"],
    },
    {
        "id": "sem_quit_job",
        "category": "semantic",
        "query": "I want to quit my job",
        "expected_titles": ["Resignation Process"],
    },
    {
        "id": "sem_counseling",
        "category": "semantic",
        "query": "counseling for stress and burnout",
        "expected_titles": ["Employee Assistance Program"],
    },
    {
        "id": "sem_sla_uptime",
        "category": "semantic",
        "query": "customer SLA uptime compensation",
        "expected_titles": ["Service Credit Policy"],
    },
    # --- multi_chunk: answer spread across sections ---------------------
    {
        "id": "multi_overseas_trip",
        "category": "multi_chunk",
        "query": "everything I need for an overseas business trip",
        "expected_titles": [
            "International Travel Approval Process",
            "International Travel Daily Allowance",
            "International Travel Insurance",
            "International Travel Visa Support",
        ],
    },
    {
        "id": "multi_new_supplier",
        "category": "multi_chunk",
        "query": "how do I buy something from a new supplier",
        "expected_titles": [
            "Purchase Requisition and Purchase Orders",
            "Vendor Onboarding and Registration",
        ],
    },
    {
        "id": "multi_p1_platinum",
        "category": "multi_chunk",
        "query": "P1 response time for platinum support",
        "expected_titles": [
            "Customer Support Service Levels",
            "Support Escalation Process",
        ],
    },
    # --- negative: must return nothing ----------------------------------
    {
        "id": "neg_ceo_salary",
        "category": "negative",
        "query": "What is the CEO's salary?",
        "expected_titles": [],
    },
    {
        "id": "neg_home_addresses",
        "category": "negative",
        "query": "employee home addresses and phone numbers",
        "expected_titles": [],
    },
]
