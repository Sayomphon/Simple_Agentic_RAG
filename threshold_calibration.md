# Semantic Threshold Calibration

- Run date: 2026-07-26
- Embedding model: `text-embedding-3-small`
- Knowledge base: `knowledge_base.txt`
- Positive source: `tests/fixtures/retrieval_cases.json` (answerable cases only)
- Negative source: `tests/fixtures/retrieval_negatives.json` (intentional calibration set, not held-out)
- Held-out source was not loaded or used for tuning.
- Embedding provider calls: 36

## Decision

| measure | value |
|---|---:|
| min positive | 0.223143 |
| max negative | 0.757588 |
| gap (`min_positive - max_negative`) | -0.534445 |
| recommended `MIN_COSINE` | **0.392817** |
| pair precision at recommendation | 0.875 |
| pair recall at recommendation | 0.921 |
| pair F0.5 at recommendation | 0.884 |
| zero-FP boundary | 0.757589 |
| positive margin | -0.169674 |
| negative margin | -0.364771 |

Strategy: **precision-weighted F0.5 sweep (overlap)**.

There is no clean positive/negative separation. A global cosine gate cannot reject every near-miss while preserving useful recall. The selected threshold maximizes pair-level F0.5 over six-decimal deployable boundaries, weighting precision twice as strongly as recall.

The zero-FP boundary `0.757589` would lose 38/38 measured positive pairs, so it is reported as a counterfactual rather than deployed.

Positive pairs lost at this threshold: **3/38**.

Hard negatives leaked at this threshold: **5/12**.

## Positive pairs below the recommended threshold

| case | score | section | query |
|---|---:|---|---|
| `international_card_focused` | 0.223143 | --- PaySiam Gateway Product Overview --- | international card |
| `international_card_fee` | 0.379605 | --- PaySiam Gateway Product Overview --- | What is the international card fee? |
| `morph_submitted_receipts` | 0.383843 | --- International Travel Daily Allowance --- | submitted expense receipts |

## Hard negatives still above the recommended threshold

| case | score | section | query |
|---|---:|---|---|
| `neg_domestic_travel_allowance` | 0.757588 | --- International Travel Daily Allowance --- | What is the domestic travel daily allowance in Thailand? |
| `neg_paysiam_chargebacks` | 0.601695 | --- PaySiam Gateway Product Overview --- | How does PaySiam handle card chargebacks? |
| `neg_paysiam_refund_fee` | 0.592364 | --- PaySiam Gateway Product Overview --- | What is PaySiam's refund processing fee? |
| `neg_maternity_leave_duration` | 0.545377 | --- Annual Leave --- | How many days of maternity leave do employees receive? |
| `neg_sick_leave_carryover` | 0.484568 | --- Annual Leave --- | Can unused sick leave be carried into next year? |

## All positive pairs (weakest first)

| case | score | section | query |
|---|---:|---|---|
| `international_card_focused` | 0.223143 | --- PaySiam Gateway Product Overview --- | international card |
| `international_card_fee` | 0.379605 | --- PaySiam Gateway Product Overview --- | What is the international card fee? |
| `morph_submitted_receipts` | 0.383843 | --- International Travel Daily Allowance --- | submitted expense receipts |
| `morph_card_fees` | 0.392818 | --- PaySiam Gateway Product Overview --- | What are the card fees? |
| `remote_general` | 0.425910 | --- Hybrid Work Guidelines --- | Can I work remotely? |
| `morph_escalating_p1` | 0.437217 | --- Customer Support Service Levels --- | escalating a P1 outage |
| `morph_reimbursing_hotel` | 0.438835 | --- International Travel Daily Allowance --- | reimbursing hotel receipts |
| `support_multi_section` | 0.444682 | --- Customer Support Service Levels --- | escalate a P1 outage |
| `travel_multi_section_short` | 0.450753 | --- International Travel Daily Allowance --- | international travel |
| `travel_all` | 0.459422 | --- International Travel Daily Allowance --- | What is the policy on international travel? |
| `southeast_asia_hotel_cap` | 0.465325 | --- International Travel Daily Allowance --- | What is the hotel cap in Southeast Asia? |
| `verbose_travel_summary` | 0.465526 | --- International Travel Daily Allowance --- | Summarize all international travel rules including approval, booking, allowance, lodging, insurance, and claims. |
| `reimbursement_deadline` | 0.473438 | --- Expense Reimbursement --- | What is the reimbursement deadline? |
| `travel_multi_section_short` | 0.482573 | --- International Travel Insurance --- | international travel |
| `p2_response_time` | 0.485972 | --- Customer Support Service Levels --- | What is the response time for a P2 major degradation? |
| `morph_reimbursing_hotel` | 0.492181 | --- Expense Reimbursement --- | reimbursing hotel receipts |
| `p1_cross_section` | 0.495436 | --- Customer Support Service Levels --- | How do I escalate a P1 outage? |
| `verbose_travel_summary` | 0.503210 | --- International Travel Insurance --- | Summarize all international travel rules including approval, booking, allowance, lodging, insurance, and claims. |
| `morph_escalating_p1` | 0.503272 | --- Support Escalation Process --- | escalating a P1 outage |
| `travel_multi_intent_titles` | 0.523618 | --- International Travel Insurance --- | international travel approval, allowance, and insurance requirements |
| `support_multi_section` | 0.527547 | --- Support Escalation Process --- | escalate a P1 outage |
| `travel_all` | 0.528404 | --- International Travel Insurance --- | What is the policy on international travel? |
| `verbose_travel_summary` | 0.528682 | --- International Travel Approval Process --- | Summarize all international travel rules including approval, booking, allowance, lodging, insurance, and claims. |
| `travel_multi_intent_titles` | 0.529015 | --- International Travel Daily Allowance --- | international travel approval, allowance, and insurance requirements |
| `remote_days_natural` | 0.553549 | --- Remote Work Policy --- | How many remote days are allowed? |
| `travel_all` | 0.560866 | --- International Travel Approval Process --- | What is the policy on international travel? |
| `travel_multi_section_short` | 0.566608 | --- International Travel Approval Process --- | international travel |
| `p1_cross_section` | 0.567319 | --- Support Escalation Process --- | How do I escalate a P1 outage? |
| `work_from_home_phrase` | 0.574512 | --- Remote Work Policy --- | How many days can an employee work from home each week? |
| `morph_submitted_receipts` | 0.580762 | --- Expense Reimbursement --- | submitted expense receipts |
| `vacation_entitlements` | 0.588454 | --- Annual Leave --- | What are the annual vacation entitlements? |
| `paid_time_off` | 0.591218 | --- Annual Leave --- | How much paid time off do full-time staff receive? |
| `remote_general` | 0.596805 | --- Remote Work Policy --- | Can I work remotely? |
| `overseas_per_diem` | 0.605604 | --- International Travel Daily Allowance --- | How much is the overseas business trip per diem? |
| `travel_insurance_coverage` | 0.633559 | --- International Travel Insurance --- | What is the international travel insurance coverage? |
| `travel_multi_intent_titles` | 0.637977 | --- International Travel Approval Process --- | international travel approval, allowance, and insurance requirements |
| `paysiam_international_cards` | 0.669587 | --- PaySiam Gateway Product Overview --- | Does PaySiam support international cards? |
| `paysiam_payment_methods` | 0.682466 | --- PaySiam Gateway Product Overview --- | What payment methods does PaySiam accept? |

## All hard-negative top hits (strongest first)

| case | score | section | query |
|---|---:|---|---|
| `neg_domestic_travel_allowance` | 0.757588 | --- International Travel Daily Allowance --- | What is the domestic travel daily allowance in Thailand? |
| `neg_paysiam_chargebacks` | 0.601695 | --- PaySiam Gateway Product Overview --- | How does PaySiam handle card chargebacks? |
| `neg_paysiam_refund_fee` | 0.592364 | --- PaySiam Gateway Product Overview --- | What is PaySiam's refund processing fee? |
| `neg_maternity_leave_duration` | 0.545377 | --- Annual Leave --- | How many days of maternity leave do employees receive? |
| `neg_sick_leave_carryover` | 0.484568 | --- Annual Leave --- | Can unused sick leave be carried into next year? |
| `neg_spacely_cancellation` | 0.387548 | --- Hybrid Work Guidelines --- | What is SpaceLy's desk reservation cancellation policy? |
| `neg_employee_health_insurance` | 0.373145 | --- International Travel Insurance --- | What does the employee health insurance plan cover? |
| `neg_travelhub_password_reset` | 0.339496 | --- International Travel Approval Process --- | How do I reset my TravelHub password? |
| `neg_customer_refund_sla` | 0.338313 | --- Customer Support Service Levels --- | What is the customer refund SLA? |
| `neg_vpn_account_unlock` | 0.334368 | --- Remote Work Policy --- | How do I unlock a disabled TigerLink VPN account? |
| `neg_manager_phone_number` | 0.325575 | --- Remote Work Policy --- | What is my line manager's mobile phone number? |
| `neg_ceo_salary` | 0.276405 | --- Annual Leave --- | What is the CEO's annual salary? |

## Configuration check

Configured `MIN_COSINE` at run time: `0.392817`.

After changing configuration, rerun both calibration and retrieval evaluation. The original held-out fixture must remain untouched.
