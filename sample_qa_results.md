# Sample Q→A Results (end-to-end)

- Run date: 2026-07-25 15:37
- Pipeline: full agentic graph via `build_graph()` — router → data retriever (with query-rewrite retry loop) → report generator
- Search mode: **hybrid**  ·  model: `gpt-5-mini`  ·  `TOP_K=4`  ·  `MAX_SEARCH_ATTEMPTS=3`
- Questions: 17 — the 15 golden-set queries (lexical, semantic, multi-chunk, negative) plus a Thai query and a greeting

Each block records the agent's actual decisions: the route, every search attempt (a new attempt means the previous one returned nothing and the query was rewritten), the evidence handed to the generator, and the final answer verbatim.

## Q1. What is Form HR-204 used for?

- category: `lexical`  ·  route: `kb_query`  ·  14.5s
- search attempts:
  1. "What is Form HR-204 used for?" → 2 result(s)
- retrieved evidence:
  - **Employee Referral Program** (score 0.0164, bm25, `01_hr_policies.txt`)
  - **Petty Cash Policy** (score 0.0161, bm25, `02_travel_and_expenses.txt`)
- answer:
  > Form HR-204 is used to register an employee referral before the candidate's first interview in SiamPeople, naming the candidate and the vacancy. [Employee Referral Program]

## Q2. SafeJourney Plan B coverage

- category: `lexical`  ·  route: `kb_query`  ·  19.7s
- search attempts:
  1. "SafeJourney Plan B coverage" → 1 result(s)
- retrieved evidence:
  - **International Travel Insurance** (score 0.0328, bm25+dense, `02_travel_and_expenses.txt`)
- answer:
  > - SafeJourney Plan B (group travel insurance) automatically covers all international business trips and is underwritten by Bangkok Assurance PCL. [International Travel Insurance]  
  > - Emergency medical treatment is covered up to 3,000,000 THB. [International Travel Insurance]  
  > - Emergency evacuation is included. [International Travel Insurance]  
  > - Lost-luggage compensation is covered up to 40,000 THB per trip. [International Travel Insurance]  
  > - Employees must carry the digital insurance card, available in the HR mobile app (SiamPeople) under "My Benefits". [International Travel Insurance]  
  > - Claims must be filed through the SafeJourney portal within 30 days of returning, with original receipts and a copy of the trip approval from TravelHub. [International Travel Insurance]  
  > - Personal side-trips extending a business trip are not covered beyond 48 hours after the last business day. [International Travel Insurance]

## Q3. Accounts Payable helpdesk extension

- category: `lexical`  ·  route: `kb_query`  ·  10.6s
- search attempts:
  1. "Accounts Payable helpdesk extension" → 3 result(s)
- retrieved evidence:
  - **Vendor Invoice and Payment Terms** (score 0.0323, bm25+dense, `02_travel_and_expenses.txt`)
  - **Expense Reimbursement** (score 0.0164, dense, `02_travel_and_expenses.txt`)
  - **Purchase Requisition and Purchase Orders** (score 0.0161, dense, `02_travel_and_expenses.txt`)
- answer:
  > The Accounts Payable helpdesk can be reached at internal extension 5-3300 for status questions. [Vendor Invoice and Payment Terms]

## Q4. StockSense Growth plan price

- category: `lexical`  ·  route: `kb_query`  ·  15.8s
- search attempts:
  1. "StockSense Growth plan price" → 1 result(s)
- retrieved evidence:
  - **StockSense Product Overview** (score 0.0328, bm25+dense, `03_products_and_support.txt`)
- answer:
  > The StockSense Growth plan costs 12,900 THB per month. [StockSense Product Overview]
  > 
  > It covers up to 10 branches and 50,000 SKUs. [StockSense Product Overview]

## Q5. When do I get paid each month?

- category: `semantic`  ·  route: `kb_query`  ·  27.4s
- search attempts:
  1. "When do I get paid each month?" → 4 result(s)
- retrieved evidence:
  - **Compensation Disbursement Schedule** (score 0.0320, bm25+dense, `01_hr_policies.txt`)
  - **Parental Leave** (score 0.0164, bm25, `01_hr_policies.txt`)
  - **International Travel Daily Allowance** (score 0.0161, bm25, `02_travel_and_expenses.txt`)
  - **Company Holidays** (score 0.0159, bm25, `01_hr_policies.txt`)
- answer:
  > - Your monthly salary is disbursed on the 28th calendar day of each month. [Compensation Disbursement Schedule]  
  > - If the 28th falls on a weekend or public holiday, disbursement moves to the preceding business day. [Compensation Disbursement Schedule]  
  > - Funds are transferred to the Thai bank account registered in SiamPeople, and itemized statements appear in SiamPeople under "My Compensation" one day before disbursement. [Compensation Disbursement Schedule]  
  > - Overtime recorded and approved in TimeTrack by the 15th of the month is included in that month's disbursement; later approvals roll into the following month. [Compensation Disbursement Schedule]  
  > - Off-cycle corrections are processed only for discrepancies exceeding 1,000 THB; smaller differences are adjusted in the following month. [Compensation Disbursement Schedule]  
  > - (If relevant) For approved international business trips, daily allowances are deposited together with the following month's payroll. [International Travel Daily Allowance]

## Q6. funeral leave for my father

- category: `semantic`  ·  route: `kb_query`  ·  18.8s
- search attempts:
  1. "funeral leave for my father" → 3 result(s)
- retrieved evidence:
  - **Parental Leave** (score 0.0164, bm25, `01_hr_policies.txt`)
  - **Bereavement and Compassionate Absence** (score 0.0164, dense, `01_hr_policies.txt`)
  - **Annual Leave** (score 0.0161, dense, `01_hr_policies.txt`)
- answer:
  > - You are entitled to up to 7 working days of compassionate absence at full pay for the passing of a parent. [Bereavement and Compassionate Absence]  
  > - Record the absence in LeaveDesk under the "Compassionate" category. [Bereavement and Compassionate Absence]  
  > - HR may request supporting documents after the leave rather than in advance. [Bereavement and Compassionate Absence]  
  > - If you must travel outside Bangkok for ceremonies, you may combine compassionate absence with other accrued time off with manager approval; additional unpaid days can be granted by the HR Business Partner. [Bereavement and Compassionate Absence]  
  > - To arrange a company wreath, contact the People Care team through SiamPeople. [Bereavement and Compassionate Absence]

## Q7. what is the dress code

- category: `semantic`  ·  route: `kb_query`  ·  14.6s
- search attempts:
  1. "what is the dress code" → 1 result(s)
- retrieved evidence:
  - **Workplace Attire Standards** (score 0.0164, dense, `01_hr_policies.txt`)
- answer:
  > Standard working attire at Siam Innovate offices is business casual (collared shirts, blouses, polos, chinos, skirts, or tailored trousers). [Workplace Attire Standards]
  > 
  > - Denim without rips is acceptable on Fridays together with the company polo. [Workplace Attire Standards]  
  > - Formal business attire (a suit or jacket) is expected when meeting clients, whether at a client site or when hosting visitors at the office. [Workplace Attire Standards]  
  > - T-shirts with slogans, shorts, sandals, and sportswear are not appropriate on office days, except during MoveFest and team sports activities. [Workplace Attire Standards]  
  > - Closed-toe footwear is required in the server room and the facilities workshop. [Workplace Attire Standards]  
  > - Teams with frequent client-facing duties may define stricter norms with HR agreement. [Workplace Attire Standards]  
  > - Religious and cultural garments are always welcome. [Workplace Attire Standards]

## Q8. I want to quit my job

- category: `semantic`  ·  route: `kb_query`  ·  33.5s
- search attempts:
  1. "I want to quit my job" → 0 result(s)
  2. "Employee resignation procedure and required notice period" → 4 result(s)
- retrieved evidence:
  - **Resignation Process** (score 0.0328, bm25+dense, `01_hr_policies.txt`)
  - **Probation Period** (score 0.0315, bm25+dense, `01_hr_policies.txt`)
  - **Annual Leave** (score 0.0161, dense, `01_hr_policies.txt`)
  - **Bereavement and Compassionate Absence** (score 0.0159, dense, `01_hr_policies.txt`)
- answer:
  > Here are the steps and key points for resigning:
  > 
  > - Give at least 30 days' written notice submitted through SiamPeople with a signed resignation letter. [Resignation Process]  
  > - If you are still in your 119-day probation period, either party may terminate with 7 days' written notice. [Probation Period]  
  > - HR will confirm your final working date after checking outstanding leave and project handover plans. [Resignation Process]  
  > - Complete the handover document and have it approved by your manager. [Resignation Process]  
  > - Return all company equipment and access cards to IT and Facilities. [Resignation Process]  
  > - Attend an exit interview with the HR Business Partner in your final week. [Resignation Process]  
  > - Any unused annual leave within the statutory entitlement will be paid out with your final salary. [Resignation Process]  
  > - Access to all systems, including SiamPeople and ExpenseFlow, is revoked at 18:00 on your last working day; submit any pending expense claims before then. [Resignation Process]

## Q9. counseling for stress and burnout

- category: `semantic`  ·  route: `kb_query`  ·  28.5s
- search attempts:
  1. "counseling for stress and burnout" → 0 result(s)
  2. "Employee Assistance Program (EAP) counseling services for work-related stress and burnout" → 3 result(s)
- retrieved evidence:
  - **Employee Assistance Program** (score 0.0328, bm25+dense, `01_hr_policies.txt`)
  - **Service Credit Policy** (score 0.0161, bm25, `03_products_and_support.txt`)
  - **BYOD Program** (score 0.0159, bm25, `01_hr_policies.txt`)
- answer:
  > - What is provided: confidential guidance sessions with licensed practitioners for personal or work‑related difficulties. [Employee Assistance Program]  
  > - Who can use it: all employees and their immediate family members. [Employee Assistance Program]  
  > - Cost and limits: each employee family is entitled to 6 sessions per calendar year at no cost. [Employee Assistance Program]  
  > - Language and format: sessions available in Thai or English, in person or by video call. [Employee Assistance Program]  
  > - How to book: make appointments directly with MindBridge by calling internal extension 5-4477 or through the MindBridge web portal linked from SiamPeople. [Employee Assistance Program]  
  > - Confidentiality: managers and HR are never informed who has used the service. [Employee Assistance Program]  
  > - Further assessment: MindBridge practitioners can refer employees to MediTrust Clinic where appropriate. [Employee Assistance Program]  
  > - Urgent support: the MindBridge line operates around the clock, every day of the year, for urgent situations outside office hours. [Employee Assistance Program]

## Q10. customer SLA uptime compensation

- category: `semantic`  ·  route: `kb_query`  ·  27.8s
- search attempts:
  1. "customer SLA uptime compensation" → 4 result(s)
- retrieved evidence:
  - **Service Credit Policy** (score 0.0164, dense, `03_products_and_support.txt`)
  - **Support Escalation Process** (score 0.0161, dense, `03_products_and_support.txt`)
  - **Customer Support Service Levels** (score 0.0159, dense, `03_products_and_support.txt`)
  - **Compensation Disbursement Schedule** (score 0.0156, dense, `01_hr_policies.txt`)
- answer:
  > - Uptime commitments: 99.9% per calendar month for Platinum and 99.5% for Standard, measured per product. [Service Credit Policy]  
  > - Measurement excludes the announced maintenance window on Sundays from 01:00 to 05:00 Bangkok time. [Service Credit Policy]  
  > - If measured availability falls below the commitment, customers may claim service credits as follows: 5% of that month's subscription fee when availability stays above 99.0%; 10% for availability between 97.0% and 99.0%; and 25% for availability below 97.0%. [Service Credit Policy]  
  > - Claims must be submitted through the customer portal within 30 days of the end of the affected month and should reference related CS case numbers where available. [Service Credit Policy]  
  > - Credits are applied to a future invoice (never paid in cash) and are capped at 25% of one month's fee. [Service Credit Policy]  
  > - Availability statistics per product are published monthly on the status page. [Service Credit Policy]

## Q11. everything I need for an overseas business trip

- category: `multi_chunk`  ·  route: `kb_query`  ·  31.9s
- search attempts:
  1. "everything I need for an overseas business trip" → 4 result(s)
- retrieved evidence:
  - **International Travel Approval Process** (score 0.0328, bm25+dense, `02_travel_and_expenses.txt`)
  - **International Travel Insurance** (score 0.0323, bm25+dense, `02_travel_and_expenses.txt`)
  - **Domestic Travel Policy** (score 0.0312, bm25+dense, `02_travel_and_expenses.txt`)
  - **Software Request and Licensing** (score 0.0159, bm25, `01_hr_policies.txt`)
- answer:
  > Checklist — everything required for an overseas business trip
  > 
  > - Obtain written approval from your department head at least 14 days before departure. [International Travel Approval Process]  
  > - Submit the request through the TravelHub system under "Overseas Trip Request" and include a trip purpose statement, an itinerary, and a cost estimate. [International Travel Approval Process]  
  > - If the trip is longer than 10 business days, obtain additional sign-off from the Managing Director. [International Travel Approval Process]  
  > - After approval, the Travel Desk will book economy-class airfare through the company’s contracted agency, ThaiOrbit Travel. [International Travel Approval Process]  
  > - Do not self-book flights unless the Travel Desk confirms in writing that no contracted option was available; otherwise self-booked flights are not reimbursable. [International Travel Approval Process]  
  > - Visa fees and airport transfers are arranged by the Travel Desk and charged directly to the department’s cost center. [International Travel Approval Process]  
  > - Your international travel is covered by the company group insurance, SafeJourney Plan B (Bangkok Assurance PCL), which covers emergency medical treatment up to 3,000,000 THB, emergency evacuation, and lost-luggage compensation up to 40,000 THB per trip. [International Travel Insurance]  
  > - Carry the digital insurance card from the HR mobile app (SiamPeople) under "My Benefits". [International Travel Insurance]  
  > - File any insurance claims through the SafeJourney portal within 30 days of returning, submitting original receipts and a copy of the TravelHub trip approval. [International Travel Insurance]  
  > - Note that personal side-trips extending a business trip are not covered beyond 48 hours after the last business day. [International Travel Insurance]

## Q12. how do I buy something from a new supplier

- category: `multi_chunk`  ·  route: `kb_query`  ·  26.6s
- search attempts:
  1. "how do I buy something from a new supplier" → 1 result(s)
- retrieved evidence:
  - **Vendor Onboarding and Registration** (score 0.0328, bm25+dense, `02_travel_and_expenses.txt`)
- answer:
  > - Register the new vendor in ProcureDesk before any first purchase order is issued. [Vendor Onboarding and Registration]  
  > - Obtain from the vendor: a company affidavit issued within the last 6 months, VAT registration (Por Por 20), a bank account confirmation letter, and a signed copy of the Siam Innovate supplier code of conduct. [Vendor Onboarding and Registration]  
  > - Procurement will screen new vendors within 10 business days; for contracts expected to exceed 1,000,000 THB per year this screening includes sanctions and litigation checks. [Vendor Onboarding and Registration]  
  > - If the vendor has had no transactions for 24 months they will be set to inactive and must be re-verified before new orders. [Vendor Onboarding and Registration]  
  > - Any changes to a vendor’s bank details take effect only after a call-back to a previously registered contact person and never on the basis of an emailed request alone. [Vendor Onboarding and Registration]

## Q13. P1 response time for platinum support

- category: `multi_chunk`  ·  route: `kb_query`  ·  14.0s
- search attempts:
  1. "P1 response time for platinum support" → 3 result(s)
- retrieved evidence:
  - **Customer Support Service Levels** (score 0.0328, bm25+dense, `03_products_and_support.txt`)
  - **Support Escalation Process** (score 0.0323, bm25+dense, `03_products_and_support.txt`)
  - **Service Credit Policy** (score 0.0159, dense, `03_products_and_support.txt`)
- answer:
  > P1 (production service fully unavailable) — first response within 15 minutes on Platinum. [Customer Support Service Levels]

## Q14. What is the CEO's salary?

- category: `negative`  ·  route: `kb_query`  ·  35.2s
- search attempts:
  1. "What is the CEO's salary?" → 0 result(s)
  2. "Chief Executive Officer annual base salary (CEO base pay)" → 1 result(s)
- retrieved evidence:
  - **Resignation Process** (score 0.0164, bm25, `01_hr_policies.txt`)
- answer:
  > I could not find this information in the knowledge base.

## Q15. employee home addresses and phone numbers

- category: `negative`  ·  route: `kb_query`  ·  33.1s
- search attempts:
  1. "employee home addresses and phone numbers" → 0 result(s)
  2. "Employee personal contact information: residential (home) addresses and personal phone numbers (personnel file)" → 1 result(s)
- retrieved evidence:
  - **Data Classification and Handling** (score 0.0164, bm25, `01_hr_policies.txt`)
- answer:
  > I could not find this information in the knowledge base.

## Q16. ลาบวชได้กี่วัน และต้องแจ้งล่วงหน้าอย่างไร?

- category: `thai`  ·  route: `kb_query`  ·  25.7s
- search attempts:
  1. "ordination leave duration and advance notice" → 4 result(s)
- retrieved evidence:
  - **Ordination Leave** (score 0.0325, bm25+dense, `01_hr_policies.txt`)
  - **Annual Leave** (score 0.0323, bm25+dense, `01_hr_policies.txt`)
  - **Sick Leave** (score 0.0320, bm25+dense, `01_hr_policies.txt`)
  - **Resignation Process** (score 0.0156, dense, `01_hr_policies.txt`)
- answer:
  > - จำนวนวัน: พนักงานสามารถลาบวชแบบได้รับค่าจ้างได้สูงสุด 15 วันทำงาน ตลอดระยะเวลาการทำงานกับบริษัท (ได้ครั้งเดียว) [Ordination Leave].  
  > - การแจ้งล่วงหน้า: ต้องบันทึกคำขอในระบบ LeaveDesk ใต้หมวด "Ordination" ล่วงหน้าอย่างน้อย 30 วัน พร้อมระบุวันที่พิธีและวัดที่จัดพิธี [Ordination Leave].  
  > - ข้อกำหนดเพิ่มเติม: สิทธิ์นี้ใช้ได้หลังผ่านการทดลองงาน และต้องส่งหนังสือยืนยันจากวัดผ่าน LeaveDesk ภายใน 7 วันนับจากวันที่กลับมาทำงาน [Ordination Leave].  
  > - กรณีต้องการลาบวชนานกว่านั้น สามารถจัดเป็นการขาดงานโดยไม่ได้รับค่าจ้างได้กรณีได้รับอนุมัติจากฝ่ายบุคคล สูงสุดไม่เกิน 120 วัน และวันหยุดนักขัตฤกษ์ระหว่างช่วงลาจะไม่ถูกนับเป็นส่วนหนึ่งของ 15 วันดังกล่าว [Ordination Leave].

## Q17. Hello! What can you do?

- category: `greeting`  ·  route: `direct`  ·  12.5s
- answer:
  > Hi — I can help with the Siam Innovate employee handbook. Ask me about leave, travel, expenses, benefits, IT security, HR processes, or products.
