# Retrieval Evaluation Results

- Run date: 2026-07-25 15:20
- Test set: 15 queries (src/evaluation/testset.py)
- Config: TOP_K=4, MIN_SCORE=2.0, MIN_COSINE=0.38, EMBEDDING_MODEL=text-embedding-3-small, FUSION_METHOD=rrf, RRF_K=60

## Overall

| mode     | hit_rate@k | recall@k | MRR   | FP_rate(neg) | avg_latency |
|----------|------------|----------|-------|--------------|-------------|
| keyword  | 62%        | 54%      | 0.558 | 0%           | 0.1 ms      |
| semantic | 77%        | 71%      | 0.718 | 0%           | 495.1 ms    |
| hybrid   | 85%        | 77%      | 0.808 | 0%           | 485.4 ms    |

## Per category

| category    | n | metric   | keyword | semantic | hybrid |
|-------------|---|----------|---------|----------|--------|
| lexical     | 4 | hit_rate | 100%    | 75%      | 100%   |
| lexical     | 4 | recall   | 100%    | 75%      | 100%   |
| lexical     | 4 | MRR      | 1.000   | 0.583    | 1.000  |
| semantic    | 6 | hit_rate | 17%     | 67%      | 67%    |
| semantic    | 6 | recall   | 17%     | 67%      | 67%    |
| semantic    | 6 | MRR      | 0.042   | 0.667    | 0.583  |
| multi_chunk | 3 | hit_rate | 100%    | 100%     | 100%   |
| multi_chunk | 3 | recall   | 67%     | 75%      | 67%    |
| multi_chunk | 3 | MRR      | 1.000   | 1.000    | 1.000  |
| negative    | 2 | FP_rate  | 0%      | 0%       | 0%     |

## Imperfect cases

| mode     | case                | expected                                                                                                                                       | retrieved (top-k)                                                                                                                   |
|----------|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| keyword  | sem_funeral         | Bereavement and Compassionate Absence                                                                                                          | Parental Leave                                                                                                                      |
| keyword  | sem_dress_code      | Workplace Attire Standards                                                                                                                     | []                                                                                                                                  |
| keyword  | sem_quit_job        | Resignation Process                                                                                                                            | []                                                                                                                                  |
| keyword  | sem_counseling      | Employee Assistance Program                                                                                                                    | []                                                                                                                                  |
| keyword  | sem_sla_uptime      | Service Credit Policy                                                                                                                          | []                                                                                                                                  |
| keyword  | multi_overseas_trip | International Travel Approval Process, International Travel Daily Allowance, International Travel Insurance, International Travel Visa Support | International Travel Approval Process, International Travel Insurance, Software Request and Licensing, Domestic Travel Policy       |
| keyword  | multi_new_supplier  | Purchase Requisition and Purchase Orders, Vendor Onboarding and Registration                                                                   | Vendor Onboarding and Registration                                                                                                  |
| semantic | lex_form_hr204      | Employee Referral Program                                                                                                                      | []                                                                                                                                  |
| semantic | sem_quit_job        | Resignation Process                                                                                                                            | []                                                                                                                                  |
| semantic | sem_counseling      | Employee Assistance Program                                                                                                                    | []                                                                                                                                  |
| semantic | multi_overseas_trip | International Travel Approval Process, International Travel Daily Allowance, International Travel Insurance, International Travel Visa Support | International Travel Approval Process, International Travel Insurance, International Travel Daily Allowance, Domestic Travel Policy |
| semantic | multi_new_supplier  | Purchase Requisition and Purchase Orders, Vendor Onboarding and Registration                                                                   | Vendor Onboarding and Registration                                                                                                  |
| hybrid   | sem_quit_job        | Resignation Process                                                                                                                            | []                                                                                                                                  |
| hybrid   | sem_counseling      | Employee Assistance Program                                                                                                                    | []                                                                                                                                  |
| hybrid   | multi_overseas_trip | International Travel Approval Process, International Travel Daily Allowance, International Travel Insurance, International Travel Visa Support | International Travel Approval Process, International Travel Insurance, Domestic Travel Policy, Software Request and Licensing       |
| hybrid   | multi_new_supplier  | Purchase Requisition and Purchase Orders, Vendor Onboarding and Registration                                                                   | Vendor Onboarding and Registration                                                                                                  |
