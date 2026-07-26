# Evaluation Plan: Numeric Retrieval and Answer Evaluation

## 1. วัตถุประสงค์

เอกสารนี้เป็นแผน implement สำหรับเพิ่ม **numeric evaluation** ที่รันได้จริงจาก
code ใน repo นี้ แล้วนำผลไปแสดงใน `README.md`

เป้าหมายคือทำให้สอง Evaluation Criteria ของโจทย์ที่เป็น subjective กลายเป็นสิ่งที่
ผู้ตรวจ **รันซ้ำเองได้**:

- *"Effective implementation of the RAG mechanism (the custom data retrieval tool)"*
- *"Clarity and quality of the final output (non-redundant, accurate based on provided info)"*

ข้อจำกัดที่ต้องรักษาไว้ตลอดทั้งแผน (ฉบับปรับให้สอดคล้องกับ `IMPROVEMENT_PLAN.md`):

- ไม่เพิ่ม Vector Database; default retrieval path ต้องเป็น lexical +
  deterministic + offline เสมอ — semantic fallback (ถ้าทำ) เป็น opt-in ตาม
  IMPROVEMENT_PLAN §4.6 และ**ห้ามนับใน offline eval**
- Retrieval evaluation ต้อง deterministic และรัน offline ได้ 100% (ไม่เรียก network)
- Answer evaluation ที่ต้องใช้ LLM ต้องเป็น **opt-in** เหมือน `tests/test_live_e2e.py`
- **ตัวงาน evaluation เอง**ห้ามเปลี่ยน behavior ของ pipeline (`PipelineState`,
  graph topology, agent prompt) — การเปลี่ยน behavior ทุกครั้งทำผ่าน phase ของ
  `IMPROVEMENT_PLAN.md` เท่านั้น แล้วรัน eval ซ้ำตามลำดับใน IMPROVEMENT_PLAN §7

---

## 2. Non-negotiables (หลักการที่ห้ามละเมิด)

ข้อเหล่านี้สำคัญกว่าตัวเลขที่สวย ถ้าข้อไหนขัดกัน ให้ยึดข้อนี้เสมอ:

| # | หลักการ | เหตุผล |
|---|---|---|
| N1 | ทุกตัวเลขใน README ต้องผลิตจาก command ที่อยู่ใน repo นี้จริง | ผู้ตรวจรันแล้วไม่ตรง = เสียหายกว่าไม่มี eval |
| N2 | ห้ามอ้าง config ที่ระบบนี้ไม่มี (`TOP_K`, `MIN_COSINE`, `RRF_K`, embedding model) | ระบบนี้เป็น lexical ล้วน ไม่มี parameter เหล่านี้ |
| N3 | ห้ามใช้ metric ตระกูล `@k` | retriever คืน threshold-gated set ขนาดไม่คงที่ — ไม่มี `k` ให้ใช้ |
| N4 | ห้ามแก้ golden set หลังเห็นผลลัพธ์ | ถ้าแก้เพื่อให้เลขผ่าน ตัวเลขจะไม่มีความหมายทันที |
| N5 | ต้องแยก **calibration set** ออกจาก **held-out set** และรายงานทั้งคู่ | constant ปัจจุบันถูกจูนบน 23 cases เดิม (ดู §3.2) |
| N6 | ถ้า held-out ได้ไม่ถึง 100% ให้รายงานตามจริง พร้อมยกเคสที่พลาดมาแสดง | เลข held-out ที่ต่ำกว่าแต่ซื่อสัตย์ น่าเชื่อกว่า 100% ที่จูนมา |

---

## 3. สถานะปัจจุบัน

### 3.1 สิ่งที่มีอยู่แล้ว (ไม่ต้องสร้างใหม่)

- **Golden set 23 cases** ที่ `tests/fixtures/retrieval_cases.json`
  มี 5 categories และมี `expected_titles` + `forbidden_titles` ครบทั้ง positive/negative
- **Metric logic ระดับหนึ่ง** อยู่ใน `tests/test_retrieval.py::RetrievalEvaluationTests`
  (บรรทัด ~389-429) ซึ่งคำนวณ micro precision, micro recall, exact-pass rate และ
  unknown-rejection rate อยู่แล้ว — แต่ **assert แล้วทิ้ง ไม่ได้ export ออกมาเป็นรายงาน**
- **Deterministic not-found guarantee** ที่ `src/agents/reporter.py:59-61`
  (snippets ว่าง → คืน `NOT_FOUND_SENTENCE` โดยไม่เรียก LLM)
- **Live-test gate pattern** ที่ `tests/test_live_e2e.py:8` (`RUN_LIVE_LLM_TESTS=1`)

**สรุป:** งานหลักคือ *"ยกตรรกะที่มีอยู่ออกมาเป็น module ที่รายงานผลได้"* ไม่ใช่สร้าง eval ใหม่จากศูนย์

### 3.2 หนี้ที่ต้องประกาศให้ชัด

`src/tools/retrieval.py:60-62` เขียนไว้ตรงๆ ว่า:

```python
# These values are calibrated against tests/fixtures/retrieval_cases.json.
```

แปลว่า **golden set ปัจจุบัน = tuning set** ตัวเลข 100% บนชุดนี้จึงเป็น *fit statistic*
ไม่ใช่ค่าประมาณความสามารถในการ generalize เอกสารนี้แก้ปัญหานั้นด้วย held-out set ใน §5.2

หนี้ข้อนี้ครอบคลุม **constant ใหม่ทุกตัวจาก IMPROVEMENT_PLAN** ด้วย
(stemming rules, `K_TF`, threshold ที่ retune) — จูนได้กับ calibration set
เท่านั้น และต้อง freeze ก่อนสร้าง held-out set (IMPROVEMENT_PLAN N8, §7)

### 3.3 ขยะที่ต้องเก็บกวาด

`src/evaluation/` และ `src/retrievers/` มีแต่ `__pycache__` ที่ติดมาจาก repo อื่น
(`run_eval`, `run_answer_eval`, `testset`, `hybrid`, `dense`, `factory`) ไม่มีไฟล์ `.py` เลย
ต้องลบทิ้งก่อนเริ่ม (Phase 0) เพื่อไม่ให้ปนกับ module ใหม่

---

## 4. Metric Design

### 4.1 ทำไมต้อง set-based ไม่ใช่ `@k`

`README.md` ข้อ 13 ของ Retrieval design ระบุเองว่า *"there is no fixed TOP_K"* —
`search()` คืนทุก section ที่ผ่าน relevance gate ดังนั้นขนาด output แปรผันตาม query
metric ที่ถูกต้องคือ metric ที่วัด **คุณภาพของ set** ไม่ใช่ **คุณภาพของ ranking ที่ตัดที่ k**

### 4.2 Retrieval metrics ที่จะรายงาน

| Metric | นิยาม | ทำไมต้องมี |
|---|---|---|
| `exact_match` | % ของ query ที่ได้ title list ตรงเป๊ะรวมลำดับ | เกณฑ์เข้มสุด ตรงกับ assertion ปัจจุบัน |
| `set_match` | % ของ query ที่ได้ title set ตรง (ไม่สนลำดับ) | แยก "ผิดเนื้อหา" ออกจาก "ผิดแค่ลำดับ" |
| `precision_macro` | เฉลี่ย precision ต่อ query | ไม่ให้ query ที่มี expected เยอะครอบงำค่าเฉลี่ย |
| `recall_macro` | เฉลี่ย recall ต่อ query | เหตุผลเดียวกัน |
| `f1_macro` | harmonic mean ของสองตัวบน | ตัวเลขสรุปตัวเดียว |
| `precision_micro` / `recall_micro` | รวมทุก query แล้วหาร | เทียบต่อเนื่องกับเลขที่ test ปัจจุบันคำนวณ |
| `mrr` | 1/rank ของ relevant ตัวแรก | output มีลำดับ metric นี้จึงยังนิยามได้ |
| `fp_rate_negative` | % ของ negative query ที่คืน section ใดๆ | วัด guardrail ตรงๆ |
| `over_retrieval` / `under_retrieval` | % query ที่คืนเกิน / ขาด | ใช้วินิจฉัยว่า threshold เอียงไปทางไหน |
| `latency_ms_p50` / `p95` | เวลาต่อ search | local scan — เป็นตัวเลขจริงที่วัดได้ ไม่ต้องมี network |

> **หมายเหตุ:** negative queries (`expected_titles == []`) ต้องถูกกันออกจากการเฉลี่ย
> precision/recall/MRR และไปนับใน `fp_rate_negative` อย่างเดียว มิฉะนั้น recall จะหาร 0

### 4.3 Answer-level metrics (deterministic ล้วน)

| Axis | นิยาม | Threshold |
|---|---|---|
| `citation_validity` | ทุก `[Section Title]` ที่ปรากฏในคำตอบ ต้องตรงกับ title ของ snippet ที่ส่งเข้า generator จริง | 100% |
| `not_found_discipline` | negative query ต้องได้ `NOT_FOUND_SENTENCE` แบบ byte-exact | 100% |
| `evidence_provenance` | ทุก snippet ที่ handoff ต้องเป็นสมาชิกของ `load_knowledge_base()` แบบตรงตัว | 100% |
| `no_llm_on_empty` | เมื่อ snippets ว่าง ต้องไม่มีการเรียก LLM เลย | 100% |
| `baseline_coverage` | snippets ที่ handoff ต้องเป็น superset ของ `search(query)` — *มีผลเฉพาะเมื่อ IMPROVEMENT_PLAN IP-6 (multi-intent decomposition) ถูก implement; ถ้าไม่ทำ ให้ตัด axis นี้ออกจากรายงานตามจริง* | 100% |
| `required_fact_coverage` | % ของ `required_facts` ต่อเคส (จาก `answer_cases.json`) ที่ปรากฏในคำตอบแบบ normalized substring match | 100% — รายงานตามจริงถ้าไม่ถึง |
| `unsupported_number_rate` | ตัวเลข/จำนวนเงินทุกตัวในคำตอบ (ดึงด้วย regex) ต้องปรากฏใน snippets ที่ handoff — ตัวที่ไม่ปรากฏนับเป็น unsupported | 0% |
| `forbidden_fact_violations` | จำนวนครั้งที่ `forbidden_facts` ของเคสโผล่ในคำตอบ (จับ cross-contamination ระหว่าง section) | 0 |

Axis ทั้งหมดนี้ **ไม่ต้องใช้ LLM judge** และครอบคลุมการันตีที่สำคัญที่สุดของระบบ
(คำตอบอ้างอิงหลักฐานจริง / ครบตาม evidence / ไม่แต่งเติมตัวเลข /
ไม่รู้ก็บอกว่าไม่รู้ / recall ไม่แย่กว่า deterministic baseline)

ข้อควรระวังการรายงานสองข้อ:

- ตัว metric คำนวณแบบ deterministic แต่ LLM output เป็น probabilistic —
  รายงานต้องระบุ model, prompt version (commit hash), และจำนวน runs ต่อเคส
  เสมอ **ห้ามเรียกผลรวมว่า "deterministic answer quality"**
  (รายละเอียดใน IMPROVEMENT_PLAN §6.1)
- หลัง IMPROVEMENT_PLAN IP-3c (runtime citation validator):
  `citation_validity` ถูก enforce ณ runtime แล้ว — ตัวเลขควรเป็น 100%
  by construction แต่ยังรายงานไว้เป็น regression tripwire

### 4.4 สิ่งที่จงใจไม่ทำใน Phase นี้

- **LLM-as-judge (faithfulness / answer relevance):** เสียเงิน, ไม่ deterministic (ขัดกับ
  จุดยืนของ repo), และที่ n ≈ 25 คำถาม ตัวเลขอย่าง 0.98 กับ 1.00 แยกกันไม่ได้ในทางสถิติ
  → ยกไปเป็น optional Phase 5 — หมายเหตุ: `required_fact_coverage` /
  `unsupported_number_rate` ใน §4.3 **ไม่ใช่** judge (เป็น string/number
  matching ล้วน) จึงอยู่ในขอบเขต Phase นี้ได้
- **Mode comparison (keyword/semantic/hybrid):** ระบบมี mode เดียว → ใช้ ablation แทน (§6)

---

## 5. Dataset Design

### 5.1 Calibration set (มีอยู่แล้ว — เปลี่ยนแค่ป้าย)

`tests/fixtures/retrieval_cases.json` — 23 cases คงเดิมทุกตัวอักษร
เปลี่ยนเฉพาะวิธีเรียกใน README/รายงาน: จาก *"golden set"* เป็น
**"calibration set (constants were tuned on this set)"**

### 5.2 Held-out set (สร้างใหม่ — หัวใจของแผนนี้)

ไฟล์ใหม่: `tests/fixtures/retrieval_heldout.json` — schema เดียวกับของเดิมเป๊ะ
เพื่อให้ loader ตัวเดียวใช้ได้ทั้งคู่

**สัดส่วนที่เสนอ (14 cases):**

| Category | n | เจตนา |
|---|---|---|
| `unseen_paraphrase` | 4 | ใช้คำที่ **ไม่มี** ใน `PHRASE_ALIASES` / `TOKEN_ALIASES` เช่น "sick day", "travel abroad", "how fast do you reply", "trip expenses" |
| `unseen_inflection` | 2 | คำผันรูปที่ไม่อยู่ใน alias table เพื่อวัดผล stemmer ของ IMPROVEMENT_PLAN §4.1 ตรง ๆ เช่น "reimbursing flights", "escalated tickets" |
| `cross_domain_precision` | 3 | คำที่กำกวมข้ามหมวด เช่น "card", "coverage", "approval" |
| `multi_section_recall` | 2 | คำถามที่ต้องได้หลาย section พร้อมกัน |
| `negative` | 3 | คำถามที่ KB ตอบไม่ได้ และไม่ซ้ำกับ 3 ตัวเดิม |

**กติกาที่ต้องบังคับตัวเอง:**

1. สร้าง held-out **หลังจาก** retrieval changes ของ IMPROVEMENT_PLAN IP-3
   ถูก freeze แล้วเท่านั้น (ลำดับใน IMPROVEMENT_PLAN §7) — กัน contamination
   จากการที่ผู้เขียน rule เห็นเคสก่อนแล้วจูนโดยไม่รู้ตัว
2. เขียน `expected_titles` โดยดูจาก `knowledge_base.txt` เท่านั้น — **ห้ามรันโค้ดก่อน**
3. commit ไฟล์ held-out เป็น commit แยก **ก่อน** commit ที่รัน eval
4. เห็นผลแล้วห้ามแก้ ถ้าเจอว่าเคสไหน label ผิดจริง ให้แก้พร้อมเขียน note กำกับว่าแก้เพราะอะไร

**ผลที่คาดหวังอย่างซื่อสัตย์:** held-out น่าจะได้ต่ำกว่า calibration เพราะ README
limitations ยอมรับเองว่า alias vocabulary เป็น curated และ *"unseen synonyms and
conceptual similarity are not understood"* — **นั่นคือผลลัพธ์ที่ถูกต้อง ไม่ใช่ความล้มเหลว**
และเป็นหลักฐานสนับสนุนหัวข้อ Limitations ที่เขียนไว้แล้ว หมายเหตุ: stemmer จาก
IMPROVEMENT_PLAN ควรช่วยหมวด `unseen_inflection` แต่**ไม่ควรช่วย** unseen
synonyms — ถ้าเลขสองหมวดนี้แยกทิศกันตามคาด นั่นคือหลักฐานว่าแต่ละชั้นทำงาน
ตามที่ออกแบบ

---

## 6. Ablation Design (ใช้แทน mode comparison)

แทนที่จะเทียบกับ retriever ที่ไม่มีอยู่ ให้เทียบ **pipeline ตัวเองแบบถอดทีละชั้น**
ตารางนี้ตอบ criteria *"Effective implementation of the RAG mechanism"* ตรงกว่า
เพราะแสดงว่า design decision แต่ละข้อซื้ออะไรมา — และรัน offline ฟรี

| Variant | เปิดอะไรบ้าง |
|---|---|
| `V0_raw_overlap` | token overlap ดิบ คืนทุก section ที่มี term ตรงอย่างน้อย 1 |
| `V1_+query_filters` | V0 + ตัด stopwords / domain-generic / framing terms |
| `V2_+aliases` | V1 + `PHRASE_ALIASES` และ `TOKEN_ALIASES` |
| `V3_+idf_title_weight` | V2 + smoothed IDF และ title weight 1.5 |
| `V4_+gate_sibling` | V3 + candidate rule + relative cutoff 0.60 + two-term sibling expansion |
| `V5_+stemming` | V4 + light inflectional stemmer (IMPROVEMENT_PLAN §4.1) |
| `V6_current` | V5 + TF saturation (IMPROVEMENT_PLAN §4.2) — คือระบบจริงหลัง IP-3 |

> **Decision gate:** ถ้า `V6` ไม่ดีกว่า `V5` ในทุก metric บน calibration
> ให้ถอด TF ออกตาม IMPROVEMENT_PLAN §4.2 — ในกรณีนั้น `V5` กลายเป็น
> `V5_current` และตารางรายงานมี 6 แถว พร้อมบันทึกเหตุผลใน
> `docs/DESIGN_NOTES.md`

### 6.1 วิธี implement แบบ minimal-invasive

เพิ่ม frozen dataclass ใน `src/tools/retrieval.py`:

```python
@dataclass(frozen=True)
class RetrievalSettings:
    use_query_filters: bool = True
    use_aliases: bool = True
    use_idf: bool = True
    title_weight: float = TITLE_MATCH_WEIGHT
    use_relevance_gate: bool = True
    use_sibling_expansion: bool = True
    use_stemming: bool = True          # IMPROVEMENT_PLAN §4.1
    use_tf_saturation: bool = True     # IMPROVEMENT_PLAN §4.2
    k_tf: float = K_TF

DEFAULT_SETTINGS = RetrievalSettings()
```

แล้วเปลี่ยน signature เป็น
`search(query, path=None, *, settings: RetrievalSettings = DEFAULT_SETTINGS)`

**เงื่อนไขบังคับ:** ต้องมี test ยืนยันว่า `search(q)` กับ
`search(q, settings=DEFAULT_SETTINGS)` ให้ผลเหมือนกันเป๊ะทุก case ใน fixture ทั้งสองชุด
— ablation ต้องไม่ทำให้ behavior ปัจจุบันเปลี่ยนแม้แต่น้อย

---

## 7. โครงสร้างไฟล์ที่จะเพิ่ม

```text
src/evaluation/
├── __init__.py
├── dataset.py              # loader เดียวใช้ร่วมกันทั้ง tests และ eval
├── metrics.py              # pure functions ทั้งหมด ไม่มี I/O
├── ablation.py             # นิยาม 5 variants
├── run_retrieval_eval.py   # entrypoint offline → evaluation_results.md
└── run_answer_eval.py      # entrypoint opt-in live → answer_eval_results.md

tests/fixtures/
├── retrieval_cases.json    # เดิม (calibration)
├── retrieval_heldout.json  # ใหม่ (held-out)
└── answer_cases.json       # ใหม่ (answer quality: required/forbidden facts)

tests/
└── test_evaluation.py      # ทดสอบ metric functions ด้วย input ที่รู้คำตอบ

evaluation_results.md       # generated — commit ไว้ให้ผู้ตรวจอ่านได้ทันที
answer_eval_results.md      # generated — commit เมื่อรัน live แล้ว
```

### 7.1 Contract ของแต่ละ module

**`dataset.py`**

```python
def load_cases(name: str) -> list[EvalCase]      # "calibration" | "heldout"
def load_all() -> dict[str, list[EvalCase]]
```

`tests/test_retrieval.py` ต้องเปลี่ยนไปเรียก loader ตัวนี้ แทน `_load_retrieval_cases()`
ที่เขียนไว้ในไฟล์ test เอง — ให้มี **single source of truth** ชุดเดียว

**`metrics.py`** — pure ทั้งหมด รับ `(retrieved_titles, expected_titles)` คืน dataclass
ไม่อ่านไฟล์ ไม่ print เพื่อให้ `tests/test_evaluation.py` ทดสอบด้วยค่าที่คำนวณมือได้

**`run_retrieval_eval.py`** — รันทุก variant × ทุก dataset แล้วเขียน markdown
ต้อง `exit(1)` เมื่อ variant `current` (V6 หรือ V5 ตาม decision gate ใน §6)
บน calibration ต่ำกว่า threshold ที่ประกาศไว้ เพื่อให้ใช้เป็น CI gate ได้

**`run_answer_eval.py`** — gate ด้วย `RUN_LIVE_LLM_TESTS=1` + ต้องมี `OPENAI_API_KEY`
รัน graph จริงผ่านทุก case (retrieval fixtures + `answer_cases.json`) แล้ววัด
axis ทั้งหมดใน §4.3 รวม deep metrics ตาม IMPROVEMENT_PLAN §6.1
(ประมาณ 40-50 LLM calls ต่อ run — retriever + reporter ต่อ query,
negative queries ไม่เรียก reporter) และพิมพ์ model, prompt version,
จำนวน runs ไว้หัวรายงานเสมอ

---

## 8. Implementation Phases

> **ลำดับจริงถูก interleave กับ IMPROVEMENT_PLAN §7:** Phase 0-1 ทำก่อน
> (= IP-0, IP-1) จากนั้นเป็นงาน improvement IP-2/IP-3 แล้วจึง Phase 3
> (= IP-4), Phase 2 (= IP-5 — **ต้องรอ IP-3 freeze ก่อน**), Phase 4 (= IP-7)
> และ Phase 5 (= IP-8) — เลข Phase ในเอกสารนี้คงเดิมเพื่อไม่ให้ reference
> ภายในพัง แต่ลำดับเวลาให้ยึดตาราง IMPROVEMENT_PLAN §7

### Phase 0 — Cleanup (15 นาที)

- ลบ `src/evaluation/__pycache__/` และ `src/retrievers/__pycache__/` ทั้งหมด
- ตรวจว่า `.gitignore` ครอบคลุม `__pycache__/` แล้วจริง

**Acceptance:** `find src -name "*.pyc"` ไม่คืนอะไร และ `git status` สะอาด

### Phase 1 — Dataset + Metrics core (2-3 ชม.)

- สร้าง `dataset.py`, `metrics.py`
- ย้าย `_load_retrieval_cases()` ออกจาก `tests/test_retrieval.py` มาใช้ `dataset.py`
- สร้าง `tests/test_evaluation.py` — ทดสอบ metric ด้วยเคสที่คำนวณมือได้
  (เช่น retrieved 2 ตัว ถูก 1 → precision 0.5, recall คิดจาก expected จริง,
  negative query ที่คืน 1 section → `fp_rate = 1.0`)

**Acceptance:** unittest ทั้งชุดผ่านเหมือนเดิม + metric functions มี test ครอบคลุม
รวมถึง edge case (expected ว่าง, retrieved ว่าง, ไม่มี intersection)

### Phase 2 — Held-out set (1-2 ชม.)

- ทำได้เฉพาะหลัง IMPROVEMENT_PLAN IP-3 (stemming/TF) ถูก freeze แล้ว (§5.2 ข้อ 1)
- เขียน `tests/fixtures/retrieval_heldout.json` 14 cases ตาม §5.2
- **commit ก่อนรัน eval**

**Acceptance:** ไฟล์ผ่าน schema validation, id ไม่ซ้ำ, มี negative ≥ 3,
และ commit hash ของไฟล์นี้เก่ากว่า commit ที่มีตัวเลขผลลัพธ์

### Phase 3 — Ablation + retrieval eval runner (3-4 ชม.)

- เพิ่ม `RetrievalSettings` ตาม §6.1 + test ยืนยัน default ไม่เปลี่ยน behavior
- สร้าง `ablation.py`, `run_retrieval_eval.py`
- รันแล้ว generate `evaluation_results.md`

**Acceptance:**
`python -m src.evaluation.run_retrieval_eval` รันจบโดยไม่ต้องมี `OPENAI_API_KEY`
และไม่มี network call, ผลลัพธ์ reproducible (รันสองครั้งได้เลขเดียวกันยกเว้น latency)

### Phase 4 — Answer eval runner (2-3 ชม.)

- เขียน `tests/fixtures/answer_cases.json` จาก `knowledge_base.txt` เท่านั้น
  และ **commit ก่อนรัน eval ครั้งแรก** (กติกาเดียวกับ held-out)
- สร้าง `run_answer_eval.py` ตาม §7.1
- รันจริงหนึ่งครั้ง generate `answer_eval_results.md`

**Acceptance:** guardrail axes (`citation_validity`, `not_found_discipline`,
`evidence_provenance`, `no_llm_on_empty`, `baseline_coverage` ถ้ามี IP-6)
และ `unsupported_number_rate` ต้องได้ตาม threshold เต็ม (ถ้าไม่ถึง = เจอ bug
จริง ต้องแก้ก่อน ไม่ใช่ลด threshold) ส่วน `required_fact_coverage` รายงาน
ตามจริง — ต่ำกว่า 100% ไม่ใช่ blocker แต่ต้องแนบเคสที่พลาดมาแสดง (N6)

### Phase 5 — README integration (1-2 ชม.)

ดู §9

### Phase 6 — LLM-as-judge (optional, ทำต่อเมื่อเวลาเหลือ)

ถ้าทำ ต้องเขียน limitation กำกับทุกครั้ง: judge ตัวเดียวไม่มี ensemble, n เล็ก,
และ axis ที่ judge ให้คะแนนสืบทอดความเข้มงวดของ judge เอง

**รวมโดยประมาณ: 9-14 ชม. สำหรับ Phase 0-5** (เฉพาะงาน evaluation;
ตัวเลขรวมทั้ง improvement + evaluation อยู่ที่ IMPROVEMENT_PLAN §7: 20-28 ชม.)

---

## 9. README Integration

### 9.1 ตำแหน่ง

แทรก section ใหม่ `## Evaluation` **ระหว่าง** `## Tests` (บรรทัด ~336) และ
`## Example results` (บรรทัด ~410) — คือหลังจากผู้อ่านรู้แล้วว่ามี test อะไรบ้าง
แต่ก่อนเห็น screenshot

### 9.2 สิ่งที่ต้องแก้ด้วยพร้อมกัน

- **`## Retrieval design` บรรทัด 156-159:** ประโยค *"achieves 100% exact-case pass,
  section precision, section recall, and unknown-query rejection on that checked
  dataset"* ต้องเพิ่มคำว่า calibration ให้ชัด และลิงก์ไปยัง held-out result
- **`## Project structure`:** เพิ่ม `src/evaluation/` และ fixture ใหม่
- **`## Limitations`:** เพิ่มบรรทัดที่อ้างตัวเลข held-out จริง แทนคำบรรยายลอยๆ
- **`## Tests`:** เพิ่ม command ของ eval runner

### 9.3 Draft ภาษาอังกฤษสำหรับ section ใหม่

> ปรับเลขจริงหลังรัน Phase 3-4 — ตัวเลขข้างล่างเป็น placeholder ที่ต้องแทนที่ทั้งหมด

````markdown
## Evaluation

Retrieval and answer quality are measured by code in this repository, not by
hand-checked examples. Both runners are reproducible:

```bash
python -m src.evaluation.run_retrieval_eval          # offline, no API key
RUN_LIVE_LLM_TESTS=1 python -m src.evaluation.run_answer_eval
```

### Datasets

Two labeled sets, both in `tests/fixtures/`:

| Set | n | Purpose |
|---|---|---|
| calibration | 23+ | The set the scoring constants were tuned against (includes the added `morphology` cases from IMPROVEMENT_PLAN §4.3). Numbers here are a fit statistic, not a generalization estimate. |
| held-out | 14 | Written from `knowledge_base.txt` alone after the retrieval implementation was frozen, and committed before the evaluator was ever run against it. Never edited to flatter a result. |

Because the retriever returns a threshold-gated set rather than a fixed-size
ranking (there is no `TOP_K`), retrieval is scored with set-based metrics.
`@k` metrics are deliberately not reported — the system has no `k`.

### Retrieval results

Negative queries are excluded from precision/recall/MRR and scored only by
false-positive rate.

| set | exact_match | precision_macro | recall_macro | F1 | MRR | FP_rate (neg) | p50 latency |
|---|---|---|---|---|---|---|---|
| calibration (23+) | …% | …% | …% | … | … | …% | … ms |
| held-out (14) | …% | …% | …% | … | … | …% | … ms |

### What each design decision buys

The pipeline is scored with each layer removed, on the held-out set:

| variant | exact_match | precision_macro | recall_macro | FP_rate (neg) |
|---|---|---|---|---|
| V0 raw token overlap | …% | …% | …% | …% |
| V1 + query-term filtering | …% | …% | …% | …% |
| V2 + phrase/token aliases | …% | …% | …% | …% |
| V3 + IDF and title weighting | …% | …% | …% | …% |
| V4 + relevance gate and sibling expansion | …% | …% | …% | …% |
| V5 + light inflectional stemming | …% | …% | …% | …% |
| V6 current (+ TF saturation) | …% | …% | …% | …% |

### Answer-level results

All axes are scored by deterministic matching — no LLM judge, no reference
answers. The generator output itself is probabilistic: results below were
produced with model `…`, prompt version `…` (commit), and `…` run(s) per case.

| axis | method | result | threshold |
|---|---|---|---|
| citation validity (runtime-enforced) | deterministic | …% | 100% |
| not-found discipline | deterministic | …% | 100% |
| evidence provenance | deterministic | …% | 100% |
| no LLM call on empty retrieval | deterministic | …% | 100% |
| baseline coverage (only if IP-6 shipped) | deterministic | …% | 100% |
| required-fact coverage | deterministic matching | …% | 100% |
| unsupported-number rate | deterministic matching | …% | 0% |

Full per-query tables, including every imperfect case, are in
[evaluation_results.md](evaluation_results.md) and
[answer_eval_results.md](answer_eval_results.md).

**Limitations:** both sets are small (n ≈ 23-27 and n = 14) over a 10-section corpus,
so a single case moves a percentage by several points. The held-out set is
written by the same author as the knowledge base. No LLM-as-judge axis is
reported — semantic faithfulness beyond citation validity is not measured.
````

---

## 10. Risks และสิ่งที่ห้ามทำ

| Risk | ผลถ้าเกิด | การป้องกัน |
|---|---|---|
| แปะตัวเลขจาก repo อื่น (3 modes, `TOP_K=4`, `MIN_COSINE`, RRF, embeddings) | ผู้ตรวจรัน command แล้วไม่มี module → ตั้งคำถามกับตัวเลข *ทุกตัว* ใน README | N1, N2 — ทุกเลขต้องมาจาก runner ใน repo นี้ |
| รายงาน 100% บน calibration โดยไม่บอกว่าเป็น tuning set | ถูกจับได้ว่า train-on-test | N5 — บังคับรายงานคู่กับ held-out เสมอ |
| แก้ held-out หลังเห็นผล | held-out หมดความหมาย | N4 — commit แยกก่อนรัน |
| ablation ทำให้ behavior ปัจจุบันเปลี่ยน | pipeline พังเพื่อ eval | test บังคับ default-equivalence (§6.1) |
| semantic retriever หลุดเข้า default path | offline eval ต้องมี network = ผู้ตรวจรันไม่ผ่าน, จุดยืน deterministic พัง | โจทย์อนุญาต semantic แต่ทำได้เฉพาะ opt-in fallback ตาม IMPROVEMENT_PLAN §4.6 (N7); default path และ offline eval ห้ามแตะ |
| answer eval เผาเงิน/ flaky ใน CI | test ไม่เสถียร | opt-in gate เดียวกับ `test_live_e2e.py` |

---

## 11. Verification

```bash
# Gate 1 — offline deterministic (ต้องผ่านเสมอ ไม่ต้องมี API key)
./.venv-clean/bin/python -m unittest discover -v
./.venv-clean/bin/python -m src.evaluation.run_retrieval_eval

# Gate 2 — live provider (opt-in, มีค่าใช้จ่าย)
RUN_LIVE_LLM_TESTS=1 ./.venv-clean/bin/python -m unittest tests.test_live_e2e -v
RUN_LIVE_LLM_TESTS=1 ./.venv-clean/bin/python -m src.evaluation.run_answer_eval
```

**Definition of done:**

1. ทุกตัวเลขใน README ผลิตซ้ำได้จาก command ข้างบน
2. `evaluation_results.md` และ `answer_eval_results.md` ถูก commit และตรงกับ README
3. README ระบุชัดว่าชุดไหนคือ calibration ชุดไหนคือ held-out
4. ไม่มีคำว่า `TOP_K`, `MIN_COSINE`, `RRF`, `hit_rate@k`, `recall@k`, หรือชื่อ embedding
   model ปรากฏใน README เว้นแต่ (ก) ในหัวข้อ Limitations ที่อธิบายว่า *ไม่ได้* ใช้ หรือ
   (ข) เป็นชื่อ parameter ที่มีอยู่จริงในโค้ด ณ commit นั้น เช่น
   `SEMANTIC_MIN_SIMILARITY` หาก IMPROVEMENT_PLAN IP-9 ถูก implement จริง
5. Offline gate ยังรันได้โดยไม่ต้องมี `OPENAI_API_KEY`
