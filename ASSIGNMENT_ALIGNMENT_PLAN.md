# แผนปรับโปรเจกต์ให้ตรงกับ AI Engineer Programming Test

## 1. เป้าหมาย

ปรับโปรเจกต์ให้ตอบโจทย์ Programming Test โดยตรงและตรวจสอบได้ง่าย โดยคงไว้เฉพาะระบบ Agentic RAG ขั้นพื้นฐานที่ประกอบด้วย:

1. **Data Retriever Agent** ใช้ custom retrieval tool ค้นข้อมูลจาก `knowledge_base.txt`
2. **Report Generator Agent** รับ raw snippets จาก Data Retriever และสังเคราะห์คำตอบสุดท้าย
3. **LangGraph sequential orchestration** ส่งข้อมูลตามลำดับจาก Retriever ไป Generator

ผลลัพธ์สุดท้ายต้องเป็นโปรเจกต์ Python ขนาดเล็ก อ่านง่าย รันได้จาก CLI และมีตัวอย่างผลลัพธ์พร้อม screenshots ตามที่โจทย์กำหนด

---

## 2. ขอบเขตของงาน

### อยู่ในขอบเขต

- LangGraph เป็น orchestration framework
- Agent จำนวน 2 ตัวเท่านั้น
- Knowledge base เป็น local text file ชื่อ `knowledge_base.txt`
- Custom keyword search tool
- การส่ง raw snippets ระหว่าง agent ผ่าน LangGraph state
- Grounded answer ที่ใช้เฉพาะข้อมูลจาก snippets
- คำตอบกรณีค้นข้อมูลไม่พบ
- CLI สำหรับรับคำถามและแสดงผล
- Unit tests เฉพาะพฤติกรรมหลัก
- README, setup instructions และ screenshots สำหรับส่งงาน

### อยู่นอกขอบเขต

ส่วนต่อไปนี้จะไม่นำมาใช้ใน submission flow เพราะเกินความต้องการของโจทย์:

- Router Agent
- Direct Responder
- Query Rewriter Agent
- Retry loop
- Semantic retrieval
- Hybrid retrieval
- OpenAI embeddings
- Reciprocal Rank Fusion
- Embedding cache
- Retrieval mode selector
- Streamlit UI
- LLM-as-judge
- Comparative retrieval benchmark
- Production monitoring และ deployment infrastructure

ก่อนถอดโค้ดส่วนเกินออกจาก submission branch ควรเก็บสถานะปัจจุบันไว้ใน Git branch หรือ tag เพื่อไม่ให้งาน advanced สูญหาย

---

## 3. Target Architecture

```text
User Query
    |
    v
Data Retriever Agent
    |
    | calls search_knowledge_base tool
    v
Raw Relevant Snippets
    |
    v
Report Generator Agent
    |
    v
Final Answer
```

LangGraph flow:

```text
START -> data_retriever -> report_generator -> END
```

State contract:

```python
class PipelineState(TypedDict):
    query: str
    snippets: list[str]
    report: str
```

ไม่มี conditional route, retry state หรือ retrieval metadata เพิ่มเติม

---

## 4. โครงสร้างไฟล์เป้าหมาย

```text
.
├── knowledge_base.txt
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── screenshots/
│   ├── 01_international_travel.png
│   ├── 02_remote_work.png
│   └── 03_not_found.png
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── graph.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── retriever.py
│   │   └── reporter.py
│   └── tools/
│       ├── __init__.py
│       └── retrieval.py
└── tests/
    ├── __init__.py
    ├── test_retrieval.py
    └── test_graph.py
```

---

## 5. Implementation Plan

### Phase 1: ทำ Knowledge Base ให้ตรงโจทย์

1. สร้างไฟล์ `knowledge_base.txt` ที่ project root
2. เลือกข้อมูลตัวอย่างประมาณ 6–10 sections จากข้อมูลเดิม โดยให้ครอบคลุมอย่างน้อย:
   - International travel
   - Remote work
   - Annual leave
   - Expense reimbursement
   - Product หรือ support information อย่างน้อยหนึ่งหัวข้อ
3. มีคำถามอย่างน้อยหนึ่งเรื่องที่ไม่มีข้อมูลใน knowledge base เพื่อทดสอบ not-found behavior
4. ใช้รูปแบบ section ที่อ่านและแบ่ง chunk ได้ง่าย เช่น:

```text
--- International Travel Policy ---
Employees must obtain manager approval before booking international travel.

--- Remote Work Policy ---
Employees may work remotely with manager approval.
```

5. เปลี่ยน `KB_PATH` ให้มีค่าเริ่มต้นเป็น `knowledge_base.txt`
6. ไม่ใช้ directory ingestion ใน submission version

Acceptance criteria:

- โปรแกรมอ่านข้อมูลจาก `knowledge_base.txt` จริง
- แบ่งข้อมูลออกเป็น chunks ได้ครบ
- ไม่มี dependency ต่อไฟล์ใน `data/`

### Phase 2: ลด Retrieval ให้เหลือ Custom Keyword Tool

1. ปรับ `src/tools/retrieval.py` ให้รับผิดชอบ:
   - อ่าน `knowledge_base.txt`
   - แบ่งเอกสารเป็น chunks
   - normalize query และ chunk เป็น lowercase
   - tokenize ด้วยกติกาที่ deterministic เช่น `re.findall(r"[a-z0-9]+", text.lower())`
   - ตัด English stopwords และ token ที่ไม่มีความหมาย เช่น `what`, `is`, `the`, `a`, `an`, `of`, `on`, `can`, `i`, `s`
   - ตัด domain-generic terms ที่ปรากฏข้ามหลาย sections และไม่ช่วยระบุหัวข้อ เช่น `policy`, `company`, `employee`, `employees`, `information`, `details`
   - ใช้ unique discriminative keywords ที่เหลือจาก query ในการให้คะแนน เพื่อลดผลกระทบจากคำที่ปรากฏซ้ำ
   - ให้คะแนนจากจำนวน discriminative query keywords ที่พบใน section title หรือเนื้อหา
   - คืนเฉพาะ chunks ที่มี discriminative keyword match อย่างน้อยหนึ่งคำ
   - เรียงจากคะแนนมากไปน้อย และใช้ลำดับเดิมในเอกสารเป็น tie-breaker เพื่อให้ผลลัพธ์ deterministic
2. ประกาศฟังก์ชันเป็น LangChain tool:

```python
@tool
def search_knowledge_base(query: str) -> list[str]:
    ...
```

3. Tool ต้องคืน raw snippets โดยไม่สรุป ไม่ตอบคำถาม และไม่เติมข้อมูลจาก LLM
4. คืนทุก chunk ที่ผ่าน relevance rule โดยไม่ใช้ fixed `TOP_K` เนื่องจาก knowledge base มีขนาดเล็ก และโจทย์ระบุให้ค้นหา all relevant snippets
5. ถ้า query ไม่มี discriminative keyword หลังตัด stopwords และ domain-generic terms หรือไม่มี discriminative keyword ตรงกับ chunk ใด ให้คืน empty list
6. ถอด dependency และ runtime path ของ:
   - BM25
   - embeddings
   - semantic search
   - hybrid search
   - retriever factory

Acceptance criteria:

- Tool ค้น international travel แล้วคืน section ที่เกี่ยวข้อง
- Query ที่มีคำทั่วไปอย่าง `policy` ต้องไม่ดึง unrelated policy sections เข้ามา
- Tool คืนทุก section ที่เกี่ยวข้องโดยไม่ถูกตัดด้วย fixed result limit
- Tool คืน empty list สำหรับคำถามที่ไม่มีข้อมูล เช่น `What is the CEO's salary?`
- Common words, stopwords หรือ domain-generic terms เพียงอย่างเดียวต้องไม่ทำให้ chunk ถูกจัดว่า relevant
- ผลลัพธ์เดียวกันต้องมีลำดับเหมือนกันทุกครั้ง
- Retrieval tests รันได้โดยไม่ต้องใช้ API key

Limitations ที่ต้องบันทึกใน README:

- Exact keyword matching ไม่เข้าใจ synonym หรือ semantic similarity
- คำถามต้องมี discriminative keyword ที่ปรากฏใน knowledge base จึงจะค้นพบข้อมูล
- คำถามกว้างที่มีเฉพาะคำทั่วไป เช่น `What policies are available?` อาจคืน not-found และผู้ใช้ต้องระบุหัวข้อให้ชัดขึ้น
- แนวทางนี้เลือกโดยตั้งใจเพื่อให้ implementation เรียบง่าย โปร่งใส และตรงกับ Programming Test

### Phase 3: ปรับ Data Retriever Agent

1. คงเฉพาะบทบาทค้นข้อมูล
2. Prompt ต้องระบุชัดเจนว่า:
   - ต้องเรียก `search_knowledge_base` หนึ่งครั้ง
   - ห้ามตอบคำถามเอง
   - ห้ามสรุปหรือแก้ไข snippets
3. Bind custom tool ด้วย `tool_choice="required"`
4. Execute tool จริงจาก tool call arguments เช่น:

```python
tool_call = response.tool_calls[0]
snippets = search_knowledge_base.invoke(tool_call["args"])
```

5. เขียนผลลัพธ์ลง `state["snippets"]`
6. ไม่ทำ query translation, query rewrite หรือ retry

Acceptance criteria:

- Data Retriever เรียก custom tool จริง
- Data Retriever ไม่สร้าง final answer
- Output ที่ส่งต่อเป็น `list[str]`

### Phase 4: ปรับ Report Generator Agent

1. รับเฉพาะ `query` และ `snippets`
2. ไม่ bind tool ใด ๆ
3. Prompt ต้องกำหนดให้:
   - ใช้เฉพาะข้อมูลใน snippets
   - รวมข้อมูลซ้ำให้เหลือครั้งเดียว
   - ตอบให้ครบและจัดรูปแบบอ่านง่าย
   - ห้ามใช้ความรู้ภายนอก
4. ถ้า `snippets` ว่าง ให้คืนประโยคคงที่โดยไม่เรียก LLM:

```text
I could not find this information in the knowledge base.
```

5. Citation สามารถคงไว้ได้ถ้าไม่เพิ่มความซับซ้อน แต่ไม่ใช่ requirement บังคับ

Acceptance criteria:

- Report Generator ไม่มี additional tools
- คำตอบอ้างอิงเฉพาะข้อมูลที่ได้รับ
- ไม่เกิด hallucination เมื่อไม่มีข้อมูล
- คำตอบไม่ซ้ำซ้อนและอ่านง่าย

### Phase 5: ลด LangGraph ให้เป็น Sequential Workflow

1. ปรับ `src/graph.py` ให้มีเพียงสอง nodes:
   - `data_retriever`
   - `report_generator`
2. ใช้ state ที่มีเพียง:
   - `query`
   - `snippets`
   - `report`
3. ต่อ graph แบบ:

```python
builder.add_edge(START, "data_retriever")
builder.add_edge("data_retriever", "report_generator")
builder.add_edge("report_generator", END)
```

4. ถอด conditional edges และ retry fields
5. ถอด imports ของ Router และ Query Rewriter

Acceptance criteria:

- ทุก query ผ่าน Data Retriever ก่อน Report Generator
- Raw snippets ถูกส่งผ่าน shared state
- Graph จบที่ Report Generator เสมอ

### Phase 6: ปรับ CLI และ Configuration

1. ลด `main.py` ให้ทำเฉพาะ:
   - โหลด environment variables
   - ตรวจ API key
   - compile graph
   - รับ query จาก command line หรือ interactive input
   - แสดง user query
   - แสดง retrieved snippets
   - แสดง final answer
2. ไม่แสดง route, retry attempts, retrieval mode หรือ telemetry
3. ให้ `src/config.py` มีเฉพาะค่าที่จำเป็น:

```text
MODEL_NAME
TEMPERATURE
KB_PATH
```

4. ใช้ Standard OpenAI API ผ่าน `ChatOpenAI` และ environment variable `OPENAI_API_KEY` ใน submission version โดยไม่เพิ่ม Azure OpenAI หรือ multi-provider configuration
5. เพิ่ม `.env.example`:

```dotenv
OPENAI_API_KEY=
MODEL_NAME=gpt-5-mini
KB_PATH=knowledge_base.txt
```

6. เพิ่ม `.gitignore` เพื่อไม่ให้ commit `.env`, virtual environment, cache และ bytecode

Acceptance criteria:

- ผู้ตรวจสามารถติดตั้งและรัน sample query จาก README ได้
- ผู้ตรวจสามารถตั้งค่า Standard OpenAI API ได้ด้วย `OPENAI_API_KEY`
- Secret ไม่อยู่ใน source code
- CLI แสดง evidence handoff และ final answer ชัดเจน

### Phase 7: ลด Dependencies

1. คงเฉพาะ dependencies ที่จำเป็น เช่น:
   - `langgraph`
   - `langchain-core`
   - `langchain-openai`
   - `python-dotenv`
   - `typing-extensions`
2. นำ dependency ที่ไม่ใช้แล้วออก:
   - `rank-bm25`
   - `numpy`
   - `streamlit`
3. Pin เวอร์ชันที่ผ่านการทดสอบ
4. ทดสอบติดตั้งใน Python 3.11 virtual environment ใหม่

Acceptance criteria:

- `pip install -r requirements.txt` สำเร็จใน clean environment
- ไม่มี unused direct dependency
- Import ทุก module สำเร็จ

### Phase 8: เขียน Tests ขั้นต่ำ

สร้าง tests เฉพาะ requirement หลัก:

1. `test_load_knowledge_base`
   - อ่านและแบ่ง chunks ได้
2. `test_keyword_search_returns_relevant_snippets`
   - ค้น international travel แล้วพบ section ที่ถูกต้อง
3. `test_keyword_search_returns_empty_for_unknown_query`
   - คำถามที่ไม่มีข้อมูลคืน empty list
4. `test_stopwords_do_not_create_false_positive`
   - Common words เพียงอย่างเดียวไม่ทำให้ unrelated chunk ถูกคืน
5. `test_generic_domain_terms_do_not_return_unrelated_sections`
   - คำว่า `policy` ไม่ทำให้ query เรื่อง international travel คืน policy อื่น
6. `test_keyword_search_returns_all_relevant_snippets`
   - คืนทุก section ที่เกี่ยวข้องโดยไม่มี fixed `TOP_K`
7. `test_retriever_executes_tool`
   - Retriever เรียก custom tool และส่ง raw snippets
8. `test_graph_handoff`
   - Output จาก Retriever ถูกส่งให้ Generator
9. `test_empty_retrieval_uses_not_found`
   - ไม่เรียก Report Generator LLM เมื่อไม่มี snippets

ไม่เพิ่ม comparative benchmark หรือ LLM-as-judge ใน submission scope

Acceptance criteria:

```bash
python -m unittest discover -v
```

ต้องผ่านทั้งหมดโดย retrieval tests ไม่ต้องใช้ network

### Phase 9: ปรับ README และหลักฐานการรัน

README ควรประกอบด้วย:

1. Objective สั้น ๆ
2. Architecture ของสอง agent
3. Project structure
4. Setup instructions
5. Environment configuration
6. Run command
7. Sample queries
8. Screenshots
9. Design decisions สั้น ๆ
10. Limitations

Sample queries ขั้นต่ำ:

```text
What is the policy on international travel?
Can I work remotely?
What is the CEO's salary?
```

สร้าง screenshots ใหม่จาก submission version:

- Positive single-topic query
- Positive query ที่ต้องรวมหลาย snippets
- Negative query ที่ต้องตอบ not found

Acceptance criteria:

- Screenshots ตรงกับโค้ดและ flow เวอร์ชันล่าสุด
- README ไม่มีคำอธิบาย Router, Rewriter, semantic หรือ hybrid
- คำสั่งทั้งหมดใน README ทดลองรันได้จริง

---

## 6. ไฟล์เดิมที่ต้องจัดการ

### แก้ไข

- `main.py`
- `requirements.txt`
- `README.md`
- `src/config.py`
- `src/graph.py`
- `src/agents/__init__.py`
- `src/agents/retriever.py`
- `src/agents/reporter.py`
- `src/tools/retrieval.py`
- tests ที่เกี่ยวข้อง

### สร้างใหม่

- `knowledge_base.txt`
- `.env.example`
- `.gitignore`
- `tests/test_graph.py`
- screenshots จากระบบเวอร์ชันสุดท้าย

### ถอดออกจาก submission branch

- `app.py`
- `src/agents/router.py`
- `src/agents/rewriter.py`
- `src/retrievers/`
- comparative evaluation scripts
- generated evaluation reports ที่ไม่เกี่ยวกับ requirement
- screenshots และเอกสารที่อธิบาย advanced flow เดิม

การถอดไฟล์ควรทำหลังจากสร้าง backup branch หรือ tag แล้ว

---

## 7. Definition of Done

งานถือว่าเสร็จเมื่อผ่านทุกข้อดังนี้:

- [ ] มี `knowledge_base.txt` ที่ project root
- [ ] ระบบมี agent เพียง Data Retriever และ Report Generator
- [ ] Data Retriever ถูก configure ให้ใช้ custom tool
- [ ] Data Retriever execute custom tool จริง
- [ ] Tool คืน raw relevant snippets
- [ ] Report Generator ไม่มี tool
- [ ] LangGraph เป็น sequential flow สอง nodes
- [ ] Final answer ใช้เฉพาะข้อมูลจาก snippets
- [ ] คำถามที่ไม่มีข้อมูลได้ deterministic not-found answer
- [ ] รัน sample international travel query สำเร็จ
- [ ] Unit tests ผ่านทั้งหมด
- [ ] ติดตั้งและรันได้ใน clean Python 3.11 environment
- [ ] มี `.env.example` และไม่มี secret ใน repository
- [ ] README สอดคล้องกับ implementation ล่าสุด
- [ ] มี screenshots อย่างน้อย 3 queries
- [ ] ไฟล์ทั้งหมดถูกจัดเตรียมสำหรับส่งผ่าน GitHub repository

---

## 8. Final Verification Commands

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# เพิ่ม OPENAI_API_KEY ใน .env

python -m unittest discover -v
python main.py "What is the policy on international travel?"
python main.py "Can I work remotely?"
python main.py "What is the CEO's salary?"
```

ผลลัพธ์ที่คาดหวัง:

- คำถาม international travel ได้ snippets ที่เกี่ยวข้องและคำตอบสรุป
- คำถาม remote work ได้คำตอบจาก policy ในไฟล์
- คำถาม CEO salary ได้ not-found sentence
- ไม่มี Router, retry หรือ retrieval mode ปรากฏใน execution flow
