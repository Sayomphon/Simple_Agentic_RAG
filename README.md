# Simple Agentic RAG with LangGraph

A small, auditable two-agent RAG system built for the AI Engineer Programming
Test. A Data Retriever Agent searches a local text knowledge base through a
custom deterministic tool. A Report Generator Agent then turns the retrieved
raw sections into a concise answer grounded only in that evidence.

## Architecture

```text
User query
    |
    v
Data Retriever Agent
    |
    | forced call: search_knowledge_base(query)
    v
Raw relevant snippets
    |
    | LangGraph shared state
    v
Report Generator Agent
    |
    v
Final grounded answer
```

The LangGraph execution path is fixed:

```text
START -> data_retriever -> report_generator -> END
```

The complete state contract is:

```python
class PipelineState(TypedDict):
    query: str
    snippets: list[str]
    report: str
```

### Agent responsibilities

1. **Data Retriever Agent**
   - Is configured with the custom `search_knowledge_base` tool.
   - Must call the tool exactly once using the user query.
   - Executes the requested tool call and writes its unmodified `list[str]`
     output to `snippets`.
   - Does not answer or summarize the evidence.

2. **Report Generator Agent**
   - Receives only the user query and retrieved snippets.
   - Has no tools.
   - Uses only facts present in the snippets and merges repeated information.
   - Returns a fixed response without calling the LLM when no evidence exists:

     ```text
     I could not find this information in the knowledge base.
     ```

## Retrieval design

`knowledge_base.txt` contains 10 clearly delimited sections covering remote
work, leave, international travel, expenses, a payment product, and customer
support.

The custom tool is deliberately small and transparent:

1. Read `knowledge_base.txt` as UTF-8.
2. Split it at headings in the form `--- Section Title ---`.
3. Normalize the query and sections with
   `re.findall(r"[a-z0-9]+", text.lower())`.
4. Remove English stopwords and broad enterprise terms such as `policy`,
   `company`, `employee`, and `information`.
5. Use unique remaining query terms for matching and scoring.
6. Accept a section when a term anchors to its title, or when all specific
   query terms occur in its body. This avoids incidental one-word matches.
7. Sort by matched-term count, then original document order for deterministic
   ties.
8. Return every section that passes the rule; there is no fixed result cap.

The tool returns source sections as raw text. It never asks an LLM to retrieve,
summarize, or enrich the content.

## Project structure

```text
.
├── knowledge_base.txt
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── screenshots/
│   └── .gitkeep
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

## Setup

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Add a Standard OpenAI API key to `.env`:

```dotenv
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-5-mini
TEMPERATURE=0
KB_PATH=knowledge_base.txt
```

`.env` is ignored by Git. Do not commit real API keys.

## Run

Run a single query:

```bash
python main.py "What is the policy on international travel?"
```

Or start interactive mode:

```bash
python main.py
```

The CLI displays three auditable stages:

1. User query
2. Raw snippets handed from the Data Retriever to the Report Generator
3. Final grounded answer

Suggested queries:

```text
What is the policy on international travel?
Can I work remotely?
What is the CEO's salary?
```

The first query retrieves three complementary international-travel sections.
The second retrieves two complementary remote-working sections. The last query
retrieves nothing and returns the exact deterministic not-found sentence.

## Tests

Run the complete offline unit test suite:

```bash
python -m unittest discover -v
```

The tests cover knowledge-base loading, focused keyword matches, unknown
queries, false-positive protection, complete relevant-section retrieval,
deterministic ordering, forced tool execution, raw-snippet handoff, graph
topology, and the no-LLM not-found path. They do not require an API key or
network access.

## Screenshots

Live screenshots will be added after the three sample queries complete
successfully against the configured OpenAI project. The old screenshots from
the larger prototype were intentionally removed because they no longer matched
this submission workflow.

## Design decisions

- **Local text file:** keeps the evidence easy to inspect and matches the test
  requirement directly.
- **Deterministic retrieval:** makes ranking explainable, repeatable, and
  testable without an API call.
- **Forced tool call:** structurally prevents the Retriever Agent from
  replacing evidence retrieval with a direct answer.
- **Raw state handoff:** makes the boundary between retrieval and generation
  explicit and observable.
- **No-evidence short circuit:** removes hallucination risk and API cost for
  empty retrieval results at the generator stage.
- **Pinned dependencies:** reduces installation drift in reviewer
  environments.

## Limitations

- Matching is lexical and does not understand synonyms or conceptual
  similarity.
- A question must contain specific English terms that occur in the knowledge
  base.
- Broad questions containing only generic terms, such as “What policies are
  available?”, intentionally return not found; the user must specify a topic.
- Inflected forms are not stemmed, so `remote` and `remotely` are distinct
  terms unless both occur in the relevant section.
- The local-file approach is appropriate for this small assignment but would
  need indexing, access controls, document lifecycle management, and retrieval
  evaluation before use with a large enterprise corpus.
