# AI Engineer Programming Test: Agentic AI

## Objective

Assess the candidate's ability to design and implement a simple agentic AI system using multi-agent orchestration, incorporating Retrieval-Augmented Generation (RAG).

## Framework Options

Candidates can choose **one** of the following free and simple frameworks:

1. **Langchain/Langgraph**
2. **OpenAI Agents SDK**

## Tools & Requirements

- **Data Handling:** Standard Python libraries and a simple local file (e.g., a `.txt` file) to simulate a knowledge base.

- **LLM:** The candidate can use any LLM available in the market (Or please send an email to [kanit.mekritthikrai@bangkokbank.com](mailto:kanit.mekritthikrai@bangkokbank.com) to grant the access to the following gpt-5-mini

  ```python
  endpoint = "https://oaibblinnocandiddate01.openai.azure.com/"
  model_name = "gpt-5-mini"
  deployment = "gpt-5-mini"
  subscription_key = "<contact>"
  ```

- **Required Skills:** Python programming, basic understanding of RAG, API interaction, and prompt design.

## Scenario

Build a system with two agents:

1. **"Data Retriever" Agent (RAG):** Specializes in retrieving specific information from a provided knowledge base (a simple text file). It does not answer questions directly but provides relevant text snippets.

2. **"Report Generator" Agent:** Receives the snippets from the Data Retriever and synthesizes them into a cohesive, non-redundant, and well-formatted answer to a user's query.

## Task Instructions

Create a text file named `knowledge_base.txt` containing a few paragraphs of sample information (e.g., company policies, product descriptions, or general facts about a topic like "cats").

## Agent Definitions

### Data Retriever

- **Role/Instructions:** Expert in information retrieval. Searches the `knowledge_base.txt` file for all snippets relevant to the user's request.

- **Tool:** Implement a custom Python function/tool that reads the `knowledge_base.txt` file and performs a simple keyword or basic semantic search. The agent should be configured to use this tool.

- **Output:** Return raw, relevant text chunks to the Report Generator.

### Report Generator

- **Role/Instructions:** Expert writer and synthesizer. Uses the provided information snippets to formulate a comprehensive, high-quality answer for the end-user.

- **Tool:** No additional tools needed.

- **Output:** The final, polished answer presented to the user.

## Task Orchestration

- Orchestrate a sequential workflow where the Data Retriever's output is passed as input to the Report Generator.

- Candidates using the **OpenAI Agents SDK/Langchain** should use either a "handoff" pattern or an "agent-as-tool" pattern or any other pattern to coordinate the two agents.

## Execution and Evaluation

- Run the system with a sample query (e.g., "What is the policy on international travel?").

- The candidate should submit the complete Python code, the `knowledge_base.txt` file, and a screenshot of the final output for a few different queries on the GitHub repository.

### Evaluation Criteria

- Correct use of the chosen framework to define agents and orchestrate the workflow.

- Effective implementation of the RAG mechanism (the custom data retrieval tool).

- Clarity and quality of the final output (non-redundant, accurate based on provided info, well-formatted).

- Code structure, readability, and adherence to Python best practices.
