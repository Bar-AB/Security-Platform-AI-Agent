# Security Platform AI Agent

An AI-powered security assistant built with **LangGraph** that answers natural language questions about vulnerabilities, pipeline findings, and application risk — backed by a local mock security platform served over MCP.

---

## Architecture

```
User query
    ↓
LangGraph Agent
    ├── Classifier node  →  "data" / "doc" / "mixed"
    │
    ├── data  →  MCP Tool Node  →  FastMCP Mock Server
    │                ↓
    │           get_security_issues / get_applications / get_pipeline_issues
    │
    ├── doc   →  RAG Node  →  ChromaDB  →  docs/*.md
    │
    └── mixed →  both, combined response
                    ↓
              Format Response Node  →  deterministic renderer (data)
                                    →  LLM formatter (aggregation / mixed)
                                    →  chart auto-generated + opened (matplotlib)
```

**Conversation memory** is persisted per session via LangGraph's `MemorySaver`, with follow-up resolution: the classifier uses recent history to rewrite context-dependent questions (e.g. "what are the steps?") into self-contained queries before routing.

---

## Features

- **Query classification** — routes to live data, documentation, or both
- **Full filter coverage** — severity, CVE ID, application/service, keyword, date ranges, pipeline, stage, scanner tool, git branch
- **Deterministic rendering** — data results are rendered directly as markdown so the LLM can't drop rows; LLM formatter is used only for aggregation queries ("how many…", "total…")
- **Chart generation** — severity distribution and top vulnerable apps charts auto-open in Preview after any data query
- **RAG over docs** — semantic search over `docs/connectors.md` and `docs/dashboard.md` with source attribution
- **Multi-turn follow-ups** — per-session memory plus query contextualization, so "what are the steps?" resolves against the previous turn
- **Mock security platform** — realistic CVE-style data served via FastMCP (no external credentials needed)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent | LangGraph + LangChain |
| LLM | OpenAI GPT-4o |
| Vector store | ChromaDB (local) |
| Embeddings | `text-embedding-3-small` |
| MCP server | FastMCP (mcp >= 1.0) |
| Charts | Matplotlib |
| Language | Python 3.12 |

---

## MCP Tools

| Tool | Filters |
|------|---------|
| `get_security_issues` | severity, category, status, application, keyword, cve_id, discovered_after/before, limit |
| `get_applications` | min_risk_score, limit (sorted by risk score descending) |
| `get_pipeline_issues` | severity, pipeline, stage, tool, branch (prefix match), keyword, detected_after/before, limit |

---

## Setup

**1. Install dependencies**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**2. Set environment variables**
```bash
cp .env.example .env
# add your OPENAI_API_KEY
```

**3. Start the mock MCP server**
```bash
uvicorn mock_server.main:app --port 8000
```

**4. Run the agent**
```bash
python main.py
```

The RAG index is built automatically on first run.

---

## Example Queries

```
Show me all critical issues
What CVEs affect the payment service?
Top 3 most vulnerable applications
Show me Semgrep findings from the auth pipeline
How many issues were discovered in November 2024?
What are the SAST findings on the main branch?
How do I connect Jira to the platform?
```

Follow-up questions work within a conversation — the agent resolves them against prior turns:

```
> How do I connect the GitHub connector?
  ...(answers with the setup steps)
> What are the steps?
  ...(knows you still mean the GitHub connector)
```

---

## Project Structure

```
agent/          LangGraph nodes, graph builder, state, prompts, chart generation
mcp_client/     MCP client + LangChain tool wrappers
mock_server/    FastMCP server with Pydantic models and mock security data
rag/            Document indexer (ChromaDB) and retriever
docs/           Markdown knowledge base for RAG
tests/          Unit tests for all components
main.py         CLI entry point
```

---

## Running Tests

```bash
pytest tests/ -v
```
