# Security-Platform-AI-Agent

## Project Overview

An AI-powered Security Assistant Agent built with LangChain/LangGraph that connects to a security platform via MCP and answers questions using vectorized documentation.

This is a portfolio project demonstrating:
- LangGraph agent with query classification and routing
- RAG pipeline over markdown documentation
- MCP integration with a local mock server
- FastAPI mock server simulating a real security platform API

**Not tied to any specific company or platform.** All data is mocked locally.

---

## Architecture

```
User query
    ↓
LangGraph Agent (query classifier)
    ├── "data query" → MCP Tool → Mock Security Server (FastAPI)
    ├── "doc query"  → RAG Tool  → ChromaDB vector store
    └── "mixed"      → both, combined response
```

### Components

- `agent/` — LangGraph agent definition, nodes, routing logic
- `rag/` — Document loader, chunker, embedder, ChromaDB index
- `mcp/` — MCP client + tool definitions
- `mock_server/` — FastAPI local server with realistic mock security data
- `docs/` — Markdown knowledge base (connectors.md, dashboard.md)
- `main.py` — Entry point (CLI or simple chat loop)

---

## Tech Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Agent | LangGraph | Stateful graph-based agent, industry standard |
| LLM | OpenAI GPT-4o (or Azure OpenAI) | Best tool use support |
| Vector store | ChromaDB | Local, no external service needed |
| Embeddings | `text-embedding-3-small` | Fast, cheap, good quality |
| Mock server | FastAPI | Familiar stack, realistic REST API |
| Language | Python 3.12 | Primary stack |

---

## MCP Tools (Generic)

| Tool | Description |
|------|-------------|
| `get_security_issues` | Fetch issues with optional filters (severity, category, status) |
| `get_applications` | Fetch application list with risk scores |
| `get_pipeline_issues` | Fetch CI/CD pipeline security findings |

The mock server returns realistic CVE-style data. No external credentials needed.

---

## Assignment Requirements

### Core (100 points)
1. **LangGraph Agent (40%)** — query classification, routing, conversation flow
2. **MCP Integration (35%)** — connect to mock server, fetch + analyze data, error handling
3. **RAG Implementation (25%)** — chunk docs, semantic search, source attribution

### Bonus (+15%)
- Chart generation: severity distribution, top vulnerable apps, category breakdown
- Multi-agent architecture

---

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start the mock MCP server
uvicorn mock_server.main:app --port 8000

# Run the agent (CLI chat loop)
python main.py

# Run tests
pytest
```

---

## Environment Variables

```
OPENAI_API_KEY=...          # Required — used for LLM + embeddings
# or
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT=...
```

---

## Python Coding Conventions

- Use explicit type hints on all function signatures
- Use Pydantic models for structured data — never raw dicts
- Early returns / guard clauses to keep functions flat (avoid deep nesting)
- Always catch specific exception types — no bare `except:`, never silently swallow errors
- Use `logger.exception()` when logging caught exceptions
- Boolean variables use `is_` / `has_` / `are_` prefixes
- Line length: 100 characters
- All code in classes — avoid standalone functions outside of classes
- Use `_` prefix for private attributes and methods

### Testing
- Wrap test functions in a class named `Test<ClassUnderTest>`
- Use `@pytest.fixture` as class methods, not module-level
- Use `patch.object` for mocking, not `patch` with string paths

---

## Key Design Decisions

- **Mock server over live API**: Makes the project fully runnable without credentials, better for portfolio/demo
- **ChromaDB over cloud vector DB**: No external service dependency, runs fully local
- **Single repo**: Agent + mock server + RAG all in one repo for easy setup
- **Python not Node.js**: Python is the primary stack, Node.js is only the bonus requirement

---

## Developer Context

Built by Bar Abulher as a portfolio project. Background: 18 months backend/AI engineering at a cybersecurity startup — shipped RAG pipelines, SIEM integrations, enrichment modules. This project intentionally mirrors real production patterns from that experience.

Target: backend SWE and AI Engineer roles at product companies.
