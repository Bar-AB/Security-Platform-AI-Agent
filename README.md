# Security Platform AI Agent

An AI-powered security assistant built with **LangGraph** that answers natural language questions about vulnerabilities, pipeline findings, and application risk — backed by a local mock security platform served over MCP.

---

## Architecture

```
User query
    ↓
LangGraph Agent
    ├── Classifier node  →  "data" / "doc" / "mixed" / "chart"
    │
    ├── data  →  MCP Tool Node  →  FastMCP Mock Server
    │                ↓
    │           get_security_issues / get_applications / get_pipeline_issues
    │
    ├── doc   →  RAG Node  →  ChromaDB  →  docs/*.md
    │                ↓
    │           [distance threshold filter — drops off-topic chunks]
    │
    ├── mixed →  both branches run in parallel, combined response
    │
    └── chart →  renders prior-turn data as chart  →  END (bypasses validator)
                    ↓
              Format Response Node  →  [no-context guard] → "I don't have info" if nothing retrieved
                                    →  deterministic renderer (pure data)
                                    →  LLM formatter (aggregation / mixed)
                    ↓
              Validate Response Node  →  LLM-as-Judge (groundedness 0–1)
                                      →  appends ⚠️ warning if score < 0.7
```

**Conversation memory** is persisted per session via LangGraph's `MemorySaver`, with follow-up resolution: the classifier uses recent history to rewrite context-dependent questions (e.g. "what are the steps?") into self-contained queries before routing.

**Hallucination prevention** runs at two layers: the RAG distance filter drops low-quality chunks before they reach the LLM, and `validate_response` (LLM-as-Judge) scores the final answer against the retrieved context and appends a warning when claims can't be verified.

---

## Features

- **Query classification** — routes to live data, documentation, or both
- **Full filter coverage** — severity, CVE ID, application/service, keyword, date ranges, pipeline, stage, scanner tool, git branch
- **ID-based lookup** — ask "tell me about PIPE-006" or "show ISS-001" and the agent calls `get_pipeline_issues(id='PIPE-006')` directly; works for both ISS-* and PIPE-* IDs
- **Deterministic rendering** — data results are rendered directly as markdown so the LLM can't drop rows; LLM formatter is used only for aggregation queries ("how many…", "total…")
- **General group-by aggregation** — queries like "how many issues by severity / application / category / status?" bypass LLM tool selection entirely (fetches both tools with no filters, then counts from raw JSON); avoids the LLM silently adding `status='open'` or making per-value calls that drop rows
- **Chart generation** — severity distribution and top vulnerable apps charts rendered inline in the browser as base64 PNG; appear below the assistant message bubble
- **Streaming responses** — token-by-token streaming via Server-Sent Events (`/chat/stream`); the backend emits `status` events at each pipeline stage ("Analyzing your question…", "Fetching security data…", "Generating response…") before the first token, replacing the typing indicator with live progress text; deterministic paths fall back to `final_response` in the done event
- **Multi-query RAG** — the retriever generates 3 alternative phrasings of every doc query, runs all four in parallel, and deduplicates results — improves recall when the user's wording doesn't closely match chunk wording
- **RAG over docs** — semantic search over `docs/connectors.md` and `docs/dashboard.md` with source attribution; breadcrumb-enriched chunks so child sections retain parent header context in embeddings
- **RAG confidence threshold** — chunks above a configurable cosine distance threshold are filtered before reaching the LLM; fully off-topic queries return an "I don't have information" message without calling the LLM (`RAG_DISTANCE_THRESHOLD`, default `0.5`)
- **LLM-as-Judge validation** — a `validate_response` node scores every response for groundedness (0–1); responses below `0.7` get a `⚠️ Validation warning` block listing unverified claims; the confidence score is shown as a badge on every assistant message in the UI
- **LangSmith observability** — optional tracing via env vars; every graph run, LLM call, and node I/O is captured automatically; RAG distances and groundedness scores are surfaced as structured output fields per trace
- **Multi-turn follow-ups** — per-session memory plus query contextualization, so "what are the steps?" resolves against the previous turn
- **React UI** — browser-based chat interface with streaming, confidence badges, inline charts, and Markdown rendering
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
| Charts | Matplotlib (base64 PNG transport) |
| Observability | LangSmith (optional) |
| API | FastAPI with SSE streaming |
| Frontend | React + TypeScript + Vite + Tailwind |
| Language | Python 3.12 |

---

## MCP Tools

| Tool | Filters |
|------|---------|
| `get_security_issues` | **id** (exact, e.g. `ISS-001`), severity, category, status, application, keyword, cve_id, discovered_after/before, limit |
| `get_applications` | min_risk_score, limit (sorted by risk score descending) |
| `get_pipeline_issues` | **id** (exact, e.g. `PIPE-006`), severity, pipeline, stage, tool, branch (prefix match), keyword, detected_after/before, limit |

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
# optional tuning:
# RAG_DISTANCE_THRESHOLD=0.5   (cosine distance cutoff for RAG chunk filtering; lower = stricter)
# RAG_TOP_K=5                  (number of chunks retrieved before filtering)
# LangSmith tracing (optional — create a free account at smith.langchain.com):
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=ls__...
# LANGCHAIN_PROJECT=security-platform-agent
```

**3. Start the mock MCP server**
```bash
uvicorn mock_server.main:app --port 8000
```

**4. Start the API server**
```bash
uvicorn api.main:app --port 8001
```

The RAG index is built automatically on first run.

**5. Start the frontend** (separate terminal)
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

**Alternative: CLI mode** (no frontend needed)
```bash
python main.py
```

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
What connectors are available on the platform?
Are there any Jira connector issues?
How do I configure a Kubernetes load balancer?   → returns "I don't have information about that"
Show me PIPE-006                                  → direct ID lookup for a specific pipeline finding
Tell me about ISS-001                             → direct ID lookup for a specific security issue
How many issues are there by severity?            → deterministic count: 5 critical, 8 high, 5 medium
How many issues per application?                  → group-by any field: application, category, status
```

Follow-up questions work within a conversation — the agent resolves them against prior turns:

```
> How do I connect the GitHub connector?
  ...(answers with the setup steps)
> What are the steps?
  ...(knows you still mean the GitHub connector)
```

The validator appends a confidence warning when the LLM makes claims not found in retrieved data:

```
> Show me issues in payment-service
  ...(lists issues with CVE details)
> What other CVEs did you mention that weren't in that list?
  ⚠️ Validation warning (confidence: 48%): some claims could not be verified...
```

The confidence badge appears on every assistant message — green for grounded, red for flagged.

---

## Project Structure

```
agent/          LangGraph nodes, graph builder, state, prompts, chart generation
api/            FastAPI app — /chat (buffered) + /chat/stream (SSE) endpoints
mcp_client/     MCP client + async LangChain tool wrappers
mock_server/    FastMCP server with Pydantic models and mock security data
rag/            Document indexer (ChromaDB) and retriever (multi-query)
docs/           Markdown knowledge base for RAG
frontend/       React + TypeScript chat UI (Vite, Tailwind, react-markdown)
tests/          Unit tests for all components (118 tests)
main.py         CLI entry point
```

---

## Running Tests

```bash
pytest tests/ -v
```
