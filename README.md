# Security Platform AI Agent

An AI-powered security assistant built with **LangGraph** that answers natural language questions about vulnerabilities, pipeline findings, and application risk — backed by a local mock security platform served over MCP.

---

## Architecture

```
User query
    ↓
LangGraph Agent
    ├── Classifier node  →  "data" / "doc" / "mixed" / "synthesis" / "chart"
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
    ├── synthesis → Synthesis Node  →  reasons over conversation history (no tool call)
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

**Conversation memory** is persisted per session via LangGraph's `MemorySaver`. The classifier uses recency-weighted history (last 4 turns at 1500 chars, older at 200 chars) to rewrite context-dependent questions into self-contained queries. A deterministic multi-entity guard ensures comparison queries always include all active entities even if the LLM abbreviated the rewrite.

**Hallucination prevention** runs at three layers: the RAG distance filter drops low-quality chunks before they reach the LLM; `validate_response` (LLM-as-Judge) scores every final answer — including synthesis answers validated against conversation history — and appends a warning when claims can't be verified; and prompt injection defense (pattern-blocking + XML delimiter isolation + tag sanitization) prevents external data from hijacking the system prompt.

---

## Features

- **Query classification** — routes to live data, documentation, synthesis, or both; classifier uses recent conversation history to rewrite context-dependent questions into self-contained queries before routing
- **Synthesis routing** — follow-up questions that reason over prior turns ("compare both", "which is worse?", "summarize what we found") are routed to a dedicated synthesis node that answers from conversation history without making a new tool call; validated by the same LLM-as-Judge groundedness check as data responses
- **Full filter coverage** — severity, CVE ID, application/service, keyword, date ranges, pipeline, stage, scanner tool, git branch
- **ID-based lookup** — ask "tell me about PIPE-006" or "show ISS-001" and the agent calls `get_pipeline_issues(id='PIPE-006')` directly; works for both ISS-* and PIPE-* IDs
- **Deterministic rendering** — data results are rendered directly as markdown so the LLM can't drop rows; LLM formatter is used only for aggregation queries ("how many…", "total…")
- **General group-by aggregation** — queries like "how many issues by severity / application / category / status?" bypass LLM tool selection entirely (fetches both tools with no filters, then counts from raw JSON); avoids the LLM silently adding `status='open'` or making per-value calls that drop rows
- **Chart generation** — severity distribution and top vulnerable apps charts rendered inline in the browser as base64 PNG; appear below the assistant message bubble
- **Streaming responses** — token-by-token streaming via Server-Sent Events (`/chat/stream`); the backend emits `status` events at each pipeline stage ("Analyzing your question…", "Fetching security data…", "Generating response…") before the first token, replacing the typing indicator with live progress text; deterministic paths fall back to `final_response` in the done event
- **Multi-query RAG** — the retriever generates 3 alternative phrasings of every doc query, runs all four in parallel, and deduplicates results — improves recall when the user's wording doesn't closely match chunk wording
- **RAG over docs** — semantic search over `docs/connectors.md` and `docs/dashboard.md` with source attribution; breadcrumb-enriched chunks so child sections retain parent header context in embeddings
- **RAG confidence threshold** — chunks above a configurable cosine distance threshold are filtered before reaching the LLM; fully off-topic queries return an "I don't have information" message without calling the LLM (`RAG_DISTANCE_THRESHOLD`, default `0.5`)
- **LLM-as-Judge validation** — a `validate_response` node scores every response for groundedness (0–1); responses below `0.7` get a `⚠️ Validation warning` block listing unverified claims; the confidence score is shown as a badge on every assistant message in the UI; synthesis answers are validated against conversation history rather than raw tool output, so hallucinated facts ("what CVSS score did it have?") get caught even with no tool call
- **Multi-entity deterministic guard** — when comparison keywords appear ("compare", "both", "vs") the classifier output is checked against known active entities; any entity that was in conversation state but dropped from the rewritten query is injected back before routing, ensuring "compare both" always covers both services even if the LLM abbreviated the query
- **Recency-weighted history truncation** — the last 4 conversation turns get up to 1500 chars each in the classifier prompt; older turns are capped at 200 chars; this keeps the prompt compact while preserving full entity names in recent context, preventing entity resolution failures at conversation depth
- **LangSmith observability** — optional tracing via env vars; every graph run, LLM call, and node I/O is captured automatically; RAG distances and groundedness scores are surfaced as structured output fields per trace
- **Multi-turn follow-ups** — per-session memory plus query contextualization, so "what are the steps?" resolves against the previous turn
- **React UI** — browser-based chat interface with streaming, confidence badges (MCP / RAG / Mixed / Synthesis), inline charts, and Markdown rendering; 8 categorised starter hints on the welcome screen cover every agent capability
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
Show a chart of issues by severity                → inline bar chart rendered in the browser
```

Multi-turn context and synthesis:

```
> Show me issues in auth-service
  ...(1 critical issue: ISS-003)
> Now show me payment-service issues
  ...(ISS-005 + 2 pipeline findings)
> compare both
  ...(synthesis: side-by-side comparison — multi-entity guard ensures both services
       are covered even if the classifier abbreviated the rewritten query)

> Show me all critical issues
  ...(3 results listed)
> Which one would you fix first and why?
  ...(synthesis: reasoned prioritisation from history — no new tool call)
> What is the CVSS score of the first issue?
  ...(synthesis: "not provided in the platform's data model, severity: critical" — refuses to invent a number)
```

Follow-up anaphora resolves correctly across turns:

```
> Show me all critical issues                     → lists ISS-001 (user-service), ISS-003, ISS-005
> Tell me more about the first one                → synthesis: details on ISS-001
> What service is it in?                          → synthesis: user-service
> Show me all issues in that service              → data: fetches user-service (resolves "that service")
```

The validator appends a confidence warning when the LLM makes claims not found in retrieved data:

```
> Show me issues in payment-service
  ...(lists issues with CVE details)
> What other CVEs did you mention that weren't in that list?
  ⚠️ Validation warning (confidence: 48%): some claims could not be verified...
```

The confidence badge appears on every assistant message — green for grounded, red for flagged.
The query-type badge (MCP / RAG / Mixed / Synthesis) shows which agent path answered.

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
tests/          Unit tests for all components (174 Python tests + 12 frontend tests)
main.py         CLI entry point
```

---

## Running Tests

```bash
pytest tests/ -v
```
