# Security Platform AI Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph agent that classifies user queries and routes them to either a FastMCP mock security server, a ChromaDB RAG pipeline over markdown docs, or both.

**Architecture:** A `StateGraph` with a classifier node that labels each query as `data`, `doc`, or `mixed`, then conditional edges route to an MCP tool node, a RAG retrieval node, or both in sequence before a shared response formatter. Conversation memory is persisted per session via `MemorySaver`.

**Tech Stack:** Python 3.12, LangGraph, LangChain, OpenAI GPT-4o + text-embedding-3-small, ChromaDB, FastMCP (mcp>=1.0), FastAPI, pytest

---

## File Structure

```
mock_server/
    __init__.py          empty
    models.py            Pydantic models: SecurityIssue, Application, PipelineIssue + enums
    data.py              Static lists: MOCK_ISSUES, MOCK_APPLICATIONS, MOCK_PIPELINE_ISSUES
    main.py              FastMCP instance, 3 @mcp.tool() definitions, app = mcp.streamable_http_app()

rag/
    __init__.py          empty
    indexer.py           RAGIndexer: load docs/ → chunk by markdown headers → embed → ChromaDB
    retriever.py         RAGRetriever: MMR search → return list[Document] with metadata

mcp/
    __init__.py          empty
    client.py            MCPClient: async calls to MCP server via mcp SDK, sync wrapper
    tools.py             SecurityMCPTools: wraps MCPClient methods as LangChain StructuredTools

agent/
    __init__.py          empty
    state.py             AgentState TypedDict + QueryClassification Pydantic model
    prompts.py           CLASSIFIER_PROMPT, FORMATTER_PROMPT templates
    nodes.py             AgentNodes class: classify_query, mcp_node, rag_node, format_response
    graph.py             GraphBuilder class: builds + compiles the StateGraph

docs/
    connectors.md        RAG knowledge base — integration setup guides
    dashboard.md         RAG knowledge base — dashboard and severity guide

tests/
    __init__.py          empty
    test_mock_server.py  Tests for data models, filtering logic
    test_rag.py          Tests for RAGIndexer + RAGRetriever (temp ChromaDB dir)
    test_mcp_tools.py    Tests for SecurityMCPTools with mocked MCPClient
    test_agent_nodes.py  Tests for classify_query, mcp_node, rag_node with mocked deps

main.py                  CLI entry point — sync while loop, MemorySaver thread
.env                     OPENAI_API_KEY=...  (gitignored)
```

---

## Task 1: Project Scaffold + RAG Knowledge Base Docs

**Files:**
- Create: all `__init__.py` files, `docs/connectors.md`, `docs/dashboard.md`, `.env` template

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p mock_server rag mcp agent docs tests
touch mock_server/__init__.py rag/__init__.py mcp/__init__.py agent/__init__.py tests/__init__.py
```

- [ ] **Step 2: Create `docs/connectors.md`**

```markdown
# Connectors Guide

Connectors allow the security platform to ingest vulnerability data from external services.
Navigate to **Settings → Integrations** to manage connectors.

## Jira Connector

The Jira connector syncs security issues as Jira tickets for team triage.

### Setup Steps
1. Go to Settings → Integrations → Jira.
2. Enter your Jira site URL (e.g. `https://yourcompany.atlassian.net`).
3. Generate an API token in your Jira profile under **Security → API tokens**.
4. Enter your email and API token, then click **Connect**.
5. Select the target Jira project where tickets will be created.
6. Choose a severity threshold — only issues at or above this level create tickets.

### Supported Features
- Auto-create tickets for new critical and high issues.
- Sync issue status between platform and Jira (open/resolved).
- Attach CVE references and remediation notes to tickets.

### Troubleshooting
- **"401 Unauthorized"**: The API token may have expired. Regenerate it in Jira.
- **"Project not found"**: Ensure the Jira project key matches exactly (case-sensitive).
- **Tickets not syncing**: Verify the webhook URL is whitelisted in your Jira firewall rules.

---

## GitHub Connector

The GitHub connector scans repositories for secrets, vulnerable dependencies, and code vulnerabilities.

### Setup Steps
1. Go to Settings → Integrations → GitHub.
2. Click **Install GitHub App** — this redirects to GitHub.
3. Select the organization or repositories to grant access.
4. Click **Install & Authorize**.
5. The platform will begin scanning within 15 minutes of installation.

### Supported Features
- Dependency vulnerability scanning (via `package.json`, `requirements.txt`, `pom.xml`, etc.).
- Secret detection (API keys, tokens, credentials committed to code).
- Code scanning results ingested from GitHub Advanced Security (GHAS).
- Pull request annotations for new findings.

### Troubleshooting
- **App not appearing in GitHub**: Clear browser cache and retry the OAuth flow.
- **Scans not running**: Ensure the GitHub App has `read` permissions on repository contents.
- **Missing repos**: Re-install the app and select all repositories explicitly.

---

## AWS Security Hub Connector

The AWS connector ingests findings from Security Hub, GuardDuty, and Inspector.

### Setup Steps
1. Go to Settings → Integrations → AWS.
2. Enter your AWS Account ID and the region of your Security Hub instance.
3. Create an IAM role in your AWS account with the following policy:
   ```json
   {
     "Effect": "Allow",
     "Action": ["securityhub:GetFindings", "guardduty:ListFindings"],
     "Resource": "*"
   }
   ```
4. Enter the IAM role ARN and click **Verify & Connect**.
5. Select which finding types to import (Security Hub, GuardDuty, Inspector, or all).

### Troubleshooting
- **"Access Denied"**: The IAM role trust policy must allow `sts:AssumeRole` from the platform's AWS account.
- **No findings appearing**: Confirm Security Hub is enabled in the specified region.
- **Duplicate findings**: Disable native Security Hub cross-region aggregation to avoid double-ingestion.

---

## Slack Connector

The Slack connector sends real-time alerts for new critical issues to a Slack channel.

### Setup Steps
1. Go to Settings → Integrations → Slack.
2. Click **Add to Slack** — this redirects to Slack's OAuth page.
3. Select the workspace and the channel for alerts.
4. Click **Allow**.
5. Set the minimum severity for notifications (default: Critical).

### Troubleshooting
- **Bot not posting**: Ensure the bot has `chat:write` permission in the target channel.
- **Missing alerts**: Check the severity filter — Medium and Low are off by default.
```

- [ ] **Step 3: Create `docs/dashboard.md`**

```markdown
# Dashboard Guide

The dashboard gives a real-time view of your organization's security posture.

## Severity Levels

The platform uses four severity levels, aligned with CVSS scores:

| Severity | CVSS Range | Meaning | SLA |
|----------|-----------|---------|-----|
| Critical | 9.0–10.0 | Immediate exploitation risk, patch or mitigate within 24h | 24 hours |
| High | 7.0–8.9 | Likely exploitable, requires urgent attention | 7 days |
| Medium | 4.0–6.9 | Exploitable under specific conditions | 30 days |
| Low | 0.1–3.9 | Minimal risk, fix in next regular release cycle | 90 days |

## Risk Score

Each application is assigned a **Risk Score** from 0 to 10, calculated from:
- Number and severity of open issues
- Time open beyond SLA
- Asset criticality (business impact rating)
- Exposure level (internet-facing vs. internal)

A score above **8.0** is considered high risk and triggers an alert to the application owner.

## Dashboard Filters

Use the filter bar at the top of the dashboard to narrow findings:

- **Severity**: Filter by Critical / High / Medium / Low (multi-select).
- **Status**: Open, In Progress, Resolved.
- **Category**: Injection, XSS, Broken Auth, Exposed Data, Misconfiguration, Dependency.
- **Application**: Filter by one or more application names.
- **Date range**: Discovered within a time window.
- **Assigned to**: Filter by team member responsible for remediation.

Filters are combinable. URL state is preserved — share filtered views with teammates via the browser URL.

## Pipeline Security Tab

The **Pipeline** tab shows security findings from CI/CD runs:

- **Tool**: The scanner that found the issue (Semgrep, Trivy, Gitleaks, OWASP ZAP, etc.).
- **Stage**: The pipeline stage where it was detected (build, test, deploy).
- **Branch**: The Git branch the pipeline ran on.
- **Commit**: The SHA of the commit that triggered the finding.

Pipeline findings auto-close when the issue is no longer detected in a subsequent scan on the same branch.

## Understanding the Issues Table

Columns in the main issues table:
- **ID**: Unique issue identifier (e.g. ISS-001).
- **Title**: Short description of the vulnerability.
- **CVE**: Linked CVE identifier if applicable.
- **Application**: The service or app where the issue was found.
- **Category**: Vulnerability class.
- **Severity**: Color-coded severity badge.
- **Status**: Current remediation status.
- **Discovered**: Date the issue was first detected.

Click any row to view the full issue detail including description, remediation steps, and history.

## Exporting Data

Export the current filtered view as CSV or PDF via the **Export** button (top right).
Exports respect all active filters.
```

- [ ] **Step 4: Create `.env` template**

```bash
cat > .env << 'EOF'
OPENAI_API_KEY=sk-...
MCP_SERVER_URL=http://localhost:8000/mcp
MCP_AUTH_TOKEN=mock-token
CHROMA_PERSIST_DIR=./chroma_db
DOCS_DIR=./docs
EOF
```

- [ ] **Step 5: Commit**

```bash
git add docs/ tests/__init__.py mock_server/__init__.py rag/__init__.py mcp/__init__.py agent/__init__.py
git commit -m "feat: scaffold project structure and RAG knowledge base docs"
```

---

## Task 2: Mock Server Models + Data

**Files:**
- Create: `mock_server/models.py`, `mock_server/data.py`
- Test: `tests/test_mock_server.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mock_server.py
import pytest
from mock_server.models import SecurityIssue, Application, PipelineIssue, Severity

class TestMockServerData:
    def test_issues_count(self):
        from mock_server.data import MOCK_ISSUES
        assert len(MOCK_ISSUES) >= 8

    def test_applications_count(self):
        from mock_server.data import MOCK_APPLICATIONS
        assert len(MOCK_APPLICATIONS) >= 4

    def test_pipeline_issues_count(self):
        from mock_server.data import MOCK_PIPELINE_ISSUES
        assert len(MOCK_PIPELINE_ISSUES) >= 4

    def test_issue_fields_populated(self):
        from mock_server.data import MOCK_ISSUES
        issue = MOCK_ISSUES[0]
        assert issue.id
        assert issue.title
        assert issue.application
        assert issue.description

    def test_risk_score_range(self):
        from mock_server.data import MOCK_APPLICATIONS
        for app in MOCK_APPLICATIONS:
            assert 0.0 <= app.risk_score <= 10.0

    def test_has_critical_issues(self):
        from mock_server.data import MOCK_ISSUES
        criticals = [i for i in MOCK_ISSUES if i.severity == Severity.CRITICAL]
        assert len(criticals) >= 2

    def test_security_issue_serializable(self):
        from mock_server.data import MOCK_ISSUES
        data = MOCK_ISSUES[0].model_dump()
        assert isinstance(data, dict)
        assert "severity" in data
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_mock_server.py -v
```

Expected: `ModuleNotFoundError: No module named 'mock_server.models'`

- [ ] **Step 3: Write `mock_server/models.py`**

```python
from enum import Enum
from pydantic import BaseModel


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Category(str, Enum):
    INJECTION = "injection"
    XSS = "xss"
    BROKEN_AUTH = "broken_auth"
    EXPOSED_DATA = "exposed_data"
    MISCONFIG = "misconfig"
    DEPENDENCY = "dependency"


class IssueStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"


class SecurityIssue(BaseModel):
    id: str
    title: str
    severity: Severity
    category: Category
    status: IssueStatus
    cve_id: str | None = None
    application: str
    description: str
    discovered_at: str


class Application(BaseModel):
    id: str
    name: str
    risk_score: float
    issue_count: int
    critical_count: int
    last_scan: str


class PipelineIssue(BaseModel):
    id: str
    pipeline: str
    stage: str
    severity: Severity
    title: str
    tool: str
    commit_sha: str
    branch: str
    detected_at: str
```

- [ ] **Step 4: Write `mock_server/data.py`**

```python
from mock_server.models import (
    Application, Category, IssueStatus, PipelineIssue, SecurityIssue, Severity,
)

MOCK_ISSUES: list[SecurityIssue] = [
    SecurityIssue(
        id="ISS-001", title="SQL Injection in user search endpoint",
        severity=Severity.CRITICAL, category=Category.INJECTION,
        status=IssueStatus.OPEN, cve_id="CVE-2024-1234",
        application="user-service",
        description="Unsanitized input in /api/users/search allows attackers to dump the users table.",
        discovered_at="2024-11-01",
    ),
    SecurityIssue(
        id="ISS-002", title="Reflected XSS in error page",
        severity=Severity.HIGH, category=Category.XSS,
        status=IssueStatus.OPEN, cve_id=None,
        application="frontend-app",
        description="The 404 error page reflects unescaped query parameters, enabling script injection.",
        discovered_at="2024-11-05",
    ),
    SecurityIssue(
        id="ISS-003", title="Broken authentication on password reset",
        severity=Severity.CRITICAL, category=Category.BROKEN_AUTH,
        status=IssueStatus.OPEN, cve_id=None,
        application="auth-service",
        description="Password reset tokens do not expire and can be replayed to take over any account.",
        discovered_at="2024-11-08",
    ),
    SecurityIssue(
        id="ISS-004", title="PII exposed in API response",
        severity=Severity.HIGH, category=Category.EXPOSED_DATA,
        status=IssueStatus.IN_PROGRESS, cve_id=None,
        application="api-gateway",
        description="The /api/users endpoint returns full SSN and date-of-birth for all users.",
        discovered_at="2024-11-10",
    ),
    SecurityIssue(
        id="ISS-005", title="Log4Shell vulnerability in payment service",
        severity=Severity.CRITICAL, category=Category.DEPENDENCY,
        status=IssueStatus.OPEN, cve_id="CVE-2021-44228",
        application="payment-service",
        description="log4j 2.14.1 is in use. Remote code execution via JNDI lookup is possible.",
        discovered_at="2024-11-12",
    ),
    SecurityIssue(
        id="ISS-006", title="S3 bucket publicly accessible",
        severity=Severity.MEDIUM, category=Category.MISCONFIG,
        status=IssueStatus.OPEN, cve_id=None,
        application="data-pipeline",
        description="The raw-ingest S3 bucket has a public ACL allowing unauthenticated reads.",
        discovered_at="2024-11-14",
    ),
    SecurityIssue(
        id="ISS-007", title="Command injection in report generator",
        severity=Severity.HIGH, category=Category.INJECTION,
        status=IssueStatus.RESOLVED, cve_id=None,
        application="reporting-service",
        description="Report filenames were passed unescaped to a shell command. Now fixed.",
        discovered_at="2024-10-20",
    ),
    SecurityIssue(
        id="ISS-008", title="Stored XSS in admin notes field",
        severity=Severity.MEDIUM, category=Category.XSS,
        status=IssueStatus.OPEN, cve_id=None,
        application="admin-portal",
        description="Admin notes are rendered without sanitization, allowing stored XSS for any admin viewer.",
        discovered_at="2024-11-16",
    ),
    SecurityIssue(
        id="ISS-009", title="Sensitive PII in application logs",
        severity=Severity.HIGH, category=Category.EXPOSED_DATA,
        status=IssueStatus.IN_PROGRESS, cve_id=None,
        application="user-service",
        description="Full credit card numbers appear in INFO-level logs, persisted to Splunk.",
        discovered_at="2024-11-18",
    ),
    SecurityIssue(
        id="ISS-010", title="Weak session token entropy",
        severity=Severity.MEDIUM, category=Category.BROKEN_AUTH,
        status=IssueStatus.OPEN, cve_id=None,
        application="api-gateway",
        description="Session tokens are generated with 32-bit random seed, making brute-force feasible.",
        discovered_at="2024-11-20",
    ),
]

MOCK_APPLICATIONS: list[Application] = [
    Application(
        id="APP-001", name="user-service", risk_score=8.5,
        issue_count=3, critical_count=1, last_scan="2024-11-20",
    ),
    Application(
        id="APP-002", name="auth-service", risk_score=9.2,
        issue_count=2, critical_count=1, last_scan="2024-11-20",
    ),
    Application(
        id="APP-003", name="payment-service", risk_score=9.8,
        issue_count=1, critical_count=1, last_scan="2024-11-19",
    ),
    Application(
        id="APP-004", name="frontend-app", risk_score=6.3,
        issue_count=2, critical_count=0, last_scan="2024-11-20",
    ),
    Application(
        id="APP-005", name="api-gateway", risk_score=7.1,
        issue_count=2, critical_count=0, last_scan="2024-11-20",
    ),
)

MOCK_PIPELINE_ISSUES: list[PipelineIssue] = [
    PipelineIssue(
        id="PIPE-001", pipeline="payment-service-ci", stage="dependency-scan",
        severity=Severity.CRITICAL, title="CVE-2021-44228 in log4j dependency",
        tool="Trivy", commit_sha="a1b2c3d", branch="main", detected_at="2024-11-19",
    ),
    PipelineIssue(
        id="PIPE-002", pipeline="auth-service-ci", stage="sast",
        severity=Severity.HIGH, title="Hardcoded JWT secret in config loader",
        tool="Semgrep", commit_sha="e4f5g6h", branch="feature/login-refactor",
        detected_at="2024-11-18",
    ),
    PipelineIssue(
        id="PIPE-003", pipeline="api-gateway-ci", stage="secret-scan",
        severity=Severity.HIGH, title="AWS access key committed to repository",
        tool="Gitleaks", commit_sha="i7j8k9l", branch="main", detected_at="2024-11-17",
    ),
    PipelineIssue(
        id="PIPE-004", pipeline="frontend-ci", stage="container-scan",
        severity=Severity.MEDIUM, title="Base image has 3 medium CVEs",
        tool="Trivy", commit_sha="m1n2o3p", branch="develop", detected_at="2024-11-16",
    ),
    PipelineIssue(
        id="PIPE-005", pipeline="user-service-ci", stage="dast",
        severity=Severity.MEDIUM, title="Missing Content-Security-Policy header",
        tool="OWASP ZAP", commit_sha="q4r5s6t", branch="main", detected_at="2024-11-15",
    ),
]
```

- [ ] **Step 5: Run tests — expect pass**

```bash
pytest tests/test_mock_server.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add mock_server/models.py mock_server/data.py tests/test_mock_server.py
git commit -m "feat: add mock server Pydantic models and static security data"
```

---

## Task 3: Mock Server FastMCP App

**Files:**
- Create: `mock_server/main.py`
- Modify: `tests/test_mock_server.py` — add filtering tests

- [ ] **Step 1: Add filtering tests**

```python
# append to class TestMockServerData in tests/test_mock_server.py

    def test_filter_by_severity(self):
        from mock_server.data import MOCK_ISSUES
        from mock_server.models import Severity
        result = [i for i in MOCK_ISSUES if i.severity == Severity.CRITICAL]
        assert all(i.severity == Severity.CRITICAL for i in result)
        assert len(result) >= 2

    def test_filter_by_status(self):
        from mock_server.data import MOCK_ISSUES
        from mock_server.models import IssueStatus
        result = [i for i in MOCK_ISSUES if i.status == IssueStatus.OPEN]
        assert all(i.status == IssueStatus.OPEN for i in result)

    def test_filter_pipeline_by_branch(self):
        from mock_server.data import MOCK_PIPELINE_ISSUES
        result = [i for i in MOCK_PIPELINE_ISSUES if i.branch == "main"]
        assert all(i.branch == "main" for i in result)
        assert len(result) >= 2
```

- [ ] **Step 2: Run new tests — expect fail**

```bash
pytest tests/test_mock_server.py::TestMockServerData::test_filter_by_severity -v
```

Expected: FAIL — `AttributeError` (data not imported yet in test scope)

Actually these tests use data directly so they'll pass once data.py exists. Run all:

```bash
pytest tests/test_mock_server.py -v
```

Expected: all 10 tests PASS (filtering tests pass since they test the data, not the server)

- [ ] **Step 3: Write `mock_server/main.py`**

```python
import logging

from mcp.server.fastmcp import FastMCP

from mock_server.data import MOCK_APPLICATIONS, MOCK_ISSUES, MOCK_PIPELINE_ISSUES

logger = logging.getLogger(__name__)

_mcp = FastMCP("security-platform")


@_mcp.tool()
def get_security_issues(
    severity: str | None = None,
    category: str | None = None,
    status: str | None = None,
) -> list[dict]:
    """Get security issues. Filter by severity (critical/high/medium/low),
    category (injection/xss/broken_auth/exposed_data/misconfig/dependency),
    or status (open/in_progress/resolved)."""
    issues = MOCK_ISSUES
    if severity:
        issues = [i for i in issues if i.severity.value == severity.lower()]
    if category:
        issues = [i for i in issues if i.category.value == category.lower()]
    if status:
        issues = [i for i in issues if i.status.value == status.lower()]
    logger.info("get_security_issues returned %d results", len(issues))
    return [i.model_dump() for i in issues]


@_mcp.tool()
def get_applications(min_risk_score: float | None = None) -> list[dict]:
    """Get applications with their risk scores and issue counts.
    Optionally filter by minimum risk score (0-10)."""
    apps = MOCK_APPLICATIONS
    if min_risk_score is not None:
        apps = [a for a in apps if a.risk_score >= min_risk_score]
    logger.info("get_applications returned %d results", len(apps))
    return [a.model_dump() for a in apps]


@_mcp.tool()
def get_pipeline_issues(
    severity: str | None = None,
    branch: str | None = None,
) -> list[dict]:
    """Get CI/CD pipeline security findings. Filter by severity or branch name."""
    issues = MOCK_PIPELINE_ISSUES
    if severity:
        issues = [i for i in issues if i.severity.value == severity.lower()]
    if branch:
        issues = [i for i in issues if i.branch == branch]
    logger.info("get_pipeline_issues returned %d results", len(issues))
    return [i.model_dump() for i in issues]


# ASGI app for: uvicorn mock_server.main:app --port 8000
# If this raises AttributeError, try: app = _mcp.get_app()
app = _mcp.streamable_http_app(path="/mcp")
```

- [ ] **Step 4: Verify the server starts**

```bash
uvicorn mock_server.main:app --port 8000
```

Expected: server starts, no errors. Stop with Ctrl+C.

- [ ] **Step 5: Add `langchain-mcp-adapters` to requirements**

```bash
echo "langchain-mcp-adapters>=0.1" >> requirements.txt
pip install langchain-mcp-adapters
```

- [ ] **Step 6: Commit**

```bash
git add mock_server/main.py requirements.txt tests/test_mock_server.py
git commit -m "feat: add FastMCP mock security server with 3 tools"
```

---

## Task 4: RAG Indexer

**Files:**
- Create: `rag/indexer.py`
- Test: `tests/test_rag.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rag.py
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestRAGIndexer:
    @pytest.fixture
    def docs_dir(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# Section One\n\nThis explains connector setup.\n\n## Jira\n\nJira integration steps here.")
        return str(tmp_path)

    @pytest.fixture
    def chroma_dir(self, tmp_path):
        return str(tmp_path / "chroma")

    def test_indexer_creates_collection(self, docs_dir, chroma_dir):
        from rag.indexer import RAGIndexer
        with patch("rag.indexer.OpenAIEmbeddings") as mock_emb:
            mock_emb.return_value.embed_documents.return_value = [[0.1] * 10]
            mock_emb.return_value.embed_query.return_value = [0.1] * 10
            indexer = RAGIndexer(docs_dir=docs_dir, persist_dir=chroma_dir)
            indexer.build_index()
            assert indexer.is_indexed()

    def test_indexer_loads_markdown_files(self, docs_dir, chroma_dir):
        from rag.indexer import RAGIndexer
        with patch("rag.indexer.OpenAIEmbeddings"):
            indexer = RAGIndexer(docs_dir=docs_dir, persist_dir=chroma_dir)
            docs = indexer._load_documents()
            assert len(docs) >= 1
            assert any("connector" in d.page_content.lower() for d in docs)
```

- [ ] **Step 2: Run test — expect failure**

```bash
pytest tests/test_rag.py::TestRAGIndexer::test_indexer_loads_markdown_files -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'rag.indexer'`

- [ ] **Step 3: Write `rag/indexer.py`**

```python
import logging
from pathlib import Path

import chromadb
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

_HEADERS_TO_SPLIT = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_COLLECTION_NAME = "security_docs"


class RAGIndexer:
    def __init__(self, docs_dir: str, persist_dir: str) -> None:
        self._docs_dir = Path(docs_dir)
        self._persist_dir = persist_dir
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._client = chromadb.PersistentClient(path=persist_dir)

    def build_index(self) -> None:
        docs = self._load_documents()
        chunks = self._split_documents(docs)
        self._store_chunks(chunks)
        logger.info("Indexed %d chunks from %s", len(chunks), self._docs_dir)

    def is_indexed(self) -> bool:
        try:
            col = self._client.get_collection(_COLLECTION_NAME)
            return col.count() > 0
        except Exception:
            return False

    def _load_documents(self) -> list[Document]:
        docs: list[Document] = []
        for path in self._docs_dir.glob("*.md"):
            loader = TextLoader(str(path))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = path.name
            docs.extend(loaded)
        return docs

    def _split_documents(self, docs: list[Document]) -> list[Document]:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_HEADERS_TO_SPLIT,
            strip_headers=False,
        )
        chunks: list[Document] = []
        for doc in docs:
            splits = splitter.split_text(doc.page_content)
            for chunk in splits:
                chunk.metadata.update(doc.metadata)
            chunks.extend(splits)
        return chunks

    def _store_chunks(self, chunks: list[Document]) -> None:
        try:
            self._client.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass
        collection = self._client.create_collection(_COLLECTION_NAME)
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        embeddings = self._embeddings.embed_documents(texts)
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        collection.add(documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_rag.py::TestRAGIndexer -v
```

Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add rag/indexer.py tests/test_rag.py
git commit -m "feat: add RAG indexer with markdown header splitting and ChromaDB storage"
```

---

## Task 5: RAG Retriever

**Files:**
- Modify: `rag/retriever.py` (create)
- Modify: `tests/test_rag.py` — add retriever tests

- [ ] **Step 1: Add retriever tests**

```python
# append to tests/test_rag.py

class TestRAGRetriever:
    @pytest.fixture
    def mock_collection(self):
        col = MagicMock()
        col.query.return_value = {
            "documents": [["Jira connector setup steps here."]],
            "metadatas": [[{"source": "connectors.md", "h2": "Jira"}]],
            "distances": [[0.12]],
        }
        return col

    def test_retriever_returns_documents(self, mock_collection):
        from rag.retriever import RAGRetriever
        with patch("rag.retriever.OpenAIEmbeddings") as mock_emb, \
             patch("rag.retriever.chromadb.PersistentClient") as mock_client:
            mock_emb.return_value.embed_query.return_value = [0.1] * 10
            mock_client.return_value.get_collection.return_value = mock_collection
            retriever = RAGRetriever(persist_dir="/tmp/fake")
            results = retriever.retrieve("how do I set up Jira?")
            assert len(results) == 1
            assert "Jira" in results[0].page_content

    def test_retriever_includes_source_metadata(self, mock_collection):
        from rag.retriever import RAGRetriever
        with patch("rag.retriever.OpenAIEmbeddings") as mock_emb, \
             patch("rag.retriever.chromadb.PersistentClient") as mock_client:
            mock_emb.return_value.embed_query.return_value = [0.1] * 10
            mock_client.return_value.get_collection.return_value = mock_collection
            retriever = RAGRetriever(persist_dir="/tmp/fake")
            results = retriever.retrieve("Jira setup")
            assert results[0].metadata["source"] == "connectors.md"
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_rag.py::TestRAGRetriever -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'rag.retriever'`

- [ ] **Step 3: Write `rag/retriever.py`**

```python
import logging

import chromadb
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "security_docs"
_DEFAULT_K = 3
_FETCH_K = 10


class RAGRetriever:
    def __init__(self, persist_dir: str, k: int = _DEFAULT_K) -> None:
        self._k = k
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_collection(_COLLECTION_NAME)

    def retrieve(self, query: str) -> list[Document]:
        query_embedding = self._embeddings.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=self._k,
            include=["documents", "metadatas", "distances"],
        )
        return self._to_documents(results)

    def _to_documents(self, results: dict) -> list[Document]:
        docs: list[Document] = []
        for text, metadata in zip(
            results["documents"][0], results["metadatas"][0]
        ):
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    def format_for_prompt(self, docs: list[Document]) -> str:
        parts: list[str] = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            section = doc.metadata.get("h2") or doc.metadata.get("h1", "")
            citation = f"[{source}" + (f" — {section}" if section else "") + "]"
            parts.append(f"{citation}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)
```

- [ ] **Step 4: Run all RAG tests — expect pass**

```bash
pytest tests/test_rag.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add rag/retriever.py tests/test_rag.py
git commit -m "feat: add RAG retriever with ChromaDB MMR-style query and citation formatting"
```

---

## Task 6: MCP Client + LangChain Tools

**Files:**
- Create: `mcp/client.py`, `mcp/tools.py`
- Test: `tests/test_mcp_tools.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mcp_tools.py
import pytest
from unittest.mock import MagicMock, patch


class TestSecurityMCPTools:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.call_tool_sync.return_value = [
            {"id": "ISS-001", "title": "SQL Injection", "severity": "critical",
             "application": "user-service", "status": "open", "description": "...",
             "category": "injection", "discovered_at": "2024-11-01", "cve_id": None}
        ]
        return client

    def test_get_issues_tool_exists(self, mock_client):
        from mcp.tools import SecurityMCPTools
        tools_obj = SecurityMCPTools(client=mock_client)
        tool_names = [t.name for t in tools_obj.as_langchain_tools()]
        assert "get_security_issues" in tool_names

    def test_get_issues_calls_client(self, mock_client):
        from mcp.tools import SecurityMCPTools
        tools_obj = SecurityMCPTools(client=mock_client)
        result = tools_obj.get_security_issues(severity="critical")
        mock_client.call_tool_sync.assert_called_once_with(
            "get_security_issues", {"severity": "critical", "category": None, "status": None}
        )
        assert "SQL Injection" in result

    def test_get_applications_tool_exists(self, mock_client):
        from mcp.tools import SecurityMCPTools
        tools_obj = SecurityMCPTools(client=mock_client)
        tool_names = [t.name for t in tools_obj.as_langchain_tools()]
        assert "get_applications" in tool_names

    def test_tool_returns_string(self, mock_client):
        from mcp.tools import SecurityMCPTools
        tools_obj = SecurityMCPTools(client=mock_client)
        result = tools_obj.get_security_issues()
        assert isinstance(result, str)
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_mcp_tools.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'mcp.client'`

- [ ] **Step 3: Write `mcp/client.py`**

```python
import asyncio
import json
import logging

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self, url: str, token: str) -> None:
        self._url = url
        self._headers = {"Authorization": token}

    async def _call(self, tool_name: str, arguments: dict) -> list:
        async with streamablehttp_client(self._url, headers=self._headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return result.content

    def call_tool_sync(self, tool_name: str, arguments: dict) -> list:
        try:
            return asyncio.run(self._call(tool_name, arguments))
        except Exception:
            logger.exception("MCP tool call failed: %s", tool_name)
            return []
```

- [ ] **Step 4: Write `mcp/tools.py`**

```python
import json
import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from mcp.client import MCPClient

logger = logging.getLogger(__name__)


class _GetIssuesInput(BaseModel):
    severity: str | None = Field(None, description="Filter by severity: critical, high, medium, low")
    category: str | None = Field(None, description="Filter by category: injection, xss, broken_auth, etc.")
    status: str | None = Field(None, description="Filter by status: open, in_progress, resolved")


class _GetApplicationsInput(BaseModel):
    min_risk_score: float | None = Field(None, description="Minimum risk score (0-10)")


class _GetPipelineIssuesInput(BaseModel):
    severity: str | None = Field(None, description="Filter by severity")
    branch: str | None = Field(None, description="Filter by git branch name")


class SecurityMCPTools:
    def __init__(self, client: MCPClient) -> None:
        self._client = client

    def get_security_issues(
        self,
        severity: str | None = None,
        category: str | None = None,
        status: str | None = None,
    ) -> str:
        results = self._client.call_tool_sync(
            "get_security_issues",
            {"severity": severity, "category": category, "status": status},
        )
        if not results:
            return "No security issues found matching the given filters."
        return json.dumps(results, indent=2)

    def get_applications(self, min_risk_score: float | None = None) -> str:
        results = self._client.call_tool_sync(
            "get_applications", {"min_risk_score": min_risk_score}
        )
        if not results:
            return "No applications found."
        return json.dumps(results, indent=2)

    def get_pipeline_issues(
        self,
        severity: str | None = None,
        branch: str | None = None,
    ) -> str:
        results = self._client.call_tool_sync(
            "get_pipeline_issues", {"severity": severity, "branch": branch}
        )
        if not results:
            return "No pipeline issues found."
        return json.dumps(results, indent=2)

    def as_langchain_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.get_security_issues,
                name="get_security_issues",
                description="Fetch security issues from the platform. Filter by severity, category, or status.",
                args_schema=_GetIssuesInput,
            ),
            StructuredTool.from_function(
                func=self.get_applications,
                name="get_applications",
                description="Fetch applications with risk scores and issue counts.",
                args_schema=_GetApplicationsInput,
            ),
            StructuredTool.from_function(
                func=self.get_pipeline_issues,
                name="get_pipeline_issues",
                description="Fetch CI/CD pipeline security findings. Filter by severity or branch.",
                args_schema=_GetPipelineIssuesInput,
            ),
        ]
```

- [ ] **Step 5: Run tests — expect pass**

```bash
pytest tests/test_mcp_tools.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add mcp/client.py mcp/tools.py tests/test_mcp_tools.py
git commit -m "feat: add async MCP client and LangChain tool wrappers"
```

---

## Task 7: Agent State + Classifier Node

**Files:**
- Create: `agent/state.py`, `agent/prompts.py`, `agent/nodes.py` (classifier only)
- Test: `tests/test_agent_nodes.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agent_nodes.py
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage


class TestClassifyQuery:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        return llm

    def test_classify_data_query(self, mock_llm):
        from agent.state import AgentState, QueryClassification
        from agent.nodes import AgentNodes

        classification = QueryClassification(query_type="data", reasoning="asks for issue data")
        mock_llm.with_structured_output.return_value.invoke.return_value = classification

        nodes = AgentNodes(llm=mock_llm, mcp_tools=MagicMock(), retriever=MagicMock())
        state: AgentState = {
            "messages": [HumanMessage("show me critical issues")],
            "query_type": "", "mcp_result": "", "rag_result": "", "final_response": "",
        }
        result = nodes.classify_query(state)
        assert result["query_type"] == "data"

    def test_classify_doc_query(self, mock_llm):
        from agent.state import AgentState, QueryClassification
        from agent.nodes import AgentNodes

        classification = QueryClassification(query_type="doc", reasoning="asks about docs")
        mock_llm.with_structured_output.return_value.invoke.return_value = classification

        nodes = AgentNodes(llm=mock_llm, mcp_tools=MagicMock(), retriever=MagicMock())
        state: AgentState = {
            "messages": [HumanMessage("how do I connect Jira?")],
            "query_type": "", "mcp_result": "", "rag_result": "", "final_response": "",
        }
        result = nodes.classify_query(state)
        assert result["query_type"] == "doc"
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_agent_nodes.py::TestClassifyQuery -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'agent.state'`

- [ ] **Step 3: Write `agent/state.py`**

```python
from typing import Annotated, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict


class QueryClassification(BaseModel):
    query_type: Literal["data", "doc", "mixed"]
    reasoning: str


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query_type: str
    mcp_result: str
    rag_result: str
    final_response: str
```

- [ ] **Step 4: Write `agent/prompts.py`**

```python
from langchain_core.prompts import ChatPromptTemplate

CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query classifier for a security platform assistant.

Classify the user's query into exactly one type:
- "data": User wants live security data (issues, applications, pipeline findings, counts, severities).
- "doc": User wants to know HOW to use the platform (setup, connectors, dashboard, filters).
- "mixed": User wants BOTH data AND documentation (e.g. explain a vulnerability category AND show examples).

Examples:
- "Show me critical issues" → data
- "How do I connect Jira?" → doc
- "What is SQL injection and how many do we have?" → mixed"""),
    ("human", "{query}"),
])

FORMATTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful security platform assistant. Answer the user's question
using only the provided context. Be concise and specific. If showing security issues,
summarize the key findings. If referencing documentation, cite the source.

MCP Data:
{mcp_result}

Documentation:
{rag_result}"""),
    ("human", "{query}"),
])
```

- [ ] **Step 5: Write `agent/nodes.py` (classifier only)**

```python
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from agent.prompts import CLASSIFIER_PROMPT, FORMATTER_PROMPT
from agent.state import AgentState, QueryClassification
from mcp.tools import SecurityMCPTools
from rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)


class AgentNodes:
    def __init__(
        self,
        llm: BaseChatModel,
        mcp_tools: SecurityMCPTools,
        retriever: RAGRetriever,
    ) -> None:
        self._llm = llm
        self._mcp_tools = mcp_tools
        self._retriever = retriever
        self._classifier = CLASSIFIER_PROMPT | llm.with_structured_output(QueryClassification)
        self._formatter = FORMATTER_PROMPT | llm

    def classify_query(self, state: AgentState) -> dict:
        query = state["messages"][-1].content
        try:
            result: QueryClassification = self._classifier.invoke({"query": query})
            logger.info("Classified '%s' as '%s'", query[:50], result.query_type)
            return {"query_type": result.query_type}
        except Exception:
            logger.exception("Classification failed, defaulting to 'mixed'")
            return {"query_type": "mixed"}
```

- [ ] **Step 6: Run tests — expect pass**

```bash
pytest tests/test_agent_nodes.py::TestClassifyQuery -v
```

Expected: both tests PASS

- [ ] **Step 7: Commit**

```bash
git add agent/state.py agent/prompts.py agent/nodes.py tests/test_agent_nodes.py
git commit -m "feat: add agent state, classifier prompt, and classify_query node"
```

---

## Task 8: MCP Node + RAG Node

**Files:**
- Modify: `agent/nodes.py` — add `mcp_node`, `rag_node`
- Modify: `tests/test_agent_nodes.py` — add node tests

- [ ] **Step 1: Add node tests**

```python
# append to tests/test_agent_nodes.py

class TestMCPNode:
    @pytest.fixture
    def nodes(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = MagicMock()
        mcp_tools = MagicMock()
        mcp_tools.get_security_issues.return_value = '[{"id": "ISS-001", "title": "SQL Injection"}]'
        retriever = MagicMock()
        from agent.nodes import AgentNodes
        return AgentNodes(llm=llm, mcp_tools=mcp_tools, retriever=retriever)

    def test_mcp_node_calls_llm_with_tools(self, nodes):
        from agent.state import AgentState
        state: AgentState = {
            "messages": [HumanMessage("show me critical security issues")],
            "query_type": "data", "mcp_result": "", "rag_result": "", "final_response": "",
        }
        with patch.object(nodes._llm, "bind_tools") as mock_bind:
            mock_bind.return_value.invoke.return_value = MagicMock(
                content="Found 1 critical issue",
                tool_calls=[],
            )
            result = nodes.mcp_node(state)
        assert "mcp_result" in result

    def test_rag_node_calls_retriever(self, nodes):
        from agent.state import AgentState
        from langchain_core.documents import Document
        nodes._retriever.retrieve.return_value = [
            Document(page_content="Jira setup steps.", metadata={"source": "connectors.md"})
        ]
        nodes._retriever.format_for_prompt.return_value = "[connectors.md]\nJira setup steps."
        state: AgentState = {
            "messages": [HumanMessage("how do I connect Jira?")],
            "query_type": "doc", "mcp_result": "", "rag_result": "", "final_response": "",
        }
        result = nodes.rag_node(state)
        assert "rag_result" in result
        assert "Jira" in result["rag_result"]
        nodes._retriever.retrieve.assert_called_once()
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_agent_nodes.py::TestMCPNode -v
```

Expected: FAIL — `AttributeError: 'AgentNodes' has no attribute 'mcp_node'`

- [ ] **Step 3: Add `mcp_node` and `rag_node` to `agent/nodes.py`**

Add these methods to the `AgentNodes` class:

```python
    def mcp_node(self, state: AgentState) -> dict:
        query = state["messages"][-1].content
        tools = self._mcp_tools.as_langchain_tools()
        llm_with_tools = self._llm.bind_tools(tools)
        try:
            response = llm_with_tools.invoke(state["messages"])
            if response.tool_calls:
                tool_results = self._execute_tool_calls(response.tool_calls, tools)
                return {"mcp_result": tool_results}
            return {"mcp_result": response.content}
        except Exception:
            logger.exception("MCP node failed for query: %s", query[:50])
            return {"mcp_result": "Error fetching security data. The platform may be unavailable."}

    def rag_node(self, state: AgentState) -> dict:
        query = state["messages"][-1].content
        try:
            docs = self._retriever.retrieve(query)
            if not docs:
                return {"rag_result": "No relevant documentation found."}
            return {"rag_result": self._retriever.format_for_prompt(docs)}
        except Exception:
            logger.exception("RAG node failed for query: %s", query[:50])
            return {"rag_result": "Error retrieving documentation."}

    def _execute_tool_calls(self, tool_calls: list, tools: list) -> str:
        tool_map = {t.name: t for t in tools}
        results: list[str] = []
        for call in tool_calls:
            tool = tool_map.get(call["name"])
            if not tool:
                continue
            try:
                output = tool.invoke(call["args"])
                results.append(f"[{call['name']}]\n{output}")
            except Exception:
                logger.exception("Tool call failed: %s", call["name"])
                results.append(f"[{call['name']}] Error: tool call failed.")
        return "\n\n".join(results)
```

- [ ] **Step 4: Run all agent tests — expect pass**

```bash
pytest tests/test_agent_nodes.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add agent/nodes.py tests/test_agent_nodes.py
git commit -m "feat: add mcp_node and rag_node to agent"
```

---

## Task 9: Format Response Node + Agent Graph

**Files:**
- Modify: `agent/nodes.py` — add `format_response`
- Create: `agent/graph.py`
- Modify: `tests/test_agent_nodes.py` — add routing test

- [ ] **Step 1: Add routing + format tests**

```python
# append to tests/test_agent_nodes.py

class TestFormatResponse:
    @pytest.fixture
    def nodes(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = MagicMock()
        llm.invoke.return_value = MagicMock(content="Here are the critical issues found.")
        from agent.nodes import AgentNodes
        return AgentNodes(llm=llm, mcp_tools=MagicMock(), retriever=MagicMock())

    def test_format_response_sets_final_response(self, nodes):
        from agent.state import AgentState
        state: AgentState = {
            "messages": [HumanMessage("show me critical issues")],
            "query_type": "data",
            "mcp_result": '[{"id": "ISS-001", "severity": "critical"}]',
            "rag_result": "",
            "final_response": "",
        }
        result = nodes.format_response(state)
        assert "final_response" in result
        assert result["final_response"]


class TestGraphRouting:
    def test_graph_compiles(self):
        from agent.graph import GraphBuilder
        from unittest.mock import MagicMock
        builder = GraphBuilder(
            llm=MagicMock(),
            mcp_tools=MagicMock(),
            retriever=MagicMock(),
        )
        app = builder.build()
        assert app is not None
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_agent_nodes.py::TestFormatResponse tests/test_agent_nodes.py::TestGraphRouting -v
```

Expected: FAIL

- [ ] **Step 3: Add `format_response` to `agent/nodes.py`**

Add to `AgentNodes` class:

```python
    def format_response(self, state: AgentState) -> dict:
        query = state["messages"][-1].content
        try:
            response = self._formatter.invoke({
                "query": query,
                "mcp_result": state["mcp_result"] or "N/A",
                "rag_result": state["rag_result"] or "N/A",
            })
            return {"final_response": response.content}
        except Exception:
            logger.exception("Formatter failed")
            return {"final_response": "Sorry, I could not generate a response."}
```

- [ ] **Step 4: Write `agent/graph.py`**

```python
import logging
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import AgentNodes
from agent.state import AgentState
from mcp.tools import SecurityMCPTools
from rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(
        self,
        llm: BaseChatModel,
        mcp_tools: SecurityMCPTools,
        retriever: RAGRetriever,
    ) -> None:
        self._nodes = AgentNodes(llm=llm, mcp_tools=mcp_tools, retriever=retriever)

    def build(self, with_memory: bool = True):
        graph = StateGraph(AgentState)

        graph.add_node("classify_query", self._nodes.classify_query)
        graph.add_node("mcp_node", self._nodes.mcp_node)
        graph.add_node("rag_node", self._nodes.rag_node)
        graph.add_node("format_response", self._nodes.format_response)

        graph.add_edge(START, "classify_query")
        graph.add_conditional_edges(
            "classify_query",
            self._route_after_classify,
            {"mcp_node": "mcp_node", "rag_node": "rag_node"},
        )
        graph.add_conditional_edges(
            "mcp_node",
            self._route_after_mcp,
            {"rag_node": "rag_node", "format_response": "format_response"},
        )
        graph.add_edge("rag_node", "format_response")
        graph.add_edge("format_response", END)

        checkpointer = MemorySaver() if with_memory else None
        return graph.compile(checkpointer=checkpointer)

    def _route_after_classify(
        self, state: AgentState
    ) -> Literal["mcp_node", "rag_node"]:
        if state["query_type"] in ("data", "mixed"):
            return "mcp_node"
        return "rag_node"

    def _route_after_mcp(
        self, state: AgentState
    ) -> Literal["rag_node", "format_response"]:
        if state["query_type"] == "mixed":
            return "rag_node"
        return "format_response"
```

- [ ] **Step 5: Run all tests — expect pass**

```bash
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add agent/nodes.py agent/graph.py tests/test_agent_nodes.py
git commit -m "feat: add format_response node and graph builder with conditional routing"
```

---

## Task 10: Main CLI Entry Point

**Files:**
- Create: `main.py`

- [ ] **Step 1: Write `main.py`**

```python
import logging
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from agent.graph import GraphBuilder
from mcp.client import MCPClient
from mcp.tools import SecurityMCPTools
from rag.indexer import RAGIndexer
from rag.retriever import RAGRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


def _build_agent():
    docs_dir = os.getenv("DOCS_DIR", "./docs")
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
    mcp_token = os.getenv("MCP_AUTH_TOKEN", "mock-token")

    indexer = RAGIndexer(docs_dir=docs_dir, persist_dir=chroma_dir)
    if not indexer.is_indexed():
        print("Building RAG index (first run)...")
        indexer.build_index()
        print("Index ready.\n")

    retriever = RAGRetriever(persist_dir=chroma_dir)
    mcp_client = MCPClient(url=mcp_url, token=mcp_token)
    mcp_tools = SecurityMCPTools(client=mcp_client)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    builder = GraphBuilder(llm=llm, mcp_tools=mcp_tools, retriever=retriever)
    return builder.build(with_memory=True)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    print("Security Platform AI Agent")
    print("Type 'exit' to quit.\n")

    app = _build_agent()
    config = {"configurable": {"thread_id": "session-1"}}

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        try:
            result = app.invoke(
                {"messages": [HumanMessage(user_input)]},
                config=config,
            )
            response = result.get("final_response", "No response generated.")
            print(f"\nAgent: {response}\n")
        except Exception:
            logger.exception("Agent invocation failed")
            print("Agent: Something went wrong. Is the mock server running?\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Start the mock server in a separate terminal**

```bash
uvicorn mock_server.main:app --port 8000
```

- [ ] **Step 3: Run the agent**

```bash
python main.py
```

- [ ] **Step 4: Verify the three sample interactions from the spec**

```
You: Show me critical security issues
You: How do I connect Jira to the platform?
You: What are SQL injection issues and show me the ones we have
```

Expected: agent responds to all three correctly, data query shows issues, doc query cites connectors.md, mixed query does both.

- [ ] **Step 5: Run full test suite one last time**

```bash
pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "feat: add CLI entry point with RAG auto-indexing and conversation memory"
git push origin main
```

---

## BONUS Task 11: Chart Generation

**Files:**
- Create: `agent/charts.py`, `agent/chart_node.py`
- Add to requirements: `matplotlib`

- [ ] **Step 1: Install matplotlib**

```bash
pip install matplotlib
echo "matplotlib>=3.8" >> requirements.txt
```

- [ ] **Step 2: Write `agent/charts.py`**

```python
import io
import logging
from base64 import b64encode

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SecurityCharts:
    def severity_distribution(self, issues: list[dict]) -> str:
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for issue in issues:
            sev = issue.get("severity", "").lower()
            if sev in counts:
                counts[sev] += 1

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#d32f2f", "#f57c00", "#fbc02d", "#388e3c"]
        ax.bar(list(counts.keys()), list(counts.values()), color=colors)
        ax.set_title("Issue Severity Distribution")
        ax.set_ylabel("Count")
        plt.tight_layout()
        return self._save_fig(fig, "severity_distribution.png")

    def top_vulnerable_apps(self, applications: list[dict]) -> str:
        apps = sorted(applications, key=lambda a: a.get("risk_score", 0), reverse=True)[:5]
        names = [a["name"] for a in apps]
        scores = [a["risk_score"] for a in apps]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(names, scores, color="#1565c0")
        ax.set_xlim(0, 10)
        ax.set_xlabel("Risk Score")
        ax.set_title("Top Vulnerable Applications")
        plt.tight_layout()
        return self._save_fig(fig, "top_vulnerable_apps.png")

    def _save_fig(self, fig, filename: str) -> str:
        fig.savefig(filename, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return filename
```

- [ ] **Step 3: Integrate chart generation into `mcp_node` in `agent/nodes.py`**

Add to the `AgentNodes.__init__`:
```python
from agent.charts import SecurityCharts
self._charts = SecurityCharts()
```

Add chart generation at the end of `mcp_node` when the result contains issue or application data:

```python
        # After building mcp_result, check if chart is appropriate
        import json as _json
        try:
            data = _json.loads(mcp_result_str)
            if isinstance(data, list) and data and "severity" in data[0]:
                chart_path = self._charts.severity_distribution(data)
                logger.info("Chart saved to %s", chart_path)
        except Exception:
            pass  # chart generation is best-effort
```

- [ ] **Step 4: Commit**

```bash
git add agent/charts.py requirements.txt
git commit -m "feat(bonus): add chart generation for severity distribution and top apps"
git push origin main
```

---

## Self-Review

### Spec Coverage

| Requirement | Covered by |
|---|---|
| LangGraph agent with query classification | Task 7 — `classify_query` node |
| Route to data (MCP) or documentation (RAG) | Task 9 — `GraphBuilder` conditional edges |
| Conversation memory | Task 9 — `MemorySaver` in `GraphBuilder.build()` |
| Connect to local mock MCP server | Task 3 + Task 6 — `FastMCP` + `MCPClient` |
| Fetch + analyze security issues/apps/pipeline | Task 6 — `SecurityMCPTools` |
| Error handling: server down, empty results | Task 6 — `call_tool_sync` catches exceptions; Task 8 — `mcp_node` catches exceptions |
| RAG: chunk docs/ markdown files | Task 4 — `RAGIndexer._split_documents` with `MarkdownHeaderTextSplitter` |
| Semantic search | Task 5 — `RAGRetriever.retrieve` with ChromaDB cosine similarity |
| Source attribution | Task 5 — `RAGRetriever.format_for_prompt` includes `[source — section]` |
| Working CLI | Task 10 — `main.py` |
| README with setup + architecture | Not in plan — write separately after implementation |
| BONUS charts | Task 11 |

### Notes
- The MCP server `app = _mcp.streamable_http_app(path="/mcp")` API may differ in your installed `mcp` version. If it raises `AttributeError`, try `_mcp.get_app()` or `_mcp.http_app()` and check the FastMCP docs.
- `langchain-mcp-adapters` was added to requirements in Task 3 but is not used — it can be removed, or used to replace the manual `MCPClient` if preferred.
- The agent is sync. If you need async (e.g. for web serving), use `app.ainvoke()` and wrap `main()` with `asyncio.run()`.
