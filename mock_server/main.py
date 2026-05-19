import logging

from mcp.server.fastmcp import FastMCP

from mock_server.data import MOCK_APPLICATIONS, MOCK_ISSUES, MOCK_PIPELINE_ISSUES
from mock_server.models import Severity

_SEVERITY_RANK = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}

logger = logging.getLogger(__name__)

_mcp = FastMCP("security-platform")


@_mcp.tool()
def get_security_issues(
    severity: str | None = None,
    category: str | None = None,
    status: str | None = None,
    application: str | None = None,
    keyword: str | None = None,
    cve_id: str | None = None,
    discovered_after: str | None = None,
    discovered_before: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Get security issues. Filters:
    - severity: critical, high, medium, low
    - category: injection, xss, broken_auth, exposed_data, misconfig, dependency
    - status: open, in_progress, resolved
    - application: service name (e.g. 'payment-service', 'auth-service', 'user-service')
    - keyword: case-insensitive substring match on the issue title
    - cve_id: exact CVE identifier (e.g. 'CVE-2021-44228')
    - discovered_after: ISO date (YYYY-MM-DD), returns issues discovered on or after this date
    - discovered_before: ISO date (YYYY-MM-DD), returns issues discovered on or before this date
    - limit: max number of results to return"""
    issues = MOCK_ISSUES
    if severity:
        issues = [i for i in issues if i.severity.value == severity.lower()]
    if category:
        issues = [i for i in issues if i.category.value == category.lower()]
    if status:
        issues = [i for i in issues if i.status.value == status.lower()]
    if application:
        app_lower = application.lower()
        issues = [i for i in issues if app_lower in i.application.lower()]
    if keyword:
        kw = keyword.lower()
        issues = [i for i in issues if kw in i.title.lower()]
    if cve_id:
        issues = [i for i in issues if i.cve_id and i.cve_id.upper() == cve_id.upper()]
    if discovered_after:
        issues = [i for i in issues if i.discovered_at >= discovered_after]
    if discovered_before:
        issues = [i for i in issues if i.discovered_at <= discovered_before]
    issues = sorted(issues, key=lambda i: _SEVERITY_RANK[i.severity])
    if limit is not None:
        issues = issues[:limit]
    logger.info("get_security_issues returned %d results", len(issues))
    return [i.model_dump() for i in issues]


@_mcp.tool()
def get_applications(min_risk_score: float | None = None, limit: int | None = None) -> list[dict]:
    """Get applications sorted by risk score descending (most vulnerable first).
    Optionally filter by minimum risk score (0-10) and/or cap results with limit."""
    apps = MOCK_APPLICATIONS
    if min_risk_score is not None:
        apps = [a for a in apps if a.risk_score >= min_risk_score]
    apps = sorted(apps, key=lambda a: a.risk_score, reverse=True)
    if limit is not None:
        apps = apps[:limit]
    logger.info("get_applications returned %d results", len(apps))
    return [a.model_dump() for a in apps]


@_mcp.tool()
def get_pipeline_issues(
    severity: str | None = None,
    pipeline: str | None = None,
    stage: str | None = None,
    tool: str | None = None,
    branch: str | None = None,
    keyword: str | None = None,
    detected_after: str | None = None,
    detected_before: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Get CI/CD pipeline security findings. Filters:
    - severity: critical, high, medium, low
    - pipeline: CI/CD pipeline name (e.g. 'auth-service-ci', 'payment-service-ci'); substring match
    - stage: pipeline stage name (e.g. 'sast', 'dependency-scan', 'secret-scan', 'container-scan', 'dast')
    - tool: scanner tool name (e.g. 'Trivy', 'Semgrep', 'Gitleaks', 'OWASP ZAP')
    - branch: git branch name; supports prefix match (e.g. 'feature' matches 'feature/login-refactor')
    - keyword: case-insensitive substring match on title (e.g. 'AWS', 'log4j', 'secret', 'JWT')
    - detected_after: ISO date (YYYY-MM-DD), returns findings detected on or after this date
    - detected_before: ISO date (YYYY-MM-DD), returns findings detected on or before this date
    - limit: max number of results to return"""
    issues = MOCK_PIPELINE_ISSUES
    if severity:
        issues = [i for i in issues if i.severity.value == severity.lower()]
    if pipeline:
        pip_lower = pipeline.lower()
        issues = [i for i in issues if pip_lower in i.pipeline.lower()]
    if stage:
        issues = [i for i in issues if stage.lower() in i.stage.lower()]
    if tool:
        issues = [i for i in issues if tool.lower() in i.tool.lower()]
    if branch:
        branch_lower = branch.lower()
        issues = [i for i in issues if i.branch.lower().startswith(branch_lower) or branch_lower == i.branch.lower()]
    if keyword:
        kw = keyword.lower()
        issues = [i for i in issues if kw in i.title.lower()]
    if detected_after:
        issues = [i for i in issues if i.detected_at >= detected_after]
    if detected_before:
        issues = [i for i in issues if i.detected_at <= detected_before]
    if limit is not None:
        issues = issues[:limit]
    logger.info("get_pipeline_issues returned %d results", len(issues))
    return [i.model_dump() for i in issues]


# ASGI app for: uvicorn mock_server.main:app --port 8000
app = _mcp.streamable_http_app()
