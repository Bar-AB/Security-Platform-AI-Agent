import logging

from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from mock_server.data import MOCK_APPLICATIONS, MOCK_ISSUES, MOCK_PIPELINE_ISSUES
from mock_server.models import Severity

_SEVERITY_RANK = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
}

logger = logging.getLogger(__name__)

_ISSUES_DESCRIPTION = (
    "Get security issues. Filters:\n"
    "- id: exact issue ID (e.g. 'ISS-001') — returns a single issue\n"
    "- severity: critical, high, medium, low\n"
    "- category: injection, xss, broken_auth, exposed_data, misconfig, dependency\n"
    "- status: open, in_progress, resolved\n"
    "- application: service name (e.g. 'payment-service', 'auth-service', 'user-service')\n"
    "- keyword: case-insensitive substring match on the issue title\n"
    "- cve_id: exact CVE identifier (e.g. 'CVE-2021-44228')\n"
    "- discovered_after: ISO date (YYYY-MM-DD), returns issues discovered on or after this date\n"
    "- discovered_before: ISO date (YYYY-MM-DD), returns issues discovered on or before this date\n"
    "- limit: max number of results to return"
)

_APPLICATIONS_DESCRIPTION = (
    "Get applications sorted by risk score descending (most vulnerable first).\n"
    "Optionally filter by minimum risk score (0-10) and/or cap results with limit."
)

_PIPELINE_DESCRIPTION = (
    "Get CI/CD pipeline security findings. Filters:\n"
    "- id: exact finding ID (e.g. 'PIPE-006') — returns a single finding\n"
    "- severity: critical, high, medium, low\n"
    "- pipeline: CI/CD pipeline name (e.g. 'auth-service-ci', 'payment-service-ci'); substring "
    "match\n"
    "- stage: pipeline stage name (e.g. 'sast', 'dependency-scan', 'secret-scan', "
    "'container-scan', 'dast')\n"
    "- tool: scanner tool name (e.g. 'Trivy', 'Semgrep', 'Gitleaks', 'OWASP ZAP')\n"
    "- branch: git branch name; supports prefix match (e.g. 'feature' matches "
    "'feature/login-refactor')\n"
    "- keyword: case-insensitive substring match on title (e.g. 'AWS', 'log4j', 'secret', 'JWT')\n"
    "- detected_after: ISO date (YYYY-MM-DD), returns findings detected on or after this date\n"
    "- detected_before: ISO date (YYYY-MM-DD), returns findings detected on or before this date\n"
    "- limit: max number of results to return"
)

_mcp = FastMCP("security-platform")


@_mcp.tool(description=_ISSUES_DESCRIPTION)
def get_security_issues(
    id: str | None = None,
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
    issues = MOCK_ISSUES
    if id:
        issues = [i for i in issues if i.id.upper() == id.upper()]
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


@_mcp.tool(description=_APPLICATIONS_DESCRIPTION)
def get_applications(min_risk_score: float | None = None, limit: int | None = None) -> list[dict]:
    apps = MOCK_APPLICATIONS
    if min_risk_score is not None:
        apps = [a for a in apps if a.risk_score >= min_risk_score]
    apps = sorted(apps, key=lambda a: a.risk_score, reverse=True)
    if limit is not None:
        apps = apps[:limit]
    logger.info("get_applications returned %d results", len(apps))
    return [a.model_dump() for a in apps]


@_mcp.tool(description=_PIPELINE_DESCRIPTION)
def get_pipeline_issues(
    id: str | None = None,
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
    issues = MOCK_PIPELINE_ISSUES
    if id:
        issues = [i for i in issues if i.id.upper() == id.upper()]
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
        issues = [
            i
            for i in issues
            if i.branch.lower().startswith(branch_lower) or branch_lower == i.branch.lower()
        ]
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


class _HealthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path != "/health":
            return await call_next(request)
        data_ok = bool(MOCK_ISSUES) and bool(MOCK_APPLICATIONS) and bool(MOCK_PIPELINE_ISSUES)
        if not data_ok:
            return JSONResponse(
                {"status": "degraded", "reason": "mock data not loaded"}, status_code=503
            )
        return JSONResponse(
            {
                "status": "ok",
                "issues": len(MOCK_ISSUES),
                "apps": len(MOCK_APPLICATIONS),
                "pipeline_issues": len(MOCK_PIPELINE_ISSUES),
            }
        )


app = _mcp.streamable_http_app()
app.add_middleware(_HealthMiddleware)
