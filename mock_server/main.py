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
app = _mcp.streamable_http_app()
