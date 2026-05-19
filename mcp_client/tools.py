import json
import logging

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from mcp_client.client import MCPClient

logger = logging.getLogger(__name__)


class _GetIssuesInput(BaseModel):
    severity: str | None = Field(None, description="Filter by severity: critical, high, medium, low")
    category: str | None = Field(None, description="Filter by category: injection, xss, broken_auth, exposed_data, misconfig, dependency")
    status: str | None = Field(None, description="Filter by status: open, in_progress, resolved")
    application: str | None = Field(None, description="Filter by application/service name (e.g. 'payment-service', 'auth-service', 'user-service')")
    keyword: str | None = Field(None, description="Case-insensitive substring match on issue title (e.g. 'SQL', 'JWT', 'log4j', 'S3')")
    cve_id: str | None = Field(None, description="Exact CVE identifier to look up (e.g. 'CVE-2021-44228')")
    discovered_after: str | None = Field(None, description="Return issues discovered on or after this ISO date (YYYY-MM-DD)")
    discovered_before: str | None = Field(None, description="Return issues discovered on or before this ISO date (YYYY-MM-DD). Combine with discovered_after for a date range.")
    limit: int | None = Field(None, description="Maximum number of results to return")


class _GetApplicationsInput(BaseModel):
    min_risk_score: float | None = Field(None, description="Minimum risk score (0-10). Results are always sorted by risk score descending.")
    limit: int | None = Field(None, description="Maximum number of applications to return (use this for 'top N' queries)")


class _GetPipelineIssuesInput(BaseModel):
    severity: str | None = Field(None, description="Filter by severity: critical, high, medium, low")
    pipeline: str | None = Field(None, description="Filter by CI/CD pipeline name (e.g. 'auth-service-ci', 'payment-service-ci'). Use '<service>-ci' pattern for a specific service.")
    stage: str | None = Field(None, description="Filter by pipeline stage (e.g. 'sast', 'dependency-scan', 'secret-scan', 'container-scan', 'dast')")
    tool: str | None = Field(None, description="Filter by scanner tool name (e.g. 'Trivy', 'Semgrep', 'Gitleaks', 'OWASP ZAP')")
    branch: str | None = Field(None, description="Filter by git branch; prefix match supported (e.g. 'feature' matches 'feature/login-refactor')")
    keyword: str | None = Field(None, description="Case-insensitive substring match on finding title (e.g. 'AWS', 'secret', 'log4j', 'JWT')")
    detected_after: str | None = Field(None, description="Return findings detected on or after this ISO date (YYYY-MM-DD)")
    detected_before: str | None = Field(None, description="Return findings detected on or before this ISO date (YYYY-MM-DD). Combine with detected_after for a date range.")
    limit: int | None = Field(None, description="Maximum number of results to return")


class SecurityMCPTools:
    def __init__(self, client: MCPClient) -> None:
        self._client = client

    def get_security_issues(
        self,
        severity: str | None = None,
        category: str | None = None,
        status: str | None = None,
        application: str | None = None,
        keyword: str | None = None,
        cve_id: str | None = None,
        discovered_after: str | None = None,
        discovered_before: str | None = None,
        limit: int | None = None,
    ) -> str:
        results = self._client.call_tool_sync(
            "get_security_issues",
            {
                "severity": severity,
                "category": category,
                "status": status,
                "application": application,
                "keyword": keyword,
                "cve_id": cve_id,
                "discovered_after": discovered_after,
                "discovered_before": discovered_before,
                "limit": limit,
            },
        )
        if not results:
            return "No security issues found matching the given filters."
        return json.dumps(results, indent=2)

    def get_applications(
        self,
        min_risk_score: float | None = None,
        limit: int | None = None,
    ) -> str:
        results = self._client.call_tool_sync(
            "get_applications", {"min_risk_score": min_risk_score, "limit": limit}
        )
        if not results:
            return "No applications found."
        return json.dumps(results, indent=2)

    def get_pipeline_issues(
        self,
        severity: str | None = None,
        pipeline: str | None = None,
        stage: str | None = None,
        tool: str | None = None,
        branch: str | None = None,
        keyword: str | None = None,
        detected_after: str | None = None,
        detected_before: str | None = None,
        limit: int | None = None,
    ) -> str:
        results = self._client.call_tool_sync(
            "get_pipeline_issues",
            {
                "severity": severity,
                "pipeline": pipeline,
                "stage": stage,
                "tool": tool,
                "branch": branch,
                "keyword": keyword,
                "detected_after": detected_after,
                "detected_before": detected_before,
                "limit": limit,
            },
        )
        if not results:
            return "No pipeline issues found matching the given filters."
        return json.dumps(results, indent=2)

    def as_langchain_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.get_security_issues,
                name="get_security_issues",
                description=(
                    "Fetch security issues. Supports filtering by severity, category, status, "
                    "application/service name, keyword (title search), cve_id (exact CVE lookup), "
                    "discovered_after and discovered_before (date range). Use limit for 'top N' queries."
                ),
                args_schema=_GetIssuesInput,
            ),
            StructuredTool.from_function(
                func=self.get_applications,
                name="get_applications",
                description=(
                    "Fetch applications sorted by risk score (highest first). "
                    "Use limit=N for 'top N riskiest/most vulnerable apps' queries. "
                    "Optionally filter by min_risk_score."
                ),
                args_schema=_GetApplicationsInput,
            ),
            StructuredTool.from_function(
                func=self.get_pipeline_issues,
                name="get_pipeline_issues",
                description=(
                    "Fetch CI/CD pipeline security findings. Supports filtering by severity, "
                    "pipeline name (e.g. 'auth-service-ci', 'payment-service-ci'), "
                    "stage (e.g. 'sast', 'dependency-scan', 'secret-scan', 'dast'), "
                    "tool (e.g. 'Trivy', 'Semgrep', 'Gitleaks'), branch (prefix match), "
                    "keyword (title search), detected_after and detected_before (date range). "
                    "Use limit for 'top N' queries."
                ),
                args_schema=_GetPipelineIssuesInput,
            ),
        ]
