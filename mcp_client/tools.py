import json
import logging

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from mcp_client.client import MCPClient

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
