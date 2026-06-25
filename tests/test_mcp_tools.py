import pytest
from unittest.mock import AsyncMock, MagicMock


class TestSecurityMCPTools:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.call_tool = AsyncMock(return_value=[
            {
                "id": "ISS-001",
                "title": "SQL Injection",
                "severity": "critical",
                "application": "user-service",
                "status": "open",
                "description": "...",
                "category": "injection",
                "discovered_at": "2024-11-01",
                "cve_id": None,
            }
        ])
        return client

    def test_tool_names(self, mock_client):
        from mcp_client.tools import SecurityMCPTools

        tool_names = [t.name for t in SecurityMCPTools(client=mock_client).as_langchain_tools()]
        assert tool_names == ["get_security_issues", "get_applications", "get_pipeline_issues"]

    @pytest.mark.asyncio
    async def test_get_security_issues_calls_client(self, mock_client):
        from mcp_client.tools import SecurityMCPTools

        tools_obj = SecurityMCPTools(client=mock_client)
        result = await tools_obj.get_security_issues(severity="critical")
        mock_client.call_tool.assert_called_once_with(
            "get_security_issues",
            {
                "id": None,
                "severity": "critical",
                "category": None,
                "status": None,
                "application": None,
                "keyword": None,
                "cve_id": None,
                "discovered_after": None,
                "discovered_before": None,
                "limit": None,
            },
        )
        assert "SQL Injection" in result

    @pytest.mark.asyncio
    async def test_get_security_issues_returns_string(self, mock_client):
        from mcp_client.tools import SecurityMCPTools

        result = await SecurityMCPTools(client=mock_client).get_security_issues()
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_get_security_issues_empty_returns_message(self, mock_client):
        from mcp_client.tools import SecurityMCPTools

        mock_client.call_tool = AsyncMock(return_value=[])
        result = await SecurityMCPTools(client=mock_client).get_security_issues()
        assert result == "[]"

    @pytest.mark.asyncio
    async def test_get_pipeline_issues_calls_client(self, mock_client):
        from mcp_client.tools import SecurityMCPTools

        mock_client.call_tool = AsyncMock(return_value=[{"id": "PIPE-001", "title": "Log4j"}])
        tools_obj = SecurityMCPTools(client=mock_client)
        result = await tools_obj.get_pipeline_issues(severity="critical")
        mock_client.call_tool.assert_called_once_with(
            "get_pipeline_issues",
            {
                "id": None,
                "severity": "critical",
                "pipeline": None,
                "stage": None,
                "tool": None,
                "branch": None,
                "keyword": None,
                "detected_after": None,
                "detected_before": None,
                "limit": None,
            },
        )
        assert "PIPE-001" in result

    @pytest.mark.asyncio
    async def test_get_applications_calls_client(self, mock_client):
        from mcp_client.tools import SecurityMCPTools

        mock_client.call_tool = AsyncMock(return_value=[{"id": "APP-001", "name": "payment-service", "risk_score": 9.1}])
        result = await SecurityMCPTools(client=mock_client).get_applications(limit=3)
        mock_client.call_tool.assert_called_once_with(
            "get_applications",
            {"min_risk_score": None, "limit": 3},
        )
        assert "payment-service" in result
