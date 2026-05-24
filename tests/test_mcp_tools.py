import pytest
from unittest.mock import MagicMock


class TestSecurityMCPTools:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        client.call_tool_sync.return_value = [
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
        ]
        return client

    def test_get_issues_tool_exists(self, mock_client):
        from mcp_client.tools import SecurityMCPTools

        tools_obj = SecurityMCPTools(client=mock_client)
        tool_names = [t.name for t in tools_obj.as_langchain_tools()]
        assert "get_security_issues" in tool_names

    def test_get_issues_calls_client(self, mock_client):
        from mcp_client.tools import SecurityMCPTools

        tools_obj = SecurityMCPTools(client=mock_client)
        result = tools_obj.get_security_issues(severity="critical")
        mock_client.call_tool_sync.assert_called_once_with(
            "get_security_issues",
            {
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

    def test_get_applications_tool_exists(self, mock_client):
        from mcp_client.tools import SecurityMCPTools

        tools_obj = SecurityMCPTools(client=mock_client)
        tool_names = [t.name for t in tools_obj.as_langchain_tools()]
        assert "get_applications" in tool_names

    def test_tool_returns_string(self, mock_client):
        from mcp_client.tools import SecurityMCPTools

        tools_obj = SecurityMCPTools(client=mock_client)
        result = tools_obj.get_security_issues()
        assert isinstance(result, str)
