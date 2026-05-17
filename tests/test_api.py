import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from agent.factory import AgentFactory
from api.main import app


class TestChatEndpoint:
    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.invoke.return_value = {
            "final_response": "Found 2 critical issues.",
            "query_type": "data",
        }
        return agent

    @pytest.fixture
    def client(self, mock_agent):
        with patch.object(AgentFactory, "build", return_value=mock_agent):
            with TestClient(app) as c:
                yield c

    def test_chat_returns_response(self, client):
        resp = client.post("/chat", json={"message": "show critical issues"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["response"] == "Found 2 critical issues."
        assert body["query_type"] == "data"

    def test_chat_passes_thread_id(self, client, mock_agent):
        client.post("/chat", json={"message": "test", "thread_id": "t-abc"})
        _, kwargs = mock_agent.invoke.call_args
        assert kwargs["config"]["configurable"]["thread_id"] == "t-abc"

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
