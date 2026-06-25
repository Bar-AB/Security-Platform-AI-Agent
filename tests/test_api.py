import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from agent.factory import AgentFactory
from api.main import app


class TestChatEndpoint:
    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.ainvoke = AsyncMock(
            return_value={
                "final_response": "Found 2 critical issues.",
                "query_type": "data",
            }
        )
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
        _, kwargs = mock_agent.ainvoke.call_args
        assert kwargs["config"]["configurable"]["thread_id"] == "t-abc"

    def test_chat_returns_error_on_agent_failure(self, client, mock_agent):
        mock_agent.ainvoke = AsyncMock(side_effect=RuntimeError("agent crashed"))
        resp = client.post("/chat", json={"message": "test"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["query_type"] == "unknown"
        assert "Something went wrong" in body["response"]

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestChatStreamEndpoint:
    """Tests for the /chat/stream SSE endpoint."""

    def _make_stream_event(self, kind: str, **kwargs) -> dict:
        """Build a minimal LangGraph astream_events event dict."""
        return {"event": kind, "metadata": {}, "data": {}, "name": "", **kwargs}

    def _token_event(self, text: str, node: str = "format_response") -> dict:
        chunk = MagicMock()
        chunk.content = text
        return {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": node},
            "data": {"chunk": chunk},
            "name": "",
        }

    def _chain_end_event(self, output: dict) -> dict:
        return {
            "event": "on_chain_end",
            "name": "LangGraph",
            "metadata": {},
            "data": {"output": output},
        }

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.ainvoke = AsyncMock(return_value={"final_response": "ok", "query_type": "data"})
        return agent

    @pytest.fixture
    def client(self, mock_agent):
        with patch.object(AgentFactory, "build", return_value=mock_agent):
            with TestClient(app) as c:
                yield c

    def _setup_stream(self, mock_agent, events: list) -> None:
        async def _async_gen(*args, **kwargs):
            for ev in events:
                yield ev

        mock_agent.astream_events = _async_gen

    def _parse_sse(self, text: str) -> list[dict]:
        """Parse SSE body into a list of JSON event dicts."""
        events = []
        for line in text.splitlines():
            if line.startswith("data: "):
                raw = line[6:].strip()
                if raw:
                    events.append(json.loads(raw))
        return events

    def test_stream_returns_200_text_event_stream(self, client, mock_agent):
        self._setup_stream(mock_agent, [])
        resp = client.post("/chat/stream", json={"message": "hello"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_emits_done_event(self, client, mock_agent):
        self._setup_stream(mock_agent, [])
        resp = client.post("/chat/stream", json={"message": "hello"})
        events = self._parse_sse(resp.text)
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1

    def test_stream_done_event_has_expected_fields(self, client, mock_agent):
        chain_end = self._chain_end_event(
            {"query_type": "data", "validation_score": 0.9, "validation_flagged": False}
        )
        self._setup_stream(mock_agent, [chain_end])
        resp = client.post("/chat/stream", json={"message": "hello"})
        events = self._parse_sse(resp.text)
        done = next(e for e in events if e.get("type") == "done")
        assert done["query_type"] == "data"
        assert done["confidence_score"] == 0.9
        assert done["validation_flagged"] is False
        assert "chart_image" in done

    def test_stream_emits_token_events_from_format_response_node(self, client, mock_agent):
        events = [
            self._token_event("Hello "),
            self._token_event("world"),
        ]
        self._setup_stream(mock_agent, events)
        resp = client.post("/chat/stream", json={"message": "hi"})
        parsed = self._parse_sse(resp.text)
        token_events = [e for e in parsed if e.get("type") == "token"]
        assert len(token_events) == 2
        assert token_events[0]["content"] == "Hello "
        assert token_events[1]["content"] == "world"

    def test_stream_ignores_tokens_from_other_nodes(self, client, mock_agent):
        events = [
            self._token_event("classifier token", node="classify_query"),
            self._token_event("formatter token", node="format_response"),
        ]
        self._setup_stream(mock_agent, events)
        resp = client.post("/chat/stream", json={"message": "hi"})
        parsed = self._parse_sse(resp.text)
        token_events = [e for e in parsed if e.get("type") == "token"]
        assert len(token_events) == 1
        assert token_events[0]["content"] == "formatter token"

    def test_stream_error_event_on_agent_exception(self, client, mock_agent):
        async def _boom(*args, **kwargs):
            raise RuntimeError("agent exploded")
            yield  # make it an async generator

        mock_agent.astream_events = _boom
        resp = client.post("/chat/stream", json={"message": "crash"})
        assert resp.status_code == 200
        parsed = self._parse_sse(resp.text)
        error_events = [e for e in parsed if e.get("type") == "error"]
        assert len(error_events) == 1
        assert error_events[0]["content"] == "Stream failed."

    def test_stream_uses_thread_id_from_request(self, client, mock_agent):
        captured_config = {}

        async def _capture(*args, **kwargs):
            captured_config.update(kwargs.get("config", {}))
            return
            yield  # make it an async generator

        mock_agent.astream_events = _capture
        client.post("/chat/stream", json={"message": "test", "thread_id": "thread-xyz"})
        assert captured_config.get("configurable", {}).get("thread_id") == "thread-xyz"

    def test_stream_no_cache_headers(self, client, mock_agent):
        self._setup_stream(mock_agent, [])
        resp = client.post("/chat/stream", json={"message": "hello"})
        assert resp.headers.get("cache-control") == "no-cache"
        assert resp.headers.get("x-accel-buffering") == "no"
