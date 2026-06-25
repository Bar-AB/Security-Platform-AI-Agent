import json
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage


class TestClassifyQuery:
    @pytest.fixture
    def llm(self):
        return MagicMock()

    @pytest.fixture
    def nodes(self, llm):
        from agent.nodes import AgentNodes
        return AgentNodes(
            llm=llm,
            mcp_tools=MagicMock(),
            retriever=MagicMock(),
        )

    def _state(self, query: str) -> dict:
        return {"messages": [HumanMessage(content=query)]}

    def test_data_query_routes_to_data(self, nodes, llm):
        from agent.state import QueryClassification
        query = "show critical issues"
        llm.with_structured_output.return_value.invoke.return_value = QueryClassification(
            query_type="data",
            reasoning="live data needed",
            docs_query=query,
            standalone_query=query,
        )
        result = nodes.classify_query(self._state(query))
        assert result["query_type"] == "data"

    def test_doc_query_routes_to_doc(self, nodes, llm):
        from agent.state import QueryClassification
        query = "How do I connect to GitHub?"
        llm.with_structured_output.return_value.invoke.return_value = QueryClassification(
            query_type="doc",
            reasoning="platform how-to",
            docs_query="GitHub connector setup",
            standalone_query=query,
        )
        result = nodes.classify_query(self._state(query))
        assert result["query_type"] == "doc"

    def test_mixed_query_routes_to_mixed(self, nodes, llm):
        from agent.state import QueryClassification
        query = "What is SQL injection and how many do we have?"
        llm.with_structured_output.return_value.invoke.return_value = QueryClassification(
            query_type="mixed",
            reasoning="needs data and docs",
            docs_query="SQL injection explanation",
            standalone_query=query,
        )
        result = nodes.classify_query(self._state(query))
        assert result["query_type"] == "mixed"

    def test_classifier_fallback_on_exception(self, nodes, llm):
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM down")
        result = nodes.classify_query(self._state("anything"))
        assert result["query_type"] == "mixed"

    def test_standalone_query_stored(self, nodes, llm):
        from agent.state import QueryClassification
        query = "and the high ones?"
        resolved = "Show me the high severity issues"
        llm.with_structured_output.return_value.invoke.return_value = QueryClassification(
            query_type="data",
            reasoning="follow-up",
            docs_query=resolved,
            standalone_query=resolved,
        )
        result = nodes.classify_query(self._state(query))
        assert result["standalone_query"] == resolved

    def test_chart_override_when_data_entities_present(self, nodes, llm):
        from agent.state import QueryClassification
        # LLM returns "chart" but query contains data entities → override to "data"
        query = "show me the issues as a chart"
        llm.with_structured_output.return_value.invoke.return_value = QueryClassification(
            query_type="chart",
            reasoning="visualize",
            docs_query=query,
            standalone_query=query,
        )
        result = nodes.classify_query(self._state(query))
        assert result["query_type"] == "data"


class TestValidateResponse:
    @pytest.fixture
    def llm(self):
        return MagicMock()

    @pytest.fixture
    def nodes(self, llm):
        from agent.nodes import AgentNodes
        return AgentNodes(
            llm=llm,
            mcp_tools=MagicMock(),
            retriever=MagicMock(),
        )

    def _state(self, response: str, mcp: str = "N/A", rag: str = "N/A") -> dict:
        return {
            "final_response": response,
            "mcp_result": mcp,
            "rag_result": rag,
            "messages": [AIMessage(content=response, id="msg-1")],
        }

    def test_skips_when_no_context(self, nodes):
        state = self._state("Some response")
        result = nodes.validate_response(state)
        assert result["validation_score"] == 1.0
        assert result["validation_flagged"] is False

    def test_passes_when_score_above_threshold(self, nodes, llm):
        from agent.state import GroundednessResult
        llm.with_structured_output.return_value.invoke.return_value = GroundednessResult(
            score=0.95,
            is_grounded=True,
            flagged_claims=[],
            reasoning="all claims supported",
        )
        state = self._state("There are 3 critical issues.", mcp="[data with 3 critical issues]")
        result = nodes.validate_response(state)
        assert result["validation_flagged"] is False
        assert result["validation_score"] == pytest.approx(0.95)

    def test_flags_when_score_below_threshold(self, nodes, llm):
        from agent.state import GroundednessResult
        llm.with_structured_output.return_value.invoke.return_value = GroundednessResult(
            score=0.4,
            is_grounded=False,
            flagged_claims=["risk score 9.9 not in context"],
            reasoning="score invented",
        )
        state = self._state("The risk score is 9.9.", mcp="[some mcp data]")
        result = nodes.validate_response(state)
        assert result["validation_flagged"] is True
        assert "Validation warning" in result["final_response"]

    def test_flags_when_is_grounded_false_despite_score(self, nodes, llm):
        from agent.state import GroundednessResult
        # Score says OK but is_grounded=False — should still flag
        llm.with_structured_output.return_value.invoke.return_value = GroundednessResult(
            score=0.8,
            is_grounded=False,
            flagged_claims=["invented claim"],
            reasoning="inconsistent",
        )
        state = self._state("Some claim.", mcp="some data")
        result = nodes.validate_response(state)
        assert result["validation_flagged"] is True

    def test_skips_gracefully_on_llm_failure(self, nodes, llm):
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("timeout")
        state = self._state("Some response.", mcp="some data")
        result = nodes.validate_response(state)
        assert result["validation_score"] == 1.0
        assert result["validation_flagged"] is False


class TestCountByField:
    @pytest.fixture
    def nodes(self):
        from agent.nodes import AgentNodes
        return AgentNodes(llm=MagicMock(), mcp_tools=MagicMock(), retriever=MagicMock())

    def _make_mcp_result(self, issues: list[dict], pipeline: list[dict]) -> str:
        iss_json = json.dumps(issues, indent=2)
        pipe_json = json.dumps(pipeline, indent=2)
        return f"[get_security_issues]\n{iss_json}\n\n[get_pipeline_issues]\n{pipe_json}"

    def test_counts_by_severity_field(self, nodes):
        issues = [
            {"id": "ISS-001", "title": "SQL Injection", "severity": "critical"},
            {"id": "ISS-002", "title": "XSS", "severity": "high"},
            {"id": "ISS-007", "title": "Command injection", "severity": "high", "status": "resolved"},
        ]
        pipeline = [
            {"id": "PIPE-001", "title": "Log4j", "severity": "critical"},
            {"id": "PIPE-002", "title": "JWT", "severity": "high"},
        ]
        result = nodes._count_by_field(self._make_mcp_result(issues, pipeline), "severity")
        assert result is not None
        assert "2 critical" in result
        assert "3 high" in result
        assert "5 issues total" in result

    def test_counts_by_application_field(self, nodes):
        issues = [
            {"id": "ISS-001", "title": "SQL Injection", "application": "user-service"},
            {"id": "ISS-002", "title": "XSS", "application": "payment-service"},
            {"id": "ISS-003", "title": "Auth bypass", "application": "user-service"},
        ]
        result = nodes._count_by_field(self._make_mcp_result(issues, []), "application")
        assert result is not None
        assert "2 user-service" in result
        assert "1 payment-service" in result
        assert "3 issues total" in result

    def test_severity_uses_natural_ordering(self, nodes):
        issues = [
            {"id": "ISS-001", "title": "A", "severity": "low"},
            {"id": "ISS-002", "title": "B", "severity": "critical"},
            {"id": "ISS-003", "title": "C", "severity": "medium"},
        ]
        result = nodes._count_by_field(self._make_mcp_result(issues, []), "severity")
        assert result is not None
        critical_pos = result.index("critical")
        medium_pos = result.index("medium")
        low_pos = result.index("low")
        assert critical_pos < medium_pos < low_pos

    def test_includes_resolved_issues(self, nodes):
        issues = [{"id": "ISS-007", "title": "Resolved issue", "severity": "high", "status": "resolved"}]
        result = nodes._count_by_field(self._make_mcp_result(issues, []), "severity")
        assert result is not None
        assert "1 high" in result
        assert "ISS-007" in result

    def test_returns_none_on_invalid_json(self, nodes):
        result = nodes._count_by_field("[get_security_issues]\nnot-json", "severity")
        assert result is None

    def test_returns_none_on_empty_data(self, nodes):
        result = nodes._count_by_field(
            "[get_security_issues]\n[]\n\n[get_pipeline_issues]\n[]", "severity"
        )
        assert result is None


class TestDetectGroupBy:
    @pytest.fixture
    def nodes(self):
        from agent.nodes import AgentNodes
        return AgentNodes(llm=MagicMock(), mcp_tools=MagicMock(), retriever=MagicMock())

    def test_detects_by_severity(self, nodes):
        assert nodes._detect_group_by("how many issues by severity?") == "severity"

    def test_detects_per_severity(self, nodes):
        assert nodes._detect_group_by("count per severity") == "severity"

    def test_detects_severity_breakdown(self, nodes):
        assert nodes._detect_group_by("give me a severity breakdown") == "severity"

    def test_detects_by_application(self, nodes):
        assert nodes._detect_group_by("how many issues by application?") == "application"

    def test_detects_by_service(self, nodes):
        assert nodes._detect_group_by("issues per service") == "application"

    def test_detects_by_category(self, nodes):
        assert nodes._detect_group_by("show category breakdown") == "category"

    def test_detects_by_status(self, nodes):
        assert nodes._detect_group_by("count per status") == "status"

    def test_returns_none_for_non_groupby_query(self, nodes):
        assert nodes._detect_group_by("show me critical issues") is None

    def test_returns_none_for_generic_count(self, nodes):
        assert nodes._detect_group_by("how many issues are there?") is None
