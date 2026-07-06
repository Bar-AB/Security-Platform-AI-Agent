import asyncio
import pydantic
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import GraphBuilder
from agent.nodes import AgentNodes
from agent.state import AgentState, GroundednessResult, QueryClassification


class TestAgentStateAndClassification:
    def test_query_classification_accepts_chart_type(self):
        qc = QueryClassification(
            query_type="chart",
            reasoning="follow-up",
            docs_query="",
            standalone_query="show me the chart of the previous results",
        )
        assert qc.query_type == "chart"

    def test_agent_state_accepts_wants_chart(self):
        state: AgentState = {
            "messages": [HumanMessage("test")],
            "query_type": "data",
            "docs_query": "",
            "mcp_result": "",
            "rag_result": "",
            "final_response": "",
            "wants_chart": True,
        }
        assert state["wants_chart"] is True


class TestClassifyQueryChartDetection:
    @pytest.fixture
    def nodes(self):
        llm = MagicMock()
        classification = QueryClassification(
            query_type="data",
            reasoning="data query",
            docs_query="",
            standalone_query="show me critical issues",
        )
        llm.with_structured_output.return_value.invoke.return_value = classification
        return AgentNodes(llm=llm, mcp_tools=MagicMock(), retriever=MagicMock())

    def _make_state(self, query: str):
        return AgentState(
            {
                "messages": [HumanMessage(query)],
                "query_type": "",
                "docs_query": "",
                "mcp_result": "",
                "rag_result": "",
                "final_response": "",
            }
        )

    def test_chart_keyword_sets_wants_chart_true(self, nodes):
        result = nodes.classify_query(
            self._make_state("show me critical issues as a chart")
        )
        assert result["wants_chart"] is True

    def test_graph_keyword_sets_wants_chart_true(self, nodes):
        result = nodes.classify_query(self._make_state("show me the graph"))
        assert result["wants_chart"] is True

    def test_visualize_keyword_sets_wants_chart_true(self, nodes):
        result = nodes.classify_query(
            self._make_state("visualize the severity distribution")
        )
        assert result["wants_chart"] is True

    def test_plot_keyword_sets_wants_chart_true(self, nodes):
        result = nodes.classify_query(self._make_state("plot the results"))
        assert result["wants_chart"] is True

    def test_no_chart_keyword_sets_wants_chart_false(self, nodes):
        result = nodes.classify_query(self._make_state("show me all critical issues"))
        assert result["wants_chart"] is False

    def test_wants_chart_false_on_plain_data_query(self, nodes):
        result = nodes.classify_query(
            self._make_state("how many open issues are there?")
        )
        assert result["wants_chart"] is False

    def test_wants_chart_case_insensitive(self, nodes):
        result = nodes.classify_query(self._make_state("Show me a CHART of severity"))
        assert result["wants_chart"] is True


class TestClassifyQuery:
    @pytest.fixture
    def mock_llm(self):
        return MagicMock()

    def test_classify_data_query(self, mock_llm):
        classification = QueryClassification(
            query_type="data",
            reasoning="asks for issue data",
            docs_query="",
            standalone_query="show me critical issues",
        )
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            classification
        )

        nodes = AgentNodes(llm=mock_llm, mcp_tools=MagicMock(), retriever=MagicMock())
        state: AgentState = {
            "messages": [HumanMessage("show me critical issues")],
            "query_type": "",
            "mcp_result": "",
            "rag_result": "",
            "final_response": "",
        }
        result = nodes.classify_query(state)
        assert result["query_type"] == "data"
        assert result["standalone_query"] == "show me critical issues"

    def test_classify_doc_query(self, mock_llm):
        classification = QueryClassification(
            query_type="doc",
            reasoning="asks about docs",
            docs_query="Jira connector setup",
            standalone_query="how do I connect Jira?",
        )
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            classification
        )

        nodes = AgentNodes(llm=mock_llm, mcp_tools=MagicMock(), retriever=MagicMock())
        state: AgentState = {
            "messages": [HumanMessage("how do I connect Jira?")],
            "query_type": "",
            "mcp_result": "",
            "rag_result": "",
            "final_response": "",
        }
        result = nodes.classify_query(state)
        assert result["query_type"] == "doc"
        assert result["standalone_query"] == "how do I connect Jira?"

    def test_data_query_resets_stale_rag_result(self, mock_llm):
        # Regression: a data follow-up after a doc turn must not inherit the prior rag_result.
        classification = QueryClassification(
            query_type="data",
            reasoning="asks for issue data",
            docs_query="",
            standalone_query="show me the high severity issues",
        )
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            classification
        )

        nodes = AgentNodes(llm=mock_llm, mcp_tools=MagicMock(), retriever=MagicMock())
        state: AgentState = {
            "messages": [HumanMessage("now show me the high severity issues")],
            "query_type": "doc",
            "mcp_result": "N/A",
            "rag_result": "[connectors.md — GitHub Connector]\nstale doc text",
            "final_response": "",
        }
        result = nodes.classify_query(state)
        assert result["rag_result"] == "N/A"
        assert result["mcp_result"] == "N/A"

    def test_chart_query_preserves_prior_mcp_result(self, mock_llm):
        # The "chart" route reuses the previous turn's mcp_result, so classify must not clear it.
        classification = QueryClassification(
            query_type="chart",
            reasoning="wants a chart of prior results",
            docs_query="",
            standalone_query="chart that",
        )
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            classification
        )

        nodes = AgentNodes(llm=mock_llm, mcp_tools=MagicMock(), retriever=MagicMock())
        state: AgentState = {
            "messages": [HumanMessage("now chart that")],
            "query_type": "data",
            "mcp_result": '[get_security_issues]\n[{"id": "ISS-001"}]',
            "rag_result": "N/A",
            "final_response": "",
        }
        result = nodes.classify_query(state)
        assert result["query_type"] == "chart"
        assert "mcp_result" not in result  # untouched → prior turn's data survives
        assert result["rag_result"] == "N/A"


class TestEnforceMultiEntity:
    @pytest.fixture
    def nodes(self):
        return AgentNodes(llm=MagicMock(), mcp_tools=MagicMock(), retriever=MagicMock())

    def test_injects_missing_entity_when_comparison_keyword_present(self, nodes):
        result = nodes._enforce_multi_entity(
            standalone="Compare auth-service vulnerabilities",
            raw_query="compare both",
            new_entities=["auth-service"],
            current_entities=["auth-service", "payment-service"],
        )
        assert "payment-service" in result

    def test_no_injection_when_no_comparison_keyword(self, nodes):
        result = nodes._enforce_multi_entity(
            standalone="Show auth-service vulnerabilities",
            raw_query="show auth-service issues",
            new_entities=["auth-service"],
            current_entities=["auth-service", "payment-service"],
        )
        assert result == "Show auth-service vulnerabilities"

    def test_no_injection_when_entity_already_in_standalone(self, nodes):
        result = nodes._enforce_multi_entity(
            standalone="Compare auth-service and payment-service",
            raw_query="compare both",
            new_entities=["auth-service", "payment-service"],
            current_entities=["auth-service", "payment-service"],
        )
        assert result == "Compare auth-service and payment-service"

    def test_caps_injection_at_two_missing_entities(self, nodes):
        result = nodes._enforce_multi_entity(
            standalone="Compare auth-service issues",
            raw_query="compare all of them",
            new_entities=["auth-service"],
            current_entities=["auth-service", "payment-service", "user-service", "api-gateway"],
        )
        # Only first 2 missing should be injected
        assert "payment-service" in result
        assert "user-service" in result
        assert "api-gateway" not in result

    def test_no_injection_when_current_entities_empty(self, nodes):
        result = nodes._enforce_multi_entity(
            standalone="Compare both",
            raw_query="compare both",
            new_entities=[],
            current_entities=[],
        )
        assert result == "Compare both"

    def test_classify_query_injects_missing_entity_on_compare_both(self, mock_llm=None):
        # End-to-end: classifier collapses "compare both" to one entity; guard must inject the other.
        mock_llm = MagicMock()
        classification = QueryClassification(
            query_type="data",
            reasoning="comparison query",
            docs_query="",
            standalone_query="Compare auth-service vulnerabilities",
            active_entities=["auth-service"],
        )
        mock_llm.with_structured_output.return_value.invoke.return_value = classification
        nodes = AgentNodes(llm=mock_llm, mcp_tools=MagicMock(), retriever=MagicMock())
        state: AgentState = {
            "messages": [HumanMessage("compare both")],
            "query_type": "",
            "mcp_result": "N/A",
            "rag_result": "N/A",
            "final_response": "",
            "active_entities": ["auth-service", "payment-service"],
        }
        result = nodes.classify_query(state)
        assert "payment-service" in result["standalone_query"]


class TestFormatHistory:
    @pytest.fixture
    def nodes(self):
        return AgentNodes(llm=MagicMock(), mcp_tools=MagicMock(), retriever=MagicMock())

    def test_no_prior_messages_returns_placeholder(self, nodes):
        history = nodes._format_history([HumanMessage("only the current message")])
        assert history == "(no prior conversation)"

    def test_includes_prior_user_and_assistant_turns(self, nodes):
        messages = [
            HumanMessage("How do I connect to GitHub?"),
            AIMessage("Use the GitHub connector under Settings."),
            HumanMessage("What are the steps?"),  # current turn, excluded
        ]
        history = nodes._format_history(messages)
        assert "User: How do I connect to GitHub?" in history
        assert "Assistant: Use the GitHub connector under Settings." in history
        # the current (last) message must not be part of the history
        assert "What are the steps?" not in history

    def test_truncates_long_message_content(self, nodes):
        long_answer = "x" * 1000
        messages = [
            HumanMessage("question"),
            AIMessage(long_answer),
            HumanMessage("follow up"),
        ]
        history = nodes._format_history(messages)
        assert "x" * 500 in history
        assert "x" * 501 not in history

    def test_recency_weighted_recent_messages_get_generous_cap(self, nodes):
        # Last recent_n messages should use recent_max_content, not max_content.
        # 6 prior messages + 1 current = 7 total; recent_n=4 → cutoff at index 2.
        # Messages at i<2 (old) get max_content=200; i>=2 (recent) get recent_max_content=1500.
        old_answer = "old" * 400    # 1200 chars — should be cut to 200
        recent_answer = "new" * 600  # 1800 chars — should be cut to 1500
        messages = [
            HumanMessage("old question"),
            AIMessage(old_answer),
            HumanMessage("turn 2 question"),
            AIMessage("turn 2 answer"),
            HumanMessage("recent question"),
            AIMessage(recent_answer),
            HumanMessage("current turn"),  # excluded as current message
        ]
        history = nodes._format_history(
            messages, recent_n=4, recent_max_content=1500, max_content=200
        )
        # Old answer (index 1, beyond recent_n cutoff) capped at 200 chars
        assert "old" * 66 in history       # 198 chars — within 200
        assert "old" * 67 not in history   # 201 chars — over 200
        # Recent answer (index 5, within recent_n) capped at 1500 chars
        assert "new" * 500 in history      # 1500 chars — at limit
        assert "new" * 501 not in history  # 1503 chars — over limit

    def test_recency_weighted_fewer_messages_than_recent_n_all_get_generous_cap(self, nodes):
        # When prior has fewer messages than recent_n, all get recent_max_content.
        long_answer = "z" * 1000
        messages = [
            HumanMessage("question"),
            AIMessage(long_answer),
            HumanMessage("follow up"),
        ]
        history = nodes._format_history(
            messages, recent_n=4, recent_max_content=1500, max_content=200
        )
        assert "z" * 1000 in history  # full content preserved, under 1500

    def test_recency_weighted_zero_recent_n_uses_flat_cap(self, nodes):
        # recent_n=0 must behave exactly like the original flat max_content.
        long_answer = "a" * 1000
        messages = [HumanMessage("q"), AIMessage(long_answer), HumanMessage("follow")]
        history = nodes._format_history(messages, recent_n=0, max_content=300)
        assert "a" * 300 in history
        assert "a" * 301 not in history


class TestMCPNode:
    @pytest.fixture
    def nodes(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = MagicMock()
        mcp_tools = MagicMock()
        mcp_tools.get_security_issues = AsyncMock(
            return_value='[{"id": "ISS-001", "title": "SQL Injection"}]'
        )
        mcp_tools.get_applications = AsyncMock(return_value="[]")
        mcp_tools.get_pipeline_issues = AsyncMock(return_value="[]")
        return AgentNodes(llm=llm, mcp_tools=mcp_tools, retriever=MagicMock())

    def test_mcp_node_returns_mcp_result_key(self, nodes):
        state: AgentState = {
            "messages": [HumanMessage("show me critical security issues")],
            "query_type": "data",
            "mcp_result": "",
            "rag_result": "",
            "final_response": "",
        }
        with patch.object(nodes._llm, "bind_tools") as mock_bind:
            mock_bind.return_value.ainvoke = AsyncMock(
                return_value=MagicMock(
                    content="Found 1 critical issue",
                    tool_calls=[],
                )
            )
            result = asyncio.run(nodes.mcp_node(state))
        assert "mcp_result" in result

    def test_mcp_node_handles_exception_gracefully(self, nodes):
        state: AgentState = {
            "messages": [HumanMessage("show me issues")],
            "query_type": "data",
            "mcp_result": "",
            "rag_result": "",
            "final_response": "",
        }
        with patch.object(nodes._llm, "bind_tools", side_effect=Exception("LLM error")):
            result = asyncio.run(nodes.mcp_node(state))
        assert "mcp_result" in result
        assert "Error" in result["mcp_result"]


class TestRAGNode:
    @pytest.fixture
    def nodes(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = MagicMock()
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            Document(
                page_content="Jira setup steps.", metadata={"source": "connectors.md"}
            )
        ]
        retriever.format_for_prompt.return_value = "[connectors.md]\nJira setup steps."
        return AgentNodes(llm=llm, mcp_tools=MagicMock(), retriever=retriever)

    def test_rag_node_returns_rag_result_key(self, nodes):
        state: AgentState = {
            "messages": [HumanMessage("how do I connect Jira?")],
            "query_type": "doc",
            "mcp_result": "",
            "rag_result": "",
            "final_response": "",
        }
        result = nodes.rag_node(state)
        assert "rag_result" in result
        assert "Jira" in result["rag_result"]
        nodes._retriever.retrieve.assert_called_once()

    def test_rag_node_handles_empty_results(self, nodes):
        nodes._retriever.retrieve.return_value = []
        state: AgentState = {
            "messages": [HumanMessage("something obscure")],
            "query_type": "doc",
            "mcp_result": "",
            "rag_result": "",
            "final_response": "",
        }
        result = nodes.rag_node(state)
        assert "rag_result" in result
        assert "No relevant" in result["rag_result"]


class TestFormatResponse:
    @pytest.fixture
    def nodes(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = MagicMock()
        llm.invoke.return_value = MagicMock(
            content="Here are the critical issues found."
        )
        return AgentNodes(llm=llm, mcp_tools=MagicMock(), retriever=MagicMock())

    def test_format_response_sets_final_response(self, nodes):
        state: AgentState = {
            "messages": [HumanMessage("show me critical issues")],
            "query_type": "data",
            "mcp_result": '[{"id": "ISS-001", "severity": "critical"}]',
            "rag_result": "",
            "final_response": "",
        }
        result = nodes.format_response(state)
        assert "final_response" in result
        assert result["final_response"]

    def test_format_response_handles_exception(self, nodes):
        # Needs both mcp and rag context so neither the no-context guard nor the
        # deterministic MCP shortcut fires — forces the code path through the LLM formatter.
        nodes._formatter = MagicMock()
        nodes._formatter.invoke.side_effect = Exception("Chain failure")
        state: AgentState = {
            "messages": [HumanMessage("query")],
            "query_type": "mixed",
            "mcp_result": '[get_security_issues]\n[{"id": "ISS-001", "severity": "critical"}]',
            "rag_result": "some doc context about connectors",
            "final_response": "",
        }
        result = nodes.format_response(state)
        assert "final_response" in result
        assert "Sorry" in result["final_response"]

    def test_format_response_returns_no_info_when_no_context(self, nodes):
        state: AgentState = {
            "messages": [HumanMessage("How do I configure a Kubernetes load balancer?")],
            "query_type": "doc",
            "mcp_result": "N/A",
            "rag_result": "No relevant documentation found.",
            "final_response": "",
        }
        result = nodes.format_response(state)
        assert "final_response" in result
        assert "don't have information" in result["final_response"]


class TestMCPNodeChartGuard:
    @pytest.fixture
    def nodes(self):
        return AgentNodes(llm=MagicMock(), mcp_tools=MagicMock(), retriever=MagicMock())

    def _make_state(self, wants_chart: bool):
        return AgentState(
            {
                "messages": [HumanMessage("show me issues")],
                "query_type": "data",
                "docs_query": "",
                "mcp_result": "",
                "rag_result": "",
                "final_response": "",
                "wants_chart": wants_chart,
            }
        )

    def test_chart_not_generated_when_wants_chart_false(self, nodes):
        tool_response = MagicMock(
            content="results",
            tool_calls=[MagicMock(name="get_security_issues", args={})],
        )
        nodes._llm.bind_tools.return_value.ainvoke = AsyncMock(return_value=tool_response)
        nodes._mcp_tools.as_langchain_tools.return_value = []

        with (
            patch.object(nodes, "_try_generate_chart") as mock_chart,
            patch.object(nodes, "_execute_tool_calls_async", new=AsyncMock(return_value="[result]")),
        ):
            asyncio.run(nodes.mcp_node(self._make_state(wants_chart=False)))
            mock_chart.assert_not_called()

    def test_chart_generated_when_wants_chart_true(self, nodes):
        tool_response = MagicMock(
            content="results",
            tool_calls=[MagicMock(name="get_security_issues", args={})],
        )
        nodes._llm.bind_tools.return_value.ainvoke = AsyncMock(return_value=tool_response)
        nodes._mcp_tools.as_langchain_tools.return_value = []

        with (
            patch.object(nodes, "_try_generate_chart") as mock_chart,
            patch.object(nodes, "_execute_tool_calls_async", new=AsyncMock(return_value="[result]")),
        ):
            asyncio.run(nodes.mcp_node(self._make_state(wants_chart=True)))
            mock_chart.assert_called_once_with("[result]")


class TestChartNode:
    @pytest.fixture
    def nodes(self):
        return AgentNodes(llm=MagicMock(), mcp_tools=MagicMock(), retriever=MagicMock())

    def _make_state(self, mcp_result: str):
        return AgentState(
            {
                "messages": [HumanMessage("show me on the chart")],
                "query_type": "chart",
                "docs_query": "",
                "mcp_result": mcp_result,
                "rag_result": "",
                "final_response": "",
                "wants_chart": True,
            }
        )

    def test_returns_no_data_message_when_mcp_result_empty(self, nodes):
        result = nodes.chart_node(self._make_state(""))
        assert (
            result["final_response"]
            == "No data available to chart. Please run a data query first."
        )

    def test_returns_no_data_message_when_mcp_result_is_na(self, nodes):
        result = nodes.chart_node(self._make_state("N/A"))
        assert (
            result["final_response"]
            == "No data available to chart. Please run a data query first."
        )

    def test_calls_try_generate_chart_with_mcp_result(self, nodes):
        mcp_result = (
            '[get_security_issues]\n[{"id": "ISS-001", "severity": "critical"}]'
        )
        with (
            patch.object(nodes, "_try_generate_chart") as mock_chart,
            patch.object(
                nodes, "_format_mcp_as_markdown", return_value="**formatted**"
            ),
        ):
            nodes.chart_node(self._make_state(mcp_result))
            mock_chart.assert_called_once_with(mcp_result)

    def test_includes_formatted_text_in_response(self, nodes):
        mcp_result = (
            '[get_security_issues]\n[{"id": "ISS-001", "severity": "critical"}]'
        )
        with (
            patch.object(nodes, "_try_generate_chart"),
            patch.object(
                nodes, "_format_mcp_as_markdown", return_value="**formatted results**"
            ),
        ):
            result = nodes.chart_node(self._make_state(mcp_result))
        assert "Chart generated from previous results." in result["final_response"]
        assert "**formatted results**" in result["final_response"]


class TestValidateResponse:
    @pytest.fixture
    def nodes(self):
        return AgentNodes(llm=MagicMock(), mcp_tools=MagicMock(), retriever=MagicMock())

    def _make_state(
        self,
        final_response: str,
        mcp_result: str = "N/A",
        rag_result: str = "N/A",
        query_type: str = "data",
        messages: list | None = None,
    ):
        # Mirrors real graph state: format_response._emit appends an AIMessage before
        # validate_response runs, so messages[-1] is always an AIMessage here.
        if messages is None:
            messages = [HumanMessage("show me issues"), AIMessage(content=final_response)]
        return AgentState(
            {
                "messages": messages,
                "query_type": query_type,
                "docs_query": "",
                "mcp_result": mcp_result,
                "rag_result": rag_result,
                "final_response": final_response,
            }
        )

    def test_skips_when_both_contexts_na(self, nodes):
        state = self._make_state("Some response.", mcp_result="N/A", rag_result="N/A")
        result = nodes.validate_response(state)
        assert result["validation_score"] == 1.0
        assert result["validation_flagged"] is False
        nodes._llm.with_structured_output.assert_not_called()

    def test_synthesis_validates_against_conversation_history(self, nodes):
        # synthesis_node sets both results to N/A; validator must NOT short-circuit —
        # it should call the LLM using conversation history as the grounding context.
        grounded = GroundednessResult(
            score=0.9, is_grounded=True, flagged_claims=[], reasoning="summary matches history"
        )
        nodes._llm.with_structured_output.return_value.invoke.return_value = grounded
        prior_messages = [
            HumanMessage("Show me critical issues"),
            AIMessage("There is 1 critical issue: ISS-001, SQL Injection in auth-service."),
            HumanMessage("summarize what we found"),
        ]
        synthesis_response = "We found 1 critical SQL Injection issue in auth-service (ISS-001)."
        state = self._make_state(
            final_response=synthesis_response,
            mcp_result="N/A",
            rag_result="N/A",
            query_type="synthesis",
            messages=prior_messages + [AIMessage(content=synthesis_response)],
        )
        result = nodes.validate_response(state)
        nodes._llm.with_structured_output.assert_called_once()
        assert result["validation_score"] == 0.9
        assert result["validation_flagged"] is False

    def test_synthesis_ungrounded_appends_warning(self, nodes):
        # If synthesis hallucinates a fact not in the history, the warning must appear.
        ungrounded = GroundednessResult(
            score=0.3,
            is_grounded=False,
            flagged_claims=["CVSS 9.8 not mentioned in conversation history"],
            reasoning="Score was not present in prior turns",
        )
        nodes._llm.with_structured_output.return_value.invoke.return_value = ungrounded
        prior_messages = [
            HumanMessage("Show me critical issues"),
            AIMessage("There is 1 critical issue: ISS-001, SQL Injection."),
            HumanMessage("what's the cvss score?"),
        ]
        synthesis_response = "The CVSS score is 9.8."
        state = self._make_state(
            final_response=synthesis_response,
            mcp_result="N/A",
            rag_result="N/A",
            query_type="synthesis",
            messages=prior_messages + [AIMessage(content=synthesis_response)],
        )
        result = nodes.validate_response(state)
        assert result["validation_flagged"] is True
        assert "⚠️" in result["final_response"]
        assert "CVSS 9.8 not mentioned in conversation history" in result["final_response"]

    def test_grounded_response_passes_without_modification(self, nodes):
        grounded = GroundednessResult(
            score=0.95, is_grounded=True, flagged_claims=[], reasoning="All claims verified."
        )
        nodes._llm.with_structured_output.return_value.invoke.return_value = grounded
        state = self._make_state(
            final_response="There is 1 critical issue: CVE-2021-44228.",
            mcp_result='[get_security_issues]\n[{"id":"ISS-001","cve":"CVE-2021-44228","severity":"critical"}]',
        )
        result = nodes.validate_response(state)
        assert result["validation_score"] == 0.95
        assert result["validation_flagged"] is False
        assert "final_response" not in result  # response unchanged

    def test_ungrounded_response_appends_warning(self, nodes):
        ungrounded = GroundednessResult(
            score=0.4,
            is_grounded=False,
            flagged_claims=["CVE-2099-99999 was mentioned but not in data"],
            reasoning="Response contains a CVE not present in the context.",
        )
        nodes._llm.with_structured_output.return_value.invoke.return_value = ungrounded
        state = self._make_state(
            final_response="The critical issue CVE-2099-99999 needs patching.",
            mcp_result='[get_security_issues]\n[{"id":"ISS-001","severity":"low"}]',
        )
        result = nodes.validate_response(state)
        assert result["validation_flagged"] is True
        assert result["validation_score"] == 0.4
        assert "⚠️" in result["final_response"]
        assert "40%" in result["final_response"]
        assert "CVE-2099-99999 was mentioned but not in data" in result["final_response"]
        assert result["final_response"].startswith("The critical issue")

    def test_validation_exception_falls_back_to_pass(self, nodes):
        nodes._llm.with_structured_output.return_value.invoke.side_effect = Exception("LLM error")
        state = self._make_state(
            final_response="Some answer.",
            mcp_result='[get_security_issues]\n[]',
        )
        result = nodes.validate_response(state)
        assert result["validation_score"] == 1.0
        assert result["validation_flagged"] is False

    def test_warning_fires_when_score_low_despite_is_grounded_true(self, nodes):
        # Inconsistent LLM output: score says hallucinated, is_grounded says fine.
        # The code must trust score, not is_grounded, to catch this.
        inconsistent = GroundednessResult(
            score=0.2,
            is_grounded=True,
            flagged_claims=["invented claim not in context"],
            reasoning="LLM forgot to set is_grounded=False",
        )
        nodes._llm.with_structured_output.return_value.invoke.return_value = inconsistent
        state = self._make_state(
            final_response="The issue count is 999.",
            mcp_result='[get_security_issues]\n[]',
        )
        result = nodes.validate_response(state)
        assert result["validation_flagged"] is True
        assert "⚠️" in result["final_response"]
        assert result["validation_score"] == 0.2

    def test_flagged_response_updates_messages_history(self, nodes):
        # When validation flags a response, the AIMessage in messages must be updated
        # so conversation history stays consistent with what the user sees.
        ungrounded = GroundednessResult(
            score=0.3, is_grounded=False, flagged_claims=["fake CVE"], reasoning="not in data"
        )
        nodes._llm.with_structured_output.return_value.invoke.return_value = ungrounded
        state = self._make_state(
            final_response="CVE-9999-9999 is critical.",
            mcp_result='[get_security_issues]\n[]',
        )
        original_msg_id = state["messages"][-1].id
        result = nodes.validate_response(state)
        assert "messages" in result
        updated_msgs = result["messages"]
        assert len(updated_msgs) == 1
        assert isinstance(updated_msgs[0], AIMessage)
        assert "⚠️" in updated_msgs[0].content
        # ID must match so LangGraph's add_messages reducer replaces, not appends
        assert updated_msgs[0].id == original_msg_id

    def test_skips_when_final_response_empty(self, nodes):
        # If format_response emitted an empty string, validation should no-op rather than
        # validating empty content against real context.
        state = self._make_state(
            final_response="",
            mcp_result='[get_security_issues]\n[{"id":"ISS-001","severity":"critical"}]',
        )
        state["messages"][-1] = AIMessage(content="")
        result = nodes.validate_response(state)
        assert result["validation_score"] == 1.0
        assert result["validation_flagged"] is False
        nodes._llm.with_structured_output.assert_not_called()


class TestGroundednessResultValidation:
    def test_score_above_one_raises_validation_error(self):
        with pytest.raises(pydantic.ValidationError):
            GroundednessResult(score=1.5, is_grounded=True, flagged_claims=[], reasoning="")

    def test_score_below_zero_raises_validation_error(self):
        with pytest.raises(pydantic.ValidationError):
            GroundednessResult(score=-0.1, is_grounded=False, flagged_claims=[], reasoning="")

    def test_score_at_boundary_values_accepted(self):
        low = GroundednessResult(score=0.0, is_grounded=False, flagged_claims=[], reasoning="r")
        high = GroundednessResult(score=1.0, is_grounded=True, flagged_claims=[], reasoning="r")
        assert low.score == 0.0
        assert high.score == 1.0


class TestGraphRouting:
    def test_graph_compiles(self):
        builder = GraphBuilder(
            llm=MagicMock(),
            mcp_tools=MagicMock(),
            retriever=MagicMock(),
        )
        app = builder.build(with_memory=False)
        assert app is not None

    def test_graph_routes_data_to_mcp(self):
        builder = GraphBuilder(
            llm=MagicMock(),
            mcp_tools=MagicMock(),
            retriever=MagicMock(),
        )
        state: AgentState = {
            "messages": [],
            "query_type": "data",
            "mcp_result": "",
            "rag_result": "",
            "final_response": "",
        }
        assert builder._route_after_classify(state) == "mcp_node"

    def test_graph_routes_doc_to_rag(self):
        builder = GraphBuilder(
            llm=MagicMock(),
            mcp_tools=MagicMock(),
            retriever=MagicMock(),
        )
        state: AgentState = {
            "messages": [],
            "query_type": "doc",
            "mcp_result": "",
            "rag_result": "",
            "final_response": "",
        }
        assert builder._route_after_classify(state) == "rag_node"

    def test_graph_routes_chart_to_chart_node(self):
        builder = GraphBuilder(
            llm=MagicMock(),
            mcp_tools=MagicMock(),
            retriever=MagicMock(),
        )
        state: AgentState = {
            "messages": [],
            "query_type": "chart",
            "docs_query": "",
            "mcp_result": "",
            "rag_result": "",
            "final_response": "",
        }
        assert builder._route_after_classify(state) == "chart_node"

    def test_graph_compiles_with_chart_and_validate_nodes(self):
        builder = GraphBuilder(
            llm=MagicMock(),
            mcp_tools=MagicMock(),
            retriever=MagicMock(),
        )
        app = builder.build(with_memory=False)
        assert app is not None
        nodes_in_graph = app.get_graph().nodes
        assert "chart_node" in nodes_in_graph
        assert "validate_response" in nodes_in_graph

    def test_chart_path_bypasses_validate_response(self):
        # Invoke the chart path end-to-end and confirm validate_response is never called.
        builder = GraphBuilder(llm=MagicMock(), mcp_tools=MagicMock(), retriever=MagicMock())
        validate_mock = MagicMock(return_value={})
        with (
            patch.object(builder._nodes, "classify_query", MagicMock(return_value={
                "query_type": "chart", "docs_query": "", "standalone_query": "show chart",
                "wants_chart": False, "rag_result": "N/A", "mcp_result": "N/A",
            })),
            patch.object(builder._nodes, "chart_node", MagicMock(return_value={
                "final_response": "Chart rendered.",
                "messages": [AIMessage("Chart rendered.")],
            })),
            patch.object(builder._nodes, "validate_response", validate_mock),
        ):
            app = builder.build(with_memory=False)
        state: AgentState = {
            "messages": [HumanMessage("show chart")], "query_type": "",
            "docs_query": "", "mcp_result": "", "rag_result": "", "final_response": "",
        }
        app.invoke(state, config={"recursion_limit": 10})
        validate_mock.assert_not_called()

    def test_format_response_connects_to_validate_response(self):
        # Invoke the data path and confirm validate_response is called exactly once.
        builder = GraphBuilder(llm=MagicMock(), mcp_tools=MagicMock(), retriever=MagicMock())
        validate_mock = MagicMock(return_value={"validation_score": 1.0, "validation_flagged": False})
        with (
            patch.object(builder._nodes, "classify_query", MagicMock(return_value={
                "query_type": "data", "docs_query": "", "standalone_query": "show issues",
                "wants_chart": False, "rag_result": "N/A", "mcp_result": "N/A",
            })),
            patch.object(builder._nodes, "mcp_node", MagicMock(return_value={
                "mcp_result": '[get_security_issues]\n[{"id":"ISS-001"}]',
            })),
            patch.object(builder._nodes, "format_response", MagicMock(return_value={
                "final_response": "Found 1 issue.",
                "messages": [AIMessage("Found 1 issue.")],
            })),
            patch.object(builder._nodes, "validate_response", validate_mock),
        ):
            app = builder.build(with_memory=False)
        state: AgentState = {
            "messages": [HumanMessage("show issues")], "query_type": "",
            "docs_query": "", "mcp_result": "", "rag_result": "", "final_response": "",
        }
        app.invoke(state, config={"recursion_limit": 10})
        validate_mock.assert_called_once()

    def test_graph_routes_mixed_mcp_then_rag(self):
        builder = GraphBuilder(
            llm=MagicMock(),
            mcp_tools=MagicMock(),
            retriever=MagicMock(),
        )
        state_mixed: AgentState = {
            "messages": [],
            "query_type": "mixed",
            "mcp_result": "",
            "rag_result": "",
            "final_response": "",
        }
        assert builder._route_after_classify(state_mixed) == ["mcp_node", "rag_node"]
