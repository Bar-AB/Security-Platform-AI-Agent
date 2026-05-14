import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage


class TestClassifyQuery:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        return llm

    def test_classify_data_query(self, mock_llm):
        from agent.state import AgentState, QueryClassification
        from agent.nodes import AgentNodes

        classification = QueryClassification(query_type="data", reasoning="asks for issue data")
        mock_llm.with_structured_output.return_value.invoke.return_value = classification

        nodes = AgentNodes(llm=mock_llm, mcp_tools=MagicMock(), retriever=MagicMock())
        state: AgentState = {
            "messages": [HumanMessage("show me critical issues")],
            "query_type": "", "mcp_result": "", "rag_result": "", "final_response": "",
        }
        result = nodes.classify_query(state)
        assert result["query_type"] == "data"

    def test_classify_doc_query(self, mock_llm):
        from agent.state import AgentState, QueryClassification
        from agent.nodes import AgentNodes

        classification = QueryClassification(query_type="doc", reasoning="asks about docs")
        mock_llm.with_structured_output.return_value.invoke.return_value = classification

        nodes = AgentNodes(llm=mock_llm, mcp_tools=MagicMock(), retriever=MagicMock())
        state: AgentState = {
            "messages": [HumanMessage("how do I connect Jira?")],
            "query_type": "", "mcp_result": "", "rag_result": "", "final_response": "",
        }
        result = nodes.classify_query(state)
        assert result["query_type"] == "doc"


class TestMCPNode:
    @pytest.fixture
    def nodes(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = MagicMock()
        mcp_tools = MagicMock()
        mcp_tools.get_security_issues.return_value = '[{"id": "ISS-001", "title": "SQL Injection"}]'
        retriever = MagicMock()
        from agent.nodes import AgentNodes
        return AgentNodes(llm=llm, mcp_tools=mcp_tools, retriever=retriever)

    def test_mcp_node_returns_mcp_result_key(self, nodes):
        from agent.state import AgentState
        state: AgentState = {
            "messages": [HumanMessage("show me critical security issues")],
            "query_type": "data", "mcp_result": "", "rag_result": "", "final_response": "",
        }
        with patch.object(nodes._llm, "bind_tools") as mock_bind:
            mock_bind.return_value.invoke.return_value = MagicMock(
                content="Found 1 critical issue",
                tool_calls=[],
            )
            result = nodes.mcp_node(state)
        assert "mcp_result" in result

    def test_mcp_node_handles_exception_gracefully(self, nodes):
        from agent.state import AgentState
        state: AgentState = {
            "messages": [HumanMessage("show me issues")],
            "query_type": "data", "mcp_result": "", "rag_result": "", "final_response": "",
        }
        with patch.object(nodes._llm, "bind_tools", side_effect=Exception("LLM error")):
            result = nodes.mcp_node(state)
        assert "mcp_result" in result
        assert "Error" in result["mcp_result"]


class TestRAGNode:
    @pytest.fixture
    def nodes(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = MagicMock()
        retriever = MagicMock()
        from langchain_core.documents import Document
        retriever.retrieve.return_value = [
            Document(page_content="Jira setup steps.", metadata={"source": "connectors.md"})
        ]
        retriever.format_for_prompt.return_value = "[connectors.md]\nJira setup steps."
        from agent.nodes import AgentNodes
        return AgentNodes(llm=llm, mcp_tools=MagicMock(), retriever=retriever)

    def test_rag_node_returns_rag_result_key(self, nodes):
        from agent.state import AgentState
        state: AgentState = {
            "messages": [HumanMessage("how do I connect Jira?")],
            "query_type": "doc", "mcp_result": "", "rag_result": "", "final_response": "",
        }
        result = nodes.rag_node(state)
        assert "rag_result" in result
        assert "Jira" in result["rag_result"]
        nodes._retriever.retrieve.assert_called_once()

    def test_rag_node_handles_empty_results(self, nodes):
        from agent.state import AgentState
        nodes._retriever.retrieve.return_value = []
        state: AgentState = {
            "messages": [HumanMessage("something obscure")],
            "query_type": "doc", "mcp_result": "", "rag_result": "", "final_response": "",
        }
        result = nodes.rag_node(state)
        assert "rag_result" in result
        assert "No relevant" in result["rag_result"]


class TestFormatResponse:
    @pytest.fixture
    def nodes(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = MagicMock()
        llm.invoke.return_value = MagicMock(content="Here are the critical issues found.")
        from agent.nodes import AgentNodes
        return AgentNodes(llm=llm, mcp_tools=MagicMock(), retriever=MagicMock())

    def test_format_response_sets_final_response(self, nodes):
        from agent.state import AgentState
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
        from agent.state import AgentState
        nodes._llm.invoke.side_effect = Exception("LLM failure")
        # Override the formatter chain to fail
        nodes._formatter = MagicMock()
        nodes._formatter.invoke.side_effect = Exception("Chain failure")
        state: AgentState = {
            "messages": [HumanMessage("query")],
            "query_type": "data", "mcp_result": "", "rag_result": "", "final_response": "",
        }
        result = nodes.format_response(state)
        assert "final_response" in result
        assert "Sorry" in result["final_response"]


class TestGraphRouting:
    def test_graph_compiles(self):
        from agent.graph import GraphBuilder
        builder = GraphBuilder(
            llm=MagicMock(),
            mcp_tools=MagicMock(),
            retriever=MagicMock(),
        )
        app = builder.build(with_memory=False)
        assert app is not None

    def test_graph_routes_data_to_mcp(self):
        from agent.graph import GraphBuilder
        from agent.state import AgentState
        builder = GraphBuilder(
            llm=MagicMock(),
            mcp_tools=MagicMock(),
            retriever=MagicMock(),
        )
        state: AgentState = {
            "messages": [], "query_type": "data",
            "mcp_result": "", "rag_result": "", "final_response": "",
        }
        assert builder._route_after_classify(state) == "mcp_node"

    def test_graph_routes_doc_to_rag(self):
        from agent.graph import GraphBuilder
        from agent.state import AgentState
        builder = GraphBuilder(
            llm=MagicMock(),
            mcp_tools=MagicMock(),
            retriever=MagicMock(),
        )
        state: AgentState = {
            "messages": [], "query_type": "doc",
            "mcp_result": "", "rag_result": "", "final_response": "",
        }
        assert builder._route_after_classify(state) == "rag_node"

    def test_graph_routes_mixed_mcp_then_rag(self):
        from agent.graph import GraphBuilder
        from agent.state import AgentState
        builder = GraphBuilder(
            llm=MagicMock(),
            mcp_tools=MagicMock(),
            retriever=MagicMock(),
        )
        state_mixed: AgentState = {
            "messages": [], "query_type": "mixed",
            "mcp_result": "", "rag_result": "", "final_response": "",
        }
        state_after_mcp: AgentState = {
            "messages": [], "query_type": "mixed",
            "mcp_result": "some data", "rag_result": "", "final_response": "",
        }
        assert builder._route_after_classify(state_mixed) == "mcp_node"
        assert builder._route_after_mcp(state_after_mcp) == "rag_node"
