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
