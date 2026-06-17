import logging

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.nodes import AgentNodes
from agent.state import AgentState
from mcp_client.tools import SecurityMCPTools
from rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)

_MAX_STEPS = 15


class GraphBuilder:
    def __init__(
        self,
        llm: BaseChatModel,
        mcp_tools: SecurityMCPTools,
        retriever: RAGRetriever,
    ) -> None:
        self._nodes = AgentNodes(llm=llm, mcp_tools=mcp_tools, retriever=retriever)

    def build(self, with_memory: bool = True) -> CompiledStateGraph:
        graph = StateGraph(AgentState)

        graph.add_node("classify_query", self._nodes.classify_query)
        graph.add_node("mcp_node", self._nodes.mcp_node)
        graph.add_node("rag_node", self._nodes.rag_node)
        graph.add_node("chart_node", self._nodes.chart_node)
        graph.add_node("format_response", self._nodes.format_response)
        graph.add_node("validate_response", self._nodes.validate_response)

        graph.add_edge(START, "classify_query")
        graph.add_conditional_edges("classify_query", self._route_after_classify)
        graph.add_edge("mcp_node", "format_response")
        graph.add_edge("rag_node", "format_response")
        # chart_node renders already-fetched data and makes no new factual claims,
        # so it routes directly to END and intentionally bypasses validate_response.
        graph.add_edge("chart_node", END)
        graph.add_edge("format_response", "validate_response")
        graph.add_edge("validate_response", END)

        checkpointer = MemorySaver() if with_memory else None
        return graph.compile(checkpointer=checkpointer)

    def _route_after_classify(self, state: AgentState) -> str | list[str]:
        qtype = state["query_type"]
        if qtype == "chart":
            return "chart_node"
        if qtype == "data":
            return "mcp_node"
        if qtype == "doc":
            return "rag_node"
        # mixed: run both branches in parallel
        return ["mcp_node", "rag_node"]
