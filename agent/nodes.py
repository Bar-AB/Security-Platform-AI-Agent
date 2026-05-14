import logging

from langchain_core.language_models import BaseChatModel

from agent.prompts import CLASSIFIER_PROMPT, FORMATTER_PROMPT
from agent.state import AgentState, QueryClassification
from mcp_client.tools import SecurityMCPTools
from rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)


class AgentNodes:
    def __init__(
        self,
        llm: BaseChatModel,
        mcp_tools: SecurityMCPTools,
        retriever: RAGRetriever,
    ) -> None:
        self._llm = llm
        self._mcp_tools = mcp_tools
        self._retriever = retriever
        self._formatter = FORMATTER_PROMPT | llm

    def classify_query(self, state: AgentState) -> dict:
        query = state["messages"][-1].content
        try:
            classifier = self._llm.with_structured_output(QueryClassification)
            prompt_value = CLASSIFIER_PROMPT.invoke({"query": query})
            result: QueryClassification = classifier.invoke(prompt_value)
            logger.info("Classified '%s' as '%s'", query[:50], result.query_type)
            return {"query_type": result.query_type}
        except Exception:
            logger.exception("Classification failed, defaulting to 'mixed'")
            return {"query_type": "mixed"}

    def mcp_node(self, state: AgentState) -> dict:
        query = state["messages"][-1].content
        tools = self._mcp_tools.as_langchain_tools()
        try:
            llm_with_tools = self._llm.bind_tools(tools)
            response = llm_with_tools.invoke(state["messages"])
            if response.tool_calls:
                tool_results = self._execute_tool_calls(response.tool_calls, tools)
                return {"mcp_result": tool_results}
            return {"mcp_result": response.content}
        except Exception:
            logger.exception("MCP node failed for query: %s", query[:50])
            return {"mcp_result": "Error fetching security data. The platform may be unavailable."}

    def rag_node(self, state: AgentState) -> dict:
        query = state["messages"][-1].content
        try:
            docs = self._retriever.retrieve(query)
            if not docs:
                return {"rag_result": "No relevant documentation found."}
            return {"rag_result": self._retriever.format_for_prompt(docs)}
        except Exception:
            logger.exception("RAG node failed for query: %s", query[:50])
            return {"rag_result": "Error retrieving documentation."}

    def _execute_tool_calls(self, tool_calls: list, tools: list) -> str:
        tool_map = {t.name: t for t in tools}
        results: list[str] = []
        for call in tool_calls:
            tool = tool_map.get(call["name"])
            if not tool:
                continue
            try:
                output = tool.invoke(call["args"])
                results.append(f"[{call['name']}]\n{output}")
            except Exception:
                logger.exception("Tool call failed: %s", call["name"])
                results.append(f"[{call['name']}] Error: tool call failed.")
        return "\n\n".join(results)
