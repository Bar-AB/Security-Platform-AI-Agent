import json
import logging

from langchain_core.language_models import BaseChatModel

from agent.charts import SecurityCharts
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
        self._charts = SecurityCharts()

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
                self._try_generate_chart(tool_results)
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

    def format_response(self, state: AgentState) -> dict:
        query = state["messages"][-1].content
        try:
            response = self._formatter.invoke({
                "query": query,
                "mcp_result": state["mcp_result"] or "N/A",
                "rag_result": state["rag_result"] or "N/A",
            })
            return {"final_response": response.content}
        except Exception:
            logger.exception("Formatter failed")
            return {"final_response": "Sorry, I could not generate a response."}

    def _try_generate_chart(self, mcp_result: str) -> None:
        try:
            for line in mcp_result.splitlines():
                line = line.strip()
                if not line.startswith("["):
                    continue
                data = json.loads(line)
                if not isinstance(data, list) or not data:
                    continue
                if "severity" in data[0]:
                    path = self._charts.severity_distribution(data)
                    logger.info("Chart saved to %s", path)
                    return
                if "risk_score" in data[0]:
                    path = self._charts.top_vulnerable_apps(data)
                    logger.info("Chart saved to %s", path)
                    return
        except Exception:
            pass

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
