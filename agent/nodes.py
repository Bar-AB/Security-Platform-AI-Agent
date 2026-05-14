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
