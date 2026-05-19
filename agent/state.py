from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict


class QueryClassification(BaseModel):
    query_type: Literal["data", "doc", "mixed"]
    reasoning: str
    docs_query: str  # refined query for RAG; doc/conceptual part only for mixed queries


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query_type: str
    docs_query: str
    mcp_result: str
    rag_result: str
    final_response: str
