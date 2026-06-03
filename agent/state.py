from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import NotRequired, TypedDict


class QueryClassification(BaseModel):
    query_type: Literal["data", "doc", "mixed", "chart"]
    reasoning: str
    docs_query: str
    standalone_query: str


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query_type: str
    docs_query: str
    standalone_query: NotRequired[str]
    mcp_result: str
    rag_result: str
    final_response: str
    wants_chart: NotRequired[bool]
