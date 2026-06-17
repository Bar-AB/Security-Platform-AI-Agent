from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict


class QueryClassification(BaseModel):
    query_type: Literal["data", "doc", "mixed", "chart"]
    reasoning: str
    docs_query: str
    standalone_query: str


class GroundednessResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)  # 0.0 (hallucinated) to 1.0 (fully grounded)
    is_grounded: bool
    flagged_claims: list[str]
    reasoning: str


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query_type: str
    docs_query: str
    standalone_query: NotRequired[str]
    mcp_result: str
    rag_result: str
    final_response: str
    wants_chart: NotRequired[bool]
    validation_score: NotRequired[float]
    validation_flagged: NotRequired[bool]
