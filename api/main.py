import json as _json
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent.factory import AgentFactory

load_dotenv()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "session-1"


class ChatResponse(BaseModel):
    response: str
    query_type: str
    confidence_score: float = 1.0
    validation_flagged: bool = False
    chart_image: str | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    app.state.agent = AgentFactory.build()
    logger.info("Agent initialized")
    yield


app = FastAPI(title="Security Agent API", lifespan=_lifespan)

_NODE_STATUS: dict[str, str] = {
    "classify_query": "Analyzing your question...",
    "mcp_node": "Fetching security data...",
    "rag_node": "Searching documentation...",
    "format_response": "Generating response...",
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health(request: Request) -> dict:
    if not hasattr(request.app.state, "agent") or request.app.state.agent is None:
        return JSONResponse({"status": "degraded", "reason": "agent not initialized"}, status_code=503)
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    config = {"configurable": {"thread_id": req.thread_id}, "recursion_limit": 15}
    try:
        result = await request.app.state.agent.ainvoke(
            {"messages": [HumanMessage(req.message)]},
            config=config,
        )
        return ChatResponse(
            response=result.get("final_response", "No response generated."),
            query_type=result.get("query_type", "unknown"),
            confidence_score=result.get("validation_score", 1.0),
            validation_flagged=result.get("validation_flagged", False),
            chart_image=result.get("chart_image"),
        )
    except Exception:  # broad catch intentional: API must return 200 with error message rather than 500
        logger.exception("Agent invocation failed")
        return ChatResponse(
            response="Something went wrong. Is the mock server running?",
            query_type="unknown",
        )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    async def event_generator():
        config = {"configurable": {"thread_id": req.thread_id}, "recursion_limit": 15}
        final_state: dict = {}
        try:
            async for event in request.app.state.agent.astream_events(
                {"messages": [HumanMessage(req.message)]},
                config=config,
                version="v2",
            ):
                kind = event["event"]
                # Emit pipeline stage status before first token arrives
                if kind == "on_chain_start":
                    node_name = event.get("name", "")
                    if (
                        node_name in _NODE_STATUS
                        and event.get("metadata", {}).get("langgraph_node") == node_name
                    ):
                        status_payload = _json.dumps({"type": "status", "text": _NODE_STATUS[node_name]})
                        yield f"data: {status_payload}\n\n"
                # Stream formatter tokens only
                elif (
                    kind == "on_chat_model_stream"
                    and event.get("metadata", {}).get("langgraph_node") == "format_response"
                ):
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        payload = _json.dumps({"type": "token", "content": chunk})
                        yield f"data: {payload}\n\n"
                # Capture final graph output
                elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        final_state = output
        except Exception:
            logger.exception("Stream failed")
            error_payload = _json.dumps({"type": "error", "content": "Stream failed."})
            yield f"data: {error_payload}\n\n"
            return
        # Send done event with metadata (only reached when no exception occurred)
        done_payload = _json.dumps({
            "type": "done",
            "query_type": final_state.get("query_type", "unknown"),
            "confidence_score": final_state.get("validation_score", 1.0),
            "validation_flagged": final_state.get("validation_flagged", False),
            "chart_image": final_state.get("chart_image"),
            "final_response": final_state.get("final_response", ""),
        })
        yield f"data: {done_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
