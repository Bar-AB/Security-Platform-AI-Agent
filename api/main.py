import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    app.state.agent = AgentFactory.build()
    logger.info("Agent initialized")
    yield


app = FastAPI(title="Security Agent API", lifespan=_lifespan)
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
        )
    except Exception:  # broad catch intentional: API must return 200 with error message rather than 500
        logger.exception("Agent invocation failed")
        return ChatResponse(
            response="Something went wrong. Is the mock server running?",
            query_type="unknown",
        )
