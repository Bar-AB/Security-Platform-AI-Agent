import logging
import os

from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from agent.graph import GraphBuilder
from mcp_client.client import MCPClient
from mcp_client.tools import SecurityMCPTools
from rag.indexer import RAGIndexer
from rag.retriever import MultiQueryRAGRetriever

logger = logging.getLogger(__name__)


class AgentFactory:
    @staticmethod
    def build() -> CompiledStateGraph:
        docs_dir = os.getenv("DOCS_DIR", "./docs")
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
        mcp_token = os.environ.get("MCP_AUTH_TOKEN")
        if mcp_token is None:
            raise ValueError(
                "MCP_AUTH_TOKEN is not set. "
                "For local development, add MCP_AUTH_TOKEN=mock-token to your .env file."
            )
        model = os.getenv("OPENAI_MODEL", "gpt-4o")

        indexer = RAGIndexer(docs_dir=docs_dir, persist_dir=chroma_dir)
        if not indexer.is_indexed():
            logger.info("Building RAG index...")
            try:
                indexer.build_index()
            except (OSError, RuntimeError):
                logger.exception("Failed to build RAG index from %s", docs_dir)
                raise

        top_k = int(os.getenv("RAG_TOP_K", "5"))
        distance_threshold = float(os.getenv("RAG_DISTANCE_THRESHOLD", "0.5"))
        mcp_client = MCPClient(url=mcp_url, token=mcp_token)
        mcp_tools = SecurityMCPTools(client=mcp_client)
        llm = ChatOpenAI(model=model, temperature=0, max_retries=3)
        retriever = MultiQueryRAGRetriever(
            persist_dir=chroma_dir,
            llm=llm,
            k=top_k,
            distance_threshold=distance_threshold,
        )

        builder = GraphBuilder(llm=llm, mcp_tools=mcp_tools, retriever=retriever)
        return builder.build(with_memory=True)
