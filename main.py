import logging
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent.graph import GraphBuilder
from mcp_client.client import MCPClient
from mcp_client.tools import SecurityMCPTools
from rag.indexer import RAGIndexer
from rag.retriever import RAGRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


def _build_agent():
    docs_dir = os.getenv("DOCS_DIR", "./docs")
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
    mcp_token = os.getenv("MCP_AUTH_TOKEN", "mock-token")

    indexer = RAGIndexer(docs_dir=docs_dir, persist_dir=chroma_dir)
    if not indexer.is_indexed():
        print("Building RAG index (first run)...")
        indexer.build_index()
        print("Index ready.\n")

    retriever = RAGRetriever(persist_dir=chroma_dir)
    mcp_client = MCPClient(url=mcp_url, token=mcp_token)
    mcp_tools = SecurityMCPTools(client=mcp_client)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    builder = GraphBuilder(llm=llm, mcp_tools=mcp_tools, retriever=retriever)
    return builder.build(with_memory=True)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    print("Security Platform AI Agent")
    print("Type 'exit' to quit.\n")

    app = _build_agent()
    config = {"configurable": {"thread_id": "session-1"}}

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        try:
            result = app.invoke(
                {"messages": [HumanMessage(user_input)]},
                config=config,
            )
            response = result.get("final_response", "No response generated.")
            print(f"\nAgent: {response}\n")
        except Exception:
            logger.exception("Agent invocation failed")
            print("Agent: Something went wrong. Is the mock server running?\n")


if __name__ == "__main__":
    main()
