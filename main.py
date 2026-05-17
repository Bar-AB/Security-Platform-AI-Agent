import logging
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent.factory import AgentFactory

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    print("Security Platform AI Agent")
    print("Type 'exit' to quit.\n")

    app = AgentFactory.build()
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
        except Exception:  # broad catch intentional: CLI loop must not crash on any agent error
            logger.exception("Agent invocation failed")
            print("Agent: Something went wrong. Is the mock server running?\n")


if __name__ == "__main__":
    main()
