import asyncio
import json
import logging

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self, url: str, token: str) -> None:
        self._url = url
        self._headers = {"Authorization": f"Bearer {token}"}

    async def _call(self, tool_name: str, arguments: dict) -> list:
        async with streamablehttp_client(self._url, headers=self._headers) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                if not result.content:
                    return []
                # FastMCP returns TextContent objects; extract and parse the JSON text
                first = result.content[0]
                if not hasattr(first, "text"):
                    return []
                try:
                    parsed = json.loads(first.text)
                    return parsed if isinstance(parsed, list) else [parsed]
                except (json.JSONDecodeError, ValueError):
                    return [first.text]

    def call_tool_sync(self, tool_name: str, arguments: dict) -> list:
        try:
            return asyncio.run(self._call(tool_name, arguments))
        except Exception:
            logger.exception("MCP tool call failed: %s", tool_name)
            return []
