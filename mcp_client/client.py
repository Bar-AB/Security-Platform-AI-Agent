import asyncio
import json
import logging

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self, url: str, token: str) -> None:
        self._url = url
        self._headers = {"Authorization": f"Bearer {token}"}

    async def _call(self, tool_name: str, arguments: dict) -> list:
        async with streamable_http_client(self._url, http_client=httpx.AsyncClient(headers=self._headers)) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                if not result.content:
                    return []
                # FastMCP returns one TextContent per item, not a single JSON array
                items: list = []
                for content in result.content:
                    if not hasattr(content, "text"):
                        continue
                    try:
                        parsed = json.loads(content.text)
                        if isinstance(parsed, list):
                            items.extend(parsed)
                        else:
                            items.append(parsed)
                    except (json.JSONDecodeError, ValueError):
                        items.append(content.text)
                return items

    def call_tool_sync(self, tool_name: str, arguments: dict) -> list:
        try:
            return asyncio.run(self._call(tool_name, arguments))
        except Exception:
            logger.exception("MCP tool call failed: %s", tool_name)
            return []
