import pytest
from unittest.mock import MagicMock, patch

from agent.factory import AgentFactory


class TestAgentFactory:
    def test_build_returns_compiled_graph(self):
        with patch("agent.factory.RAGIndexer") as mock_idx, \
             patch("agent.factory.RAGRetriever"), \
             patch("agent.factory.MCPClient"), \
             patch("agent.factory.SecurityMCPTools"), \
             patch("agent.factory.ChatOpenAI"), \
             patch("agent.factory.GraphBuilder") as mock_builder:
            mock_idx.return_value.is_indexed.return_value = True
            mock_graph = MagicMock()
            mock_builder.return_value.build.return_value = mock_graph
            result = AgentFactory.build()
            assert result is mock_graph
