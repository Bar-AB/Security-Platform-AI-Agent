import pytest
from unittest.mock import MagicMock, patch

import agent.factory as _factory_module
from agent.factory import AgentFactory


class TestAgentFactory:
    def test_build_returns_compiled_graph(self):
        with patch.object(_factory_module, "RAGIndexer") as mock_idx, \
             patch.object(_factory_module, "RAGRetriever"), \
             patch.object(_factory_module, "MCPClient"), \
             patch.object(_factory_module, "SecurityMCPTools"), \
             patch.object(_factory_module, "ChatOpenAI"), \
             patch.object(_factory_module, "GraphBuilder") as mock_builder:
            mock_idx.return_value.is_indexed.return_value = True
            mock_graph = MagicMock()
            mock_builder.return_value.build.return_value = mock_graph
            result = AgentFactory.build()
            assert result is mock_graph

    def test_build_triggers_indexing_when_not_indexed(self):
        with patch.object(_factory_module, "RAGIndexer") as mock_idx, \
             patch.object(_factory_module, "RAGRetriever"), \
             patch.object(_factory_module, "MCPClient"), \
             patch.object(_factory_module, "SecurityMCPTools"), \
             patch.object(_factory_module, "ChatOpenAI"), \
             patch.object(_factory_module, "GraphBuilder"):
            mock_idx.return_value.is_indexed.return_value = False
            AgentFactory.build()
            mock_idx.return_value.build_index.assert_called_once()

    def test_build_raises_when_indexing_fails(self):
        with patch.object(_factory_module, "RAGIndexer") as mock_idx, \
             patch.object(_factory_module, "RAGRetriever"), \
             patch.object(_factory_module, "MCPClient"), \
             patch.object(_factory_module, "SecurityMCPTools"), \
             patch.object(_factory_module, "ChatOpenAI"), \
             patch.object(_factory_module, "GraphBuilder"):
            mock_idx.return_value.is_indexed.return_value = False
            mock_idx.return_value.build_index.side_effect = OSError("disk full")
            with pytest.raises(OSError, match="disk full"):
                AgentFactory.build()
