import logging
from unittest.mock import MagicMock, patch

import pytest

from rag.indexer import RAGIndexer
from rag.retriever import RAGRetriever


class TestRAGIndexer:
    @pytest.fixture
    def docs_dir(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text(
            "# Section One\n\nThis explains connector setup.\n\n## Jira\n\nJira integration steps "
            "here."
        )
        return str(tmp_path)

    @pytest.fixture
    def chroma_dir(self, tmp_path):
        return str(tmp_path / "chroma")

    def test_indexer_creates_collection(self, docs_dir, chroma_dir):
        with patch("rag.indexer.OpenAIEmbeddings") as mock_emb:
            mock_emb.return_value.embed_documents.side_effect = lambda texts: [
                [0.1] * 10 for _ in texts
            ]
            mock_emb.return_value.embed_query.return_value = [0.1] * 10
            indexer = RAGIndexer(docs_dir=docs_dir, persist_dir=chroma_dir)
            indexer.build_index()
            assert indexer.is_indexed()

    def test_indexer_loads_markdown_files(self, docs_dir, chroma_dir):
        with patch("rag.indexer.OpenAIEmbeddings"):
            indexer = RAGIndexer(docs_dir=docs_dir, persist_dir=chroma_dir)
            docs = indexer._load_documents()
            assert len(docs) >= 1
            assert any("connector" in d.page_content.lower() for d in docs)

    def test_indexer_loads_utf8_markdown_with_file_name_source(self, tmp_path, chroma_dir):
        content = "# Jira — Setup\n\nClick “Connect” ✓"
        (tmp_path / "utf8.md").write_text(content, encoding="utf-8")
        with patch("rag.indexer.OpenAIEmbeddings"):
            indexer = RAGIndexer(docs_dir=str(tmp_path), persist_dir=chroma_dir)
            docs = indexer._load_documents()
        assert [(d.page_content, d.metadata) for d in docs] == [(content, {"source": "utf8.md"})]


class TestRAGRetriever:
    @pytest.fixture
    def mock_collection(self):
        col = MagicMock()
        col.query.return_value = {
            "documents": [["Jira connector setup steps here."]],
            "metadatas": [[{"source": "connectors.md", "h2": "Jira"}]],
            "distances": [[0.12]],
        }
        return col

    def test_retriever_returns_documents(self, mock_collection):
        with (
            patch("rag.retriever.OpenAIEmbeddings") as mock_emb,
            patch("rag.retriever.chromadb.PersistentClient") as mock_client,
        ):
            mock_emb.return_value.embed_query.return_value = [0.1] * 10
            mock_client.return_value.get_collection.return_value = mock_collection
            retriever = RAGRetriever(persist_dir="/tmp/fake")
            results = retriever.retrieve("how do I set up Jira?")
            assert len(results) == 1
            assert "Jira" in results[0].page_content

    def test_retriever_includes_source_metadata(self, mock_collection):
        with (
            patch("rag.retriever.OpenAIEmbeddings") as mock_emb,
            patch("rag.retriever.chromadb.PersistentClient") as mock_client,
        ):
            mock_emb.return_value.embed_query.return_value = [0.1] * 10
            mock_client.return_value.get_collection.return_value = mock_collection
            retriever = RAGRetriever(persist_dir="/tmp/fake")
            results = retriever.retrieve("Jira setup")
            assert results[0].metadata["source"] == "connectors.md"

    def test_retriever_attaches_distance_to_metadata(self, mock_collection):
        with (
            patch("rag.retriever.OpenAIEmbeddings") as mock_emb,
            patch("rag.retriever.chromadb.PersistentClient") as mock_client,
        ):
            mock_emb.return_value.embed_query.return_value = [0.1] * 10
            mock_client.return_value.get_collection.return_value = mock_collection
            retriever = RAGRetriever(persist_dir="/tmp/fake")
            results = retriever.retrieve("Jira setup")
            assert "distance" in results[0].metadata
            assert results[0].metadata["distance"] == 0.12

    def test_retriever_filters_low_confidence_chunks(self, caplog):
        high_distance_collection = MagicMock()
        high_distance_collection.query.return_value = {
            "documents": [["Totally unrelated content about cooking recipes."]],
            "metadatas": [[{"source": "connectors.md"}]],
            "distances": [[0.85]],
        }
        with (
            patch("rag.retriever.OpenAIEmbeddings") as mock_emb,
            patch("rag.retriever.chromadb.PersistentClient") as mock_client,
        ):
            mock_emb.return_value.embed_query.return_value = [0.1] * 10
            mock_client.return_value.get_collection.return_value = high_distance_collection
            retriever = RAGRetriever(persist_dir="/tmp/fake", distance_threshold=0.5)
            with caplog.at_level(logging.INFO, logger="rag.retriever"):
                results = retriever.retrieve("Jira connector setup")
        assert results == []
        assert "All 1 retrieved chunks exceeded distance threshold" in caplog.text

    def test_retriever_keeps_chunks_within_threshold(self, mock_collection):
        with (
            patch("rag.retriever.OpenAIEmbeddings") as mock_emb,
            patch("rag.retriever.chromadb.PersistentClient") as mock_client,
        ):
            mock_emb.return_value.embed_query.return_value = [0.1] * 10
            mock_client.return_value.get_collection.return_value = mock_collection
            retriever = RAGRetriever(persist_dir="/tmp/fake", distance_threshold=0.5)
            results = retriever.retrieve("Jira setup")
            assert len(results) == 1

    def test_retriever_keeps_chunk_at_exact_threshold(self):
        at_threshold_collection = MagicMock()
        at_threshold_collection.query.return_value = {
            "documents": [["Some content at the boundary."]],
            "metadatas": [[{"source": "connectors.md"}]],
            "distances": [[0.5]],
        }
        with (
            patch("rag.retriever.OpenAIEmbeddings") as mock_emb,
            patch("rag.retriever.chromadb.PersistentClient") as mock_client,
        ):
            mock_emb.return_value.embed_query.return_value = [0.1] * 10
            mock_client.return_value.get_collection.return_value = at_threshold_collection
            retriever = RAGRetriever(persist_dir="/tmp/fake", distance_threshold=0.5)
            results = retriever.retrieve("anything")
        assert len(results) == 1
        assert results[0].metadata["distance"] == 0.5

    def test_retriever_passes_chunks_when_distances_absent(self):
        no_distances_collection = MagicMock()
        no_distances_collection.query.return_value = {
            "documents": [["Some content."]],
            "metadatas": [[{"source": "connectors.md"}]],
        }
        with (
            patch("rag.retriever.OpenAIEmbeddings") as mock_emb,
            patch("rag.retriever.chromadb.PersistentClient") as mock_client,
        ):
            mock_emb.return_value.embed_query.return_value = [0.1] * 10
            mock_client.return_value.get_collection.return_value = no_distances_collection
            retriever = RAGRetriever(persist_dir="/tmp/fake", distance_threshold=0.5)
            results = retriever.retrieve("anything")
        assert len(results) == 1
        assert results[0].metadata["distance"] == 0.0
