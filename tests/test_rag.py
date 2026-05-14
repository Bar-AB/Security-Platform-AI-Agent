import pytest
from unittest.mock import patch, MagicMock


class TestRAGIndexer:
    @pytest.fixture
    def docs_dir(self, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("# Section One\n\nThis explains connector setup.\n\n## Jira\n\nJira integration steps here.")
        return str(tmp_path)

    @pytest.fixture
    def chroma_dir(self, tmp_path):
        return str(tmp_path / "chroma")

    def test_indexer_creates_collection(self, docs_dir, chroma_dir):
        from rag.indexer import RAGIndexer
        with patch("rag.indexer.OpenAIEmbeddings") as mock_emb:
            mock_emb.return_value.embed_documents.side_effect = lambda texts: [[0.1] * 10 for _ in texts]
            mock_emb.return_value.embed_query.return_value = [0.1] * 10
            indexer = RAGIndexer(docs_dir=docs_dir, persist_dir=chroma_dir)
            indexer.build_index()
            assert indexer.is_indexed()

    def test_indexer_loads_markdown_files(self, docs_dir, chroma_dir):
        from rag.indexer import RAGIndexer
        with patch("rag.indexer.OpenAIEmbeddings"):
            indexer = RAGIndexer(docs_dir=docs_dir, persist_dir=chroma_dir)
            docs = indexer._load_documents()
            assert len(docs) >= 1
            assert any("connector" in d.page_content.lower() for d in docs)


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
        from rag.retriever import RAGRetriever
        with patch("rag.retriever.OpenAIEmbeddings") as mock_emb, \
             patch("rag.retriever.chromadb.PersistentClient") as mock_client:
            mock_emb.return_value.embed_query.return_value = [0.1] * 10
            mock_client.return_value.get_collection.return_value = mock_collection
            retriever = RAGRetriever(persist_dir="/tmp/fake")
            results = retriever.retrieve("how do I set up Jira?")
            assert len(results) == 1
            assert "Jira" in results[0].page_content

    def test_retriever_includes_source_metadata(self, mock_collection):
        from rag.retriever import RAGRetriever
        with patch("rag.retriever.OpenAIEmbeddings") as mock_emb, \
             patch("rag.retriever.chromadb.PersistentClient") as mock_client:
            mock_emb.return_value.embed_query.return_value = [0.1] * 10
            mock_client.return_value.get_collection.return_value = mock_collection
            retriever = RAGRetriever(persist_dir="/tmp/fake")
            results = retriever.retrieve("Jira setup")
            assert results[0].metadata["source"] == "connectors.md"
