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
            mock_emb.return_value.embed_documents.return_value = [[0.1] * 10]
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
