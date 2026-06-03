import hashlib
import logging
from pathlib import Path

import chromadb
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

_HEADERS_TO_SPLIT = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_COLLECTION_NAME = "security_docs"


class RAGIndexer:
    def __init__(self, docs_dir: str, persist_dir: str) -> None:
        self._docs_dir = Path(docs_dir)
        self._persist_dir = persist_dir
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._client = chromadb.PersistentClient(path=persist_dir)

    def build_index(self) -> None:
        docs = self._load_documents()
        chunks = self._split_documents(docs)
        docs_hash = self._compute_docs_hash()
        self._store_chunks(chunks, docs_hash)
        logger.info("Indexed %d chunks from %s", len(chunks), self._docs_dir)

    def is_indexed(self) -> bool:
        try:
            col = self._client.get_collection(_COLLECTION_NAME)
            if col.count() == 0:
                return False
            stored_hash = (col.metadata or {}).get("docs_hash")
            return stored_hash == self._compute_docs_hash()
        except Exception:
            logger.debug("Collection not found or not indexed", exc_info=True)
            return False

    def _compute_docs_hash(self) -> str:
        h = hashlib.md5()
        for path in sorted(self._docs_dir.glob("*.md")):
            h.update(path.name.encode())
            h.update(path.read_bytes())
        return h.hexdigest()

    def _load_documents(self) -> list[Document]:
        docs: list[Document] = []
        for path in self._docs_dir.glob("*.md"):
            loader = TextLoader(str(path))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = path.name
            docs.extend(loaded)
        return docs

    def _split_documents(self, docs: list[Document]) -> list[Document]:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_HEADERS_TO_SPLIT,
            strip_headers=False,
        )
        chunks: list[Document] = []
        for doc in docs:
            splits = splitter.split_text(doc.page_content)
            for chunk in splits:
                chunk.metadata.update(doc.metadata)
            chunks.extend(splits)
        return chunks

    def _store_chunks(self, chunks: list[Document], docs_hash: str) -> None:
        try:
            self._client.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass
        collection = self._client.create_collection(
            _COLLECTION_NAME,
            # cosine is the right metric for text embeddings; ChromaDB otherwise defaults to L2.
            metadata={"docs_hash": docs_hash, "hnsw:space": "cosine"},
        )
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        embeddings = self._embeddings.embed_documents(texts)
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        collection.add(
            documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids  # type: ignore[arg-type]
        )
