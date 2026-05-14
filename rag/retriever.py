import logging

import chromadb
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "security_docs"
_DEFAULT_K = 3


class RAGRetriever:
    def __init__(self, persist_dir: str, k: int = _DEFAULT_K) -> None:
        self._k = k
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_collection(_COLLECTION_NAME)

    def retrieve(self, query: str) -> list[Document]:
        query_embedding = self._embeddings.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=self._k,
            include=["documents", "metadatas", "distances"],
        )
        return self._to_documents(results)

    def format_for_prompt(self, docs: list[Document]) -> str:
        parts: list[str] = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            section = doc.metadata.get("h2") or doc.metadata.get("h1", "")
            citation = f"[{source}" + (f" — {section}" if section else "") + "]"
            parts.append(f"{citation}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _to_documents(self, results: dict) -> list[Document]:
        docs: list[Document] = []
        for text, metadata in zip(
            results["documents"][0], results["metadatas"][0]
        ):
            docs.append(Document(page_content=text, metadata=metadata))
        return docs
