import logging

import chromadb
from chromadb import QueryResult
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
        results: QueryResult = self._collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
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

    def _to_documents(self, results: QueryResult) -> list[Document]:
        documents = results["documents"]
        metadatas = results["metadatas"]
        if not documents or not metadatas:
            return []
        return [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(documents[0], metadatas[0])
        ]
