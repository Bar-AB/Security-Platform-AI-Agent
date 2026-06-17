import logging

import chromadb
from chromadb import QueryResult
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "security_docs"
_DEFAULT_K = 5
# Cosine distance threshold: 1 - cosine_similarity, so 0.5 requires ≥ 0.5 cosine similarity.
# Chunks above this are off-topic and excluded to prevent low-quality context from reaching the LLM.
_DEFAULT_DISTANCE_THRESHOLD = 0.5


class RAGRetriever:
    def __init__(
        self,
        persist_dir: str,
        k: int = _DEFAULT_K,
        distance_threshold: float = _DEFAULT_DISTANCE_THRESHOLD,
    ) -> None:
        self._k = k
        self._distance_threshold = distance_threshold
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
        distances = results.get("distances")
        if not documents or not metadatas:
            return []
        # distances is always present when include=["distances"] is passed, but we
        # defensively fall back to 0.0 (pass-through) rather than rejecting chunks
        # whose distance is unknown — dropping context silently is worse than keeping it.
        raw_distances = distances[0] if distances else []
        docs: list[Document] = []
        for i, (text, meta) in enumerate(zip(documents[0], metadatas[0])):
            distance = raw_distances[i] if i < len(raw_distances) else 0.0
            if distance > self._distance_threshold:
                logger.debug(
                    "Filtered chunk (distance=%.3f > threshold=%.3f): '%s...'",
                    distance,
                    self._distance_threshold,
                    text[:60],
                )
                continue
            docs.append(Document(page_content=text, metadata={**meta, "distance": distance}))
        if not docs and documents[0]:
            logger.info(
                "All %d retrieved chunks exceeded distance threshold %.3f — no confident RAG results",
                len(documents[0]),
                self._distance_threshold,
            )
        return docs
