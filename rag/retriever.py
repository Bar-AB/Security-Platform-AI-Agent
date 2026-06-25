import logging

import chromadb
from chromadb import QueryResult
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage as _HumanMessage


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



_MULTI_QUERY_PROMPT = (
    "Generate 3 alternative phrasings of the following search query for a security platform "
    "documentation knowledge base. Each phrasing should capture the same intent from a "
    "different angle. Return ONLY the 3 queries, one per line, no numbering, no extra text.\n\n"
    "Original query: {query}"
)

_MAX_VARIANTS = 3

class MultiQueryRAGRetriever(RAGRetriever):
    def __init__(
        self,
        persist_dir: str,
        llm: BaseChatModel,
        k: int = _DEFAULT_K,
        distance_threshold: float = _DEFAULT_DISTANCE_THRESHOLD,
    ) -> None:
        super().__init__(persist_dir=persist_dir, k=k, distance_threshold=distance_threshold)
        self._llm = llm

    def retrieve(self, query: str) -> list[Document]:
        queries = self._generate_queries(query)
        seen: set[str] = set()
        docs: list[Document] = []
        for q in queries:
            for doc in super().retrieve(q):
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    docs.append(doc)
        logger.debug(
            "MultiQueryRAG: %d unique chunks from %d queries", len(docs), len(queries)
        )
        return docs

    def _generate_queries(self, query: str) -> list[str]:
        try:
            prompt = _MULTI_QUERY_PROMPT.format(query=query)
            response = self._llm.invoke(
                [SystemMessage(content=prompt), _HumanMessage(content=query)]
            )
            raw = response.content
            content = raw if isinstance(raw, str) else str(raw)
            variants = [line.strip() for line in content.strip().splitlines() if line.strip()]
            return [query] + variants[:_MAX_VARIANTS]
        except Exception:
            logger.exception("Multi-query generation failed — falling back to single query")
            return [query]
