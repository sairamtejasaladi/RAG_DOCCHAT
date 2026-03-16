"""
Retriever Builder — creates a hybrid BM25 + vector retriever.
Uses Chroma's built-in ONNX embeddings (all-MiniLM-L6-v2) — no torch needed.
"""
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain.retrievers.ensemble import EnsembleRetriever
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_core.embeddings import Embeddings
from docchat.config.settings import settings
from docchat.utils.logging import logger


class ChromaDefaultEmbeddings(Embeddings):
    """Wraps Chroma's built-in ONNX embedding function as a LangChain Embeddings."""

    def __init__(self):
        self._ef = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._ef(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._ef([text])[0]


class RetrieverBuilder:
    def __init__(self):
        """Initialize the retriever builder with Chroma's built-in embeddings."""
        logger.info("Initializing Chroma default ONNX embeddings (all-MiniLM-L6-v2)...")
        self.embeddings = ChromaDefaultEmbeddings()
        logger.info("Embedding model loaded successfully.")

    def build_hybrid_retriever(self, docs):
        """Build a hybrid retriever using BM25 and vector-based retrieval."""
        # Create Chroma vector store — DB latency auto-captured by @observe_infra
        vector_store = self._build_vector_store(docs)
        logger.info("Vector store created successfully.")

        # Create BM25 retriever — tool execution auto-captured by @observe_tool
        bm25 = self._build_bm25(docs)
        logger.info("BM25 retriever created successfully.")

        # Combine retrievers into a hybrid retriever
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25, vector_store.as_retriever(
                search_kwargs={"k": settings.VECTOR_SEARCH_K}
            )],
            weights=settings.HYBRID_RETRIEVER_WEIGHTS,
        )
        logger.info("Hybrid retriever created successfully.")
        return hybrid_retriever

    def _build_vector_store(self, docs):
        """Build the Chroma vector store."""
        return Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
        )

    def _build_bm25(self, docs):
        """Build the BM25 retriever."""
        return BM25Retriever.from_documents(docs)
