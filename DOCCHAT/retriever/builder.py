"""
Retriever Builder — creates a hybrid BM25 + vector retriever.
Uses Chroma's built-in ONNX embeddings (all-MiniLM-L6-v2) — no torch needed.
"""
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
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


# class RetrieverBuilder:
#     def __init__(self):
#         self.embeddings = ChromaDefaultEmbeddings()

#     def reciprocal_rank_fusion(search_results: list[list], k=60):
#         """
#         Combines multiple ranked lists using Reciprocal Rank Fusion.
#         k is a smoothing constant (standard is 60).
#         """
#         fused_scores = {}
#         for docs in search_results:
#             for rank, doc in enumerate(docs):
#                 # Use page_content as a unique key for deduplication
#                 content = doc.page_content
#                 if content not in fused_scores:
#                     fused_scores[content] = {"doc": doc, "score": 0}
                
#                 # Formula: 1 / (rank + k)
#                 fused_scores[content]["score"] += 1 / (rank + k)

#         # Sort documents by the new fused score
#         reranked_docs = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
#         return [item["doc"] for item in reranked_docs]

#     def build_hybrid_retriever(self, docs):
#         vector_store = self._build_vector_store(docs)
#         bm25 = self._build_bm25(docs)
        
#         # We create a simple wrapper function instead of an EnsembleRetriever
#         def manual_hybrid_search(query: str):
#             # 1. Get results from both
#             vector_docs = vector_store.as_retriever(search_kwargs={"k": 10}).invoke(query)
#             bm25_docs = bm25.invoke(query)
            
#             # 2. Fuse them
#             return self.reciprocal_rank_fusion([vector_docs, bm25_docs])
            
#         return manual_hybrid_search

#     def _build_vector_store(self, docs):
#         return Chroma.from_documents(documents=docs, embedding=self.embeddings)

#     def _build_bm25(self, docs):
#         return BM25Retriever.from_documents(docs)
class HybridRetriever:
    """Manual hybrid retriever that merges BM25 and Vector search results."""
    
    def __init__(self, vector_retriever, bm25_retriever, weights: list[float]):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.weights = weights # e.g., [0.4, 0.6]

    def invoke(self, query: str) -> list:
        # 1. Get results from both retrievers
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        
        # 2. Apply Reciprocal Rank Fusion (RRF) with weights
        return self._reciprocal_rank_fusion([bm25_docs, vector_docs])

    def _reciprocal_rank_fusion(self, search_results: list[list], k=60):
        fused_scores = {}
        
        # Zip results with their corresponding weights from settings
        for docs, weight in zip(search_results, self.weights):
            for rank, doc in enumerate(docs):
                content = doc.page_content
                if content not in fused_scores:
                    fused_scores[content] = {"doc": doc, "score": 0}
                
                # Standard RRF formula weighted by your settings
                # Weighted Score = Weight * (1 / (rank + k))
                fused_scores[content]["score"] += weight * (1 / (rank + k))

        # Sort by the new fused score
        reranked_docs = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in reranked_docs]


class RetrieverBuilder:
    def __init__(self):
        logger.info("Initializing Chroma default ONNX embeddings...")
        self.embeddings = ChromaDefaultEmbeddings()

    def build_hybrid_retriever(self, docs):
        # Build the underlying retrievers
        vector_store = self._build_vector_store(docs)
        vector_retriever = vector_store.as_retriever(
            search_kwargs={"k": settings.VECTOR_SEARCH_K}
        )
        bm25_retriever = self._build_bm25(docs)

        # Return our custom class instead of EnsembleRetriever
        logger.info("Building custom HybridRetriever...")
        return HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            weights=settings.HYBRID_RETRIEVER_WEIGHTS
        )

    def _build_vector_store(self, docs):
        return Chroma.from_documents(documents=docs, embedding=self.embeddings)

    def _build_bm25(self, docs):
        return BM25Retriever.from_documents(docs)