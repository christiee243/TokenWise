"""
TokenWise — RAG (Retrieval-Augmented Generation) Module
Member 2 deliverable: Query understanding, text embeddings, vector storage,
and relevant chunk retrieval.

Usage:
    from rag import RAGRetriever

    retriever = RAGRetriever()
    retriever.index_document(chunks, document_id="my_doc")
    results = retriever.retrieve("How does attention work?", document_id="my_doc")
"""

from .retriever import RAGRetriever
from .embeddings import EmbeddingEngine
from .vector_store import VectorStore
from .query_analyzer import QueryAnalyzer, QueryAnalysis

__all__ = [
    "RAGRetriever",
    "EmbeddingEngine",
    "VectorStore",
    "QueryAnalyzer",
    "QueryAnalysis",
]
