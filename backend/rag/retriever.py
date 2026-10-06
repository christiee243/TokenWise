"""
TokenWise RAG Module — Retriever
Main orchestrator that ties together embedding, indexing, query analysis,
and vector search into a single high-level interface.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from .config import DEFAULT_TOP_K, MAX_TOP_K
from .embeddings import EmbeddingEngine
from .query_analyzer import QueryAnalyzer
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    High-level RAG retriever — the main entry point for M4 (FastAPI).

    Typical workflow:
        1. After M1's ``process_pdf`` returns chunks, call ``index_document``
           to embed and store them.
        2. When the user asks a question, call ``retrieve`` to get the most
           relevant chunks along with query analysis metadata.

    The retriever supports multiple documents via ``document_id``.
    Each document gets its own isolated vector store.
    """

    def __init__(self) -> None:
        self.embedding_engine = EmbeddingEngine()
        self.query_analyzer = QueryAnalyzer()

        # Per-document vector stores: {document_id: VectorStore}
        self._stores: Dict[str, VectorStore] = {}

    # ── Document Indexing ────────────────────────────────────────────────

    def index_document(
        self,
        chunks: List[Dict[str, Any]],
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Index chunks produced by M1's ``process_pdf`` into the vector store.

        Args:
            chunks: List of chunk dicts from M1, each containing at minimum
                    ``text`` (str). Optional fields: ``chunk_id``, ``section``,
                    ``tokens``.
            document_id: Unique identifier for the document. Auto-generated
                         if not provided.

        Returns:
            Dict with indexing status, document_id, and chunk count.
        """
        if document_id is None:
            document_id = str(uuid.uuid4())[:12]

        # Extract text and metadata from M1's chunk format
        texts: List[str] = []
        metadata: List[Dict[str, Any]] = []

        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            texts.append(text)
            metadata.append({
                "chunk_id": chunk.get("chunk_id", len(texts) - 1),
                "section": chunk.get("section", "General"),
                "tokens": chunk.get("tokens", 0),
            })

        if not texts:
            return {
                "status": "empty",
                "document_id": document_id,
                "chunks_indexed": 0,
                "embedding_dimension": self.embedding_engine.dimension,
            }

        # Create a fresh store for this document (replaces any existing one)
        store = VectorStore(dimension=self.embedding_engine.dimension)

        # Embed and index
        logger.info(
            "Indexing %d chunks for document '%s'...", len(texts), document_id
        )
        embeddings = self.embedding_engine.embed_chunks(texts)
        store.add(embeddings, texts, metadata)

        self._stores[document_id] = store

        logger.info(
            "Document '%s' indexed: %d chunks, %d-dim embeddings.",
            document_id,
            store.size,
            self.embedding_engine.dimension,
        )

        return {
            "status": "indexed",
            "document_id": document_id,
            "chunks_indexed": store.size,
            "embedding_dimension": self.embedding_engine.dimension,
        }

    # ── Retrieval ────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        document_id: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve the most relevant chunks for a natural language query.

        Args:
            query: The user's question or search query.
            document_id: Which document's index to search.
            top_k: Override the number of results. If None, the query
                   analyzer's suggestion is used.

        Returns:
            Dict containing query_analysis, retrieved_chunks list,
            num_chunks_retrieved, and suggested_top_k.
        """
        # Validate document exists
        if document_id not in self._stores:
            return {
                "error": f"Document '{document_id}' not found. Index it first.",
                "query_analysis": None,
                "retrieved_chunks": [],
                "num_chunks_retrieved": 0,
                "suggested_top_k": DEFAULT_TOP_K,
            }

        store = self._stores[document_id]

        # Analyze the query
        analysis = self.query_analyzer.analyze(query)

        # Determine top_k
        k = top_k if top_k is not None else analysis.suggested_top_k
        k = max(1, min(k, MAX_TOP_K))

        # Embed query and search
        query_embedding = self.embedding_engine.embed_query(query)
        results = store.search(query_embedding, top_k=k)

        retrieved_chunks = [
            {
                "text": text,
                "relevance_score": round(score, 4),
                "metadata": meta,
            }
            for text, score, meta in results
        ]

        logger.info(
            "Query '%s' on doc '%s': intent=%s, complexity=%s, returned %d chunks.",
            query[:60],
            document_id,
            analysis.intent,
            analysis.complexity,
            len(retrieved_chunks),
        )

        return {
            "query_analysis": {
                "intent": analysis.intent,
                "complexity": analysis.complexity,
                "keywords": analysis.keywords,
            },
            "retrieved_chunks": retrieved_chunks,
            "num_chunks_retrieved": len(retrieved_chunks),
            "suggested_top_k": analysis.suggested_top_k,
        }

    # ── Management ───────────────────────────────────────────────────────

    def has_document(self, document_id: str) -> bool:
        """Check if a document is currently indexed."""
        return document_id in self._stores

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents and their chunk counts."""
        return [
            {"document_id": doc_id, "chunks_indexed": store.size}
            for doc_id, store in self._stores.items()
        ]

    def remove_document(self, document_id: str) -> bool:
        """Remove a document's index. Returns True if it existed."""
        if document_id in self._stores:
            self._stores[document_id].reset()
            del self._stores[document_id]
            logger.info("Document '%s' removed from index.", document_id)
            return True
        return False

    def reset(self) -> None:
        """Clear all document indexes."""
        for store in self._stores.values():
            store.reset()
        self._stores.clear()
        logger.info("All document indexes cleared.")
