"""
TokenWise RAG Module — Retriever
Main orchestrator that ties together embedding, indexing, query analysis,
and vector search into a single high-level interface.
"""

import logging
import re
import uuid
from typing import Any, Dict, List, Optional
import numpy as np

from .config import DEFAULT_TOP_K, MAX_TOP_K
from .database import SQLiteVectorRepository
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

    def __init__(self, database_path: str = ":memory:") -> None:
        self.embedding_engine = EmbeddingEngine()
        self.query_analyzer = QueryAnalyzer()
        self._repository = SQLiteVectorRepository(database_path)

        # Per-document vector stores: {document_id: VectorStore}
        self._stores: Dict[str, VectorStore] = {}
        self._load_persisted_documents()

    def _load_persisted_documents(self) -> None:
        """Rebuild searchable vector indexes from persisted embeddings."""
        for document_id, texts, metadata, embeddings in self._repository.load_all_documents():
            store = VectorStore(dimension=embeddings.shape[1])
            store.add(embeddings, texts, metadata)
            self._stores[document_id] = store

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
        self._repository.replace_document(document_id, embeddings, texts, metadata)

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

        # Synthesize direct answer from the top retrieved chunks
        direct_answer = self.synthesize_direct_answer(
            query=query,
            retrieved_chunks=retrieved_chunks,
            query_embedding=query_embedding,
        )

        return {
            "query_analysis": {
                "intent": analysis.intent,
                "complexity": analysis.complexity,
                "keywords": analysis.keywords,
            },
            "direct_answer": direct_answer,
            "retrieved_chunks": retrieved_chunks,
            "num_chunks_retrieved": len(retrieved_chunks),
            "suggested_top_k": analysis.suggested_top_k,
        }

    def synthesize_direct_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        query_embedding: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """
        Extract the most direct, high-relevance answering sentence from the
        top retrieved chunks to provide a concise, direct answer to the user's question.
        """
        if not retrieved_chunks:
            return None

        # Focus on top 2 retrieved chunks
        candidate_texts = [c["text"] for c in retrieved_chunks[:2] if c.get("text")]
        if not candidate_texts:
            return None

        sentences: List[str] = []
        for t in candidate_texts:
            # Clean citation brackets like [2], [a], etc. for cleaner readability
            cleaned_t = re.sub(r'\[[0-9a-zA-Z,\s]+\]', '', t)
            splits = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned_t) if s.strip()]
            sentences.extend(splits)

        if not sentences:
            return None

        if len(sentences) == 1:
            return sentences[0]

        q_lower = query.lower()
        stopwords = {
            'is', 'are', 'was', 'were', 'a', 'an', 'the', 'in', 'on', 'at', 'to',
            'for', 'of', 'and', 'or', 'do', 'does', 'did', 'how', 'what', 'when',
            'why', 'who', 'where', 'which', 'can', 'could', 'would', 'will', 'should'
        }
        q_tokens = [w for w in re.findall(r'\b\w+\b', q_lower) if w not in stopwords]

        def stem(w: str) -> str:
            return w.lower().rstrip('sedting')

        q_stems = set(stem(w) for w in q_tokens)

        # Compute sentence embeddings
        s_embeddings = self.embedding_engine.embed_chunks(sentences)
        if query_embedding is None:
            query_embedding = self.embedding_engine.embed_query(query)

        emb_scores = np.dot(s_embeddings, query_embedding)

        scores: List[float] = []
        for i, s in enumerate(sentences):
            s_lower = s.lower()
            s_tokens = re.findall(r'\b\w+\b', s_lower)
            s_stems = set(stem(w) for w in s_tokens)

            overlap = len(q_stems & s_stems) / max(1, len(q_stems))

            intent_boost = 0.0
            if 'how many' in q_lower or 'how much' in q_lower:
                if any(w in s_lower for w in ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'number', 'count']) or re.search(r'\b\d+\b', s_lower):
                    intent_boost += 0.5
            elif 'when' in q_lower:
                if any(w in s_lower for w in ['once', 'when', 'after', 'until', 'upon', 'time', 'during']):
                    intent_boost += 0.5
            elif 'win' in q_lower or 'winner' in q_lower:
                if any(w in s_lower for w in ['win', 'wins', 'winner', 'won', 'victory']):
                    intent_boost += 0.5
            elif 'score' in q_lower or 'run' in q_lower or 'point' in q_lower:
                if any(w in s_lower for w in ['score', 'scores', 'run', 'runs', 'point', 'points', 'base', 'bases']):
                    intent_boost += 0.4
            elif 'who' in q_lower:
                if any(w in s_lower for w in ['player', 'team', 'pitcher', 'batter', 'author', 'lead']):
                    intent_boost += 0.35
            elif 'where' in q_lower:
                if any(w in s_lower for w in ['field', 'pitch', 'circuit', 'area', 'location', 'place']):
                    intent_boost += 0.35

            total = (float(emb_scores[i]) * 0.3) + (overlap * 0.4) + intent_boost
            scores.append(total)

        best_idx = int(np.argmax(scores))
        return sentences[best_idx]

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
        if not self._repository.delete_document(document_id):
            return False
        store = self._stores.pop(document_id, None)
        if store is not None:
            store.reset()
        logger.info("Document '%s' removed from index.", document_id)
        return True

    def reset(self) -> None:
        """Clear all document indexes."""
        for store in self._stores.values():
            store.reset()
        self._stores.clear()
        self._repository.clear()
        logger.info("All document indexes cleared.")

    def close(self) -> None:
        """Close the persistent storage connection."""
        self._repository.close()
