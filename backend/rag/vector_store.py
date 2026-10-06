"""
TokenWise RAG Module — Vector Store
FAISS-backed in-memory vector index (with NumPy fallback) using cosine similarity search,
metadata storage, save/load support, and relevance filtering.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    faiss = None  # type: ignore

from .config import EMBEDDING_DIMENSION, MIN_RELEVANCE_SCORE

logger = logging.getLogger(__name__)


class VectorStore:
    """
    In-memory vector store using Inner Product on L2-normalized
    vectors (equivalent to cosine similarity).

    Uses FAISS if available; otherwise falls back to pure NumPy.
    Stores chunk text and arbitrary metadata alongside the index so that
    search results include full context without a separate lookup.
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.chunks: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.use_faiss = HAS_FAISS

        if self.use_faiss:
            self.index: Optional[faiss.IndexFlatIP] = faiss.IndexFlatIP(dimension)
            self._np_vectors: Optional[np.ndarray] = None
        else:
            self.index = None
            self._np_vectors = np.empty((0, dimension), dtype=np.float32)

    # ── Core Operations ──────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of vectors currently stored."""
        if self.use_faiss and self.index is not None:
            return self.index.ntotal
        elif self._np_vectors is not None:
            return len(self._np_vectors)
        return len(self.chunks)

    def add(
        self,
        embeddings: np.ndarray,
        chunks: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Add chunk embeddings and their associated text/metadata to the store.

        Args:
            embeddings: (N, dimension) float32 array of chunk embeddings.
            chunks: Corresponding text for each embedding.
            metadata: Optional list of dicts with per-chunk metadata
                      (e.g. chunk_id, section, tokens, page).

        Returns:
            Total number of vectors in the store after insertion.
        """
        if len(chunks) == 0:
            return self.size

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"store dimension {self.dimension}."
            )

        # Normalize to unit vectors for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings_norm = (embeddings / norms).astype(np.float32)

        if self.use_faiss and self.index is not None:
            self.index.add(embeddings_norm)
        else:
            if self._np_vectors is None or len(self._np_vectors) == 0:
                self._np_vectors = embeddings_norm
            else:
                self._np_vectors = np.vstack([self._np_vectors, embeddings_norm])

        self.chunks.extend(chunks)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(chunks))

        logger.info(
            "Added %d vectors to store (total: %d).", len(chunks), self.size
        )
        return self.size

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: float = MIN_RELEVANCE_SCORE,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Find the top_k most similar chunks to the query embedding.

        Args:
            query_embedding: 1D float32 array of the query vector.
            top_k: Maximum number of results to return.
            min_score: Minimum cosine similarity to include in results.

        Returns:
            List of (chunk_text, similarity_score, metadata) tuples,
            sorted by descending relevance.
        """
        if self.size == 0:
            return []

        query_vec = np.ascontiguousarray(
            query_embedding.reshape(1, -1), dtype=np.float32
        )
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec /= q_norm

        k = min(top_k, self.size)

        if self.use_faiss and self.index is not None:
            scores, indices = self.index.search(query_vec, k)
            score_list = scores[0]
            idx_list = indices[0]
        else:
            # NumPy dot product on normalized vectors
            sims = np.dot(self._np_vectors, query_vec.T).flatten()
            top_indices = np.argsort(-sims)[:k]
            score_list = sims[top_indices]
            idx_list = top_indices

        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for score, idx in zip(score_list, idx_list):
            if idx < 0 or idx >= len(self.chunks):
                continue
            sim = float(score)
            if sim < min_score:
                continue
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            results.append((self.chunks[idx], sim, meta))

        return results

    def reset(self) -> None:
        """Clear all vectors, chunks, and metadata."""
        if self.use_faiss and self.index is not None:
            self.index.reset()
        self._np_vectors = np.empty((0, self.dimension), dtype=np.float32)
        self.chunks.clear()
        self.metadata.clear()
        logger.info("Vector store reset.")

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """
        Persist the index and metadata to disk.

        Args:
            directory: Path to the directory where files will be saved.
        """
        os.makedirs(directory, exist_ok=True)
        if self.use_faiss and self.index is not None:
            faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        elif self._np_vectors is not None:
            np.save(os.path.join(directory, "vectors.npy"), self._np_vectors)

        with open(os.path.join(directory, "store_data.json"), "w", encoding="utf-8") as f:
            json.dump({"chunks": self.chunks, "metadata": self.metadata}, f)
        logger.info("Vector store saved to %s (%d vectors).", directory, self.size)

    def load(self, directory: str) -> None:
        """
        Load a previously saved index and metadata from disk.

        Args:
            directory: Path to the directory containing saved files.
        """
        data_path = os.path.join(directory, "store_data.json")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Vector store data not found in {directory}.")

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        self.metadata = data["metadata"]

        index_path = os.path.join(directory, "index.faiss")
        vec_path = os.path.join(directory, "vectors.npy")

        if self.use_faiss and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.dimension = self.index.d
        elif os.path.exists(vec_path):
            self._np_vectors = np.load(vec_path)
            self.dimension = self._np_vectors.shape[1] if len(self._np_vectors) > 0 else self.dimension
            self.use_faiss = False
        else:
            raise FileNotFoundError(f"Neither index.faiss nor vectors.npy found in {directory}.")

        logger.info("Vector store loaded from %s (%d vectors).", directory, self.size)
