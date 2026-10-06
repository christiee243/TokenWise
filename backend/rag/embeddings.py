"""
TokenWise RAG Module — Embedding Engine
Converts text chunks and queries into dense vector embeddings using sentence-transformers,
with a deterministic fallback for offline or constrained environments.
"""

import hashlib
import logging
import re
from typing import List, Optional

import numpy as np

from .config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None  # type: ignore


class EmbeddingEngine:
    """Manages an embedding model for encoding text into vector embeddings."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._dimension = EMBEDDING_DIMENSION
        self._use_fallback = not HAS_SENTENCE_TRANSFORMERS

    @property
    def model(self):
        """Lazy-load the model on first use to avoid slow import-time downloads."""
        if self._model is None and not self._use_fallback:
            try:
                logger.info("Loading embedding model: %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                # Verify dimension matches config
                test_emb = self._model.encode(["test"])
                actual_dim = test_emb.shape[1]
                if actual_dim != self._dimension:
                    logger.warning(
                        "Model dimension %d differs from configured %d — updating.",
                        actual_dim,
                        self._dimension,
                    )
                    self._dimension = actual_dim
                logger.info("Embedding model loaded. Dimension: %d", self._dimension)
            except Exception as e:
                logger.warning(
                    "Failed to load SentenceTransformer (%s). Falling back to hash embeddings.",
                    e,
                )
                self._use_fallback = True
                self._model = None
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        if not self._use_fallback and self._model is None:
            _ = self.model
        return self._dimension

    def _fallback_encode(self, text: str) -> np.ndarray:
        """Deterministic feature hashing embedding fallback."""
        vec = np.zeros(self._dimension, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())
        if not words:
            vec[0] = 1.0
            return vec

        for word in words:
            # Word hash
            h_val = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h_val % self._dimension
            sign = 1.0 if (h_val >> 16) % 2 == 0 else -1.0
            vec[idx] += sign

            # Bigram sub-features
            for i in range(len(word) - 2):
                gram = word[i : i + 3]
                gh = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                g_idx = gh % self._dimension
                vec[g_idx] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_chunks(self, chunks: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Encode a list of text chunks into a 2D float32 numpy array.

        Args:
            chunks: List of text strings to embed.
            batch_size: Number of chunks to encode per batch.

        Returns:
            np.ndarray of shape (len(chunks), dimension) with dtype float32.
        """
        if not chunks:
            return np.empty((0, self._dimension), dtype=np.float32)

        if not self._use_fallback and self.model is not None:
            embeddings = self.model.encode(
                chunks,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            return embeddings.astype(np.float32)

        # Fallback encoding
        vectors = [self._fallback_encode(c) for c in chunks]
        return np.vstack(vectors).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string into a 1D float32 numpy array.

        Args:
            query: The search query text.

        Returns:
            np.ndarray of shape (dimension,) with dtype float32.
        """
        if not self._use_fallback and self.model is not None:
            embedding = self.model.encode(
                [query],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            return embedding[0].astype(np.float32)

        return self._fallback_encode(query)
