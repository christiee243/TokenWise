"""
TokenWise RAG Module — Configuration
Centralized settings for embedding models, vector dimensions, and retrieval thresholds.
"""

# ── Embedding Model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384          # Output dimension of all-MiniLM-L6-v2

# ── Vector Search Defaults ───────────────────────────────────────────────────
DEFAULT_TOP_K = 5                  # Default number of chunks to retrieve
MAX_TOP_K = 20                     # Hard cap on top_k to prevent abuse
MIN_RELEVANCE_SCORE = 0.15        # Discard results below this cosine similarity

# ── Query Analyzer ───────────────────────────────────────────────────────────
TOP_K_BY_COMPLEXITY = {
    "simple": 3,
    "moderate": 5,
    "complex": 8,
}

# ── Stop Words (for keyword extraction) ──────────────────────────────────────
STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "what", "which", "who", "whom", "whose", "where", "when", "how", "why",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than", "too",
    "very", "just", "about", "also", "this", "that", "these", "those",
    "it", "its", "i", "me", "my", "we", "us", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their",
})
