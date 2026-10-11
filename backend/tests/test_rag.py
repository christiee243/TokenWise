"""
TokenWise RAG Module — End-to-End Tests
Tests the complete pipeline: embedding → indexing → query analysis → retrieval.
Also tests individual components in isolation.
Run: python -m pytest backend/tests/test_rag.py -v
  or: cd backend && python tests/test_rag.py
"""

import sys
import os
import tempfile

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure backend is on the path so `rag` package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.config import EMBEDDING_DIMENSION, TOP_K_BY_COMPLEXITY
from rag.query_analyzer import QueryAnalyzer
from rag.embeddings import EmbeddingEngine
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever


# ── Sample Data ──────────────────────────────────────────────────────────────
# Simulates chunks from M1's process_pdf output
SAMPLE_CHUNKS = [
    {
        "chunk_id": 0,
        "text": "The Transformer model architecture relies entirely on self-attention mechanisms, dispensing with recurrence and convolutions. It uses multi-head attention to jointly attend to information from different representation subspaces.",
        "section": "Abstract",
        "tokens": 38,
    },
    {
        "chunk_id": 1,
        "text": "Recurrent neural networks, long short-term memory networks, and gated recurrent neural networks have been firmly established as state of the art approaches in sequence modeling and transduction problems.",
        "section": "Introduction",
        "tokens": 33,
    },
    {
        "chunk_id": 2,
        "text": "An attention function maps a query and a set of key-value pairs to an output. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.",
        "section": "Model Architecture",
        "tokens": 48,
    },
    {
        "chunk_id": 3,
        "text": "We trained our models on the standard WMT 2014 English-to-German and English-to-French translation tasks. The English-to-German dataset consists of about 4.5 million sentence pairs.",
        "section": "Training",
        "tokens": 35,
    },
    {
        "chunk_id": 4,
        "text": "The Transformer achieves 28.4 BLEU on the English-to-German translation task, improving over the existing best results by over 2 BLEU. On the English-to-French task, it achieves a new single-model state-of-the-art BLEU score of 41.0.",
        "section": "Results",
        "tokens": 47,
    },
    {
        "chunk_id": 5,
        "text": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this capability.",
        "section": "Model Architecture",
        "tokens": 31,
    },
    {
        "chunk_id": 6,
        "text": "In encoder-decoder attention layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence.",
        "section": "Model Architecture",
        "tokens": 44,
    },
    {
        "chunk_id": 7,
        "text": "Positional encoding is added to give the model information about the relative or absolute position of tokens in the sequence. We use sine and cosine functions of different frequencies for positional encodings.",
        "section": "Model Architecture",
        "tokens": 36,
    },
]


# ═════════════════════════════════════════════════════════════════════════════
# 1. Query Analyzer Tests
# ═════════════════════════════════════════════════════════════════════════════

def test_query_analyzer_intent():
    """Test intent classification for different query types."""
    qa = QueryAnalyzer()

    # Factual (default)
    result = qa.analyze("How does the Transformer model work?")
    assert result.intent == "factual", f"Expected factual, got {result.intent}"

    # Summary
    result = qa.analyze("Give me a summary of the paper")
    assert result.intent == "summary", f"Expected summary, got {result.intent}"

    # Comparison
    result = qa.analyze("Compare the Transformer with RNN approaches")
    assert result.intent == "comparison", f"Expected comparison, got {result.intent}"

    # Extraction
    result = qa.analyze("List all the BLEU scores mentioned")
    assert result.intent == "extraction", f"Expected extraction, got {result.intent}"

    print("  ✓ Intent classification: PASSED")


def test_query_analyzer_complexity():
    """Test complexity scoring."""
    qa = QueryAnalyzer()

    # Simple
    result = qa.analyze("What is attention?")
    assert result.complexity == "simple", f"Expected simple, got {result.complexity}"
    assert result.suggested_top_k == TOP_K_BY_COMPLEXITY["simple"]

    # Moderate
    result = qa.analyze("Explain how attention works in the model")
    assert result.complexity == "moderate", f"Expected moderate, got {result.complexity}"

    # Complex
    result = qa.analyze("Analyze the relationship between attention and performance")
    assert result.complexity == "complex", f"Expected complex, got {result.complexity}"
    assert result.suggested_top_k == TOP_K_BY_COMPLEXITY["complex"]

    print("  ✓ Complexity scoring: PASSED")


def test_query_analyzer_keywords():
    """Test keyword extraction."""
    qa = QueryAnalyzer()
    result = qa.analyze("How does multi-head attention improve translation quality?")
    assert len(result.keywords) > 0, "Should extract at least one keyword"
    assert "multi-head" in result.keywords or "attention" in result.keywords
    print(f"  ✓ Keyword extraction: PASSED (keywords: {result.keywords})")


# ═════════════════════════════════════════════════════════════════════════════
# 2. Embedding Engine Tests
# ═════════════════════════════════════════════════════════════════════════════

def test_embedding_engine():
    """Test that the embedding engine produces correct-dimension vectors."""
    engine = EmbeddingEngine()

    # Single query
    query_emb = engine.embed_query("test query")
    assert query_emb.shape == (engine.dimension,), (
        f"Expected ({engine.dimension},), got {query_emb.shape}"
    )

    # Multiple chunks
    texts = ["chunk one", "chunk two", "chunk three"]
    chunk_embs = engine.embed_chunks(texts)
    assert chunk_embs.shape == (3, engine.dimension), (
        f"Expected (3, {engine.dimension}), got {chunk_embs.shape}"
    )

    # Empty list
    empty_embs = engine.embed_chunks([])
    assert empty_embs.shape[0] == 0, "Empty input should return empty array"

    print(f"  ✓ Embedding engine: PASSED (dimension={engine.dimension})")


# ═════════════════════════════════════════════════════════════════════════════
# 3. Vector Store Tests
# ═════════════════════════════════════════════════════════════════════════════

def test_vector_store():
    """Test add, search, and reset operations."""
    import numpy as np

    store = VectorStore(dimension=EMBEDDING_DIMENSION)
    assert store.size == 0, "New store should be empty"

    # Add some fake embeddings
    n_chunks = 5
    embeddings = np.random.randn(n_chunks, EMBEDDING_DIMENSION).astype(np.float32)
    chunks = [f"chunk {i}" for i in range(n_chunks)]
    metadata = [{"chunk_id": i, "section": f"Section {i}"} for i in range(n_chunks)]

    store.add(embeddings, chunks, metadata)
    assert store.size == n_chunks, f"Expected {n_chunks}, got {store.size}"

    # Search
    query = np.random.randn(EMBEDDING_DIMENSION).astype(np.float32)
    results = store.search(query, top_k=3, min_score=-1.0)  # low min_score for random vectors
    assert len(results) <= 3, "Should return at most 3 results"
    assert all(len(r) == 3 for r in results), "Each result should be (text, score, meta)"

    # Reset
    store.reset()
    assert store.size == 0, "Store should be empty after reset"

    print("  ✓ Vector store: PASSED")


# ═════════════════════════════════════════════════════════════════════════════
# 4. End-to-End Retriever Tests
# ═════════════════════════════════════════════════════════════════════════════

def test_retriever_index_and_retrieve():
    """Full pipeline: index M1 chunks → query → verify relevant results."""
    retriever = RAGRetriever()

    # Index sample document
    result = retriever.index_document(SAMPLE_CHUNKS, document_id="test_doc")
    assert result["status"] == "indexed", f"Expected 'indexed', got {result['status']}"
    assert result["chunks_indexed"] == len(SAMPLE_CHUNKS)
    assert result["document_id"] == "test_doc"
    print(f"  ✓ Indexing: {result['chunks_indexed']} chunks indexed")

    # Query about attention mechanism
    query_result = retriever.retrieve(
        "How does the attention mechanism work?",
        document_id="test_doc",
    )
    assert "error" not in query_result, f"Retrieval error: {query_result.get('error')}"
    assert query_result["num_chunks_retrieved"] > 0, "Should retrieve at least one chunk"
    assert query_result["query_analysis"]["intent"] == "factual"

    # Check that attention-related chunks are ranked high
    top_chunk = query_result["retrieved_chunks"][0]
    assert top_chunk["relevance_score"] > 0.0, "Top result should have positive score"
    assert "attention" in top_chunk["text"].lower(), (
        "Top result for attention query should mention 'attention'"
    )
    print(f"  ✓ Attention query: top chunk score={top_chunk['relevance_score']:.4f}")
    print(f"    Section: {top_chunk['metadata'].get('section', 'N/A')}")

    # Query about training data
    train_result = retriever.retrieve(
        "What datasets were used for training?",
        document_id="test_doc",
    )
    assert train_result["num_chunks_retrieved"] > 0
    top_train = train_result["retrieved_chunks"][0]
    assert any(
        kw in top_train["text"].lower()
        for kw in ["training", "wmt", "dataset", "english"]
    ), "Training query should surface training-related chunks"
    print(f"  ✓ Training query: top chunk score={top_train['relevance_score']:.4f}")
    print(f"    Section: {top_train['metadata'].get('section', 'N/A')}")

    # Query about BLEU scores
    bleu_result = retriever.retrieve(
        "What BLEU scores did the Transformer achieve?",
        document_id="test_doc",
    )
    assert bleu_result["num_chunks_retrieved"] > 0
    top_bleu = bleu_result["retrieved_chunks"][0]
    print(f"  ✓ BLEU query: top chunk score={top_bleu['relevance_score']:.4f}")
    print(f"    Section: {top_bleu['metadata'].get('section', 'N/A')}")

    print("  ✓ End-to-end retrieval: PASSED")


def test_retriever_missing_document():
    """Querying a non-existent document should return an error."""
    retriever = RAGRetriever()
    result = retriever.retrieve("test query", document_id="nonexistent")
    assert "error" in result, "Should return error for missing document"
    assert result["num_chunks_retrieved"] == 0
    print("  ✓ Missing document handling: PASSED")


def test_retriever_document_management():
    """Test list, has, remove, and reset operations."""
    retriever = RAGRetriever()

    # Index two documents
    retriever.index_document(SAMPLE_CHUNKS[:4], document_id="doc_a")
    retriever.index_document(SAMPLE_CHUNKS[4:], document_id="doc_b")

    assert retriever.has_document("doc_a")
    assert retriever.has_document("doc_b")
    assert not retriever.has_document("doc_c")

    docs = retriever.list_documents()
    assert len(docs) == 2

    # Remove one
    assert retriever.remove_document("doc_a")
    assert not retriever.has_document("doc_a")
    assert retriever.has_document("doc_b")

    # Reset all
    retriever.reset()
    assert len(retriever.list_documents()) == 0

    print("  ✓ Document management: PASSED")


def test_retriever_persistence():
    """Indexed documents should remain available after the retriever restarts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        database_path = os.path.join(temp_dir, "tokenwise.sqlite3")
        retriever = RAGRetriever(database_path=database_path)
        retriever.index_document(SAMPLE_CHUNKS[:3], document_id="persisted_doc")
        retriever.close()

        restarted_retriever = RAGRetriever(database_path=database_path)
        assert restarted_retriever.has_document("persisted_doc")
        documents = restarted_retriever.list_documents()
        assert documents == [{"document_id": "persisted_doc", "chunks_indexed": 3}]

        result = restarted_retriever.retrieve(
            "How does attention work?", document_id="persisted_doc"
        )
        assert result["num_chunks_retrieved"] > 0
        assert "error" not in result

        restarted_retriever.remove_document("persisted_doc")
        restarted_retriever.close()

        verify_retriever = RAGRetriever(database_path=database_path)
        assert not verify_retriever.has_document("persisted_doc")
        verify_retriever.close()

    print("  ✓ SQLite persistence across restarts: PASSED")


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    """Run all M2 RAG tests."""
    print("=" * 60)
    print("TokenWise M2 — RAG Module Tests")
    print("=" * 60)

    print("\n1. Query Analyzer")
    test_query_analyzer_intent()
    test_query_analyzer_complexity()
    test_query_analyzer_keywords()

    print("\n2. Embedding Engine")
    test_embedding_engine()

    print("\n3. Vector Store")
    test_vector_store()

    print("\n4. End-to-End Retriever")
    test_retriever_index_and_retrieve()
    test_retriever_missing_document()
    test_retriever_document_management()
    test_retriever_persistence()

    print("\n" + "=" * 60)
    print("ALL M2 TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
