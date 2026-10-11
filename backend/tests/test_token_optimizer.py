"""
TokenWise - Standalone Unit Tests for Member 3 Token Optimizer Module
"""
import unittest
import os
import sys

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from token_optimizer import TokenOptimizer, count_tokens


class TestTokenOptimizer(unittest.TestCase):

    def setUp(self):
        self.optimizer = TokenOptimizer()

    def test_count_tokens(self):
        text = "The Transformer model uses self-attention mechanisms for sequence modeling."
        tokens = count_tokens(text)
        self.assertIsInstance(tokens, int)
        self.assertGreater(tokens, 0)

    def test_estimate_budget_by_complexity(self):
        b_simple = self.optimizer.estimate_budget("factual", "simple", 5000)
        b_mod = self.optimizer.estimate_budget("factual", "moderate", 5000)
        b_comp = self.optimizer.estimate_budget("factual", "complex", 5000)

        self.assertTrue(b_simple < b_mod < b_comp)

    def test_estimate_budget_by_intent(self):
        b_factual = self.optimizer.estimate_budget("factual", "moderate", 5000)
        b_summary = self.optimizer.estimate_budget("summary", "moderate", 5000)

        self.assertTrue(b_summary > b_factual)

    def test_estimate_budget_override(self):
        b_override = self.optimizer.estimate_budget("factual", "simple", 5000, max_budget_override=800)
        self.assertEqual(b_override, 800)

    def test_remove_redundancy(self):
        chunks = [
            {"text": "The Transformer model follows an encoder-decoder architecture with self-attention mechanisms.", "tokens": 15},
            {"text": "The Transformer model follows an encoder-decoder architecture with self-attention mechanisms.", "tokens": 15},  # Duplicate
            {"text": "Positional encodings are added to input embeddings to convey word order.", "tokens": 12},
        ]

        clean, purged = self.optimizer.remove_redundancy(chunks, overlap_threshold=0.65)
        self.assertEqual(len(clean), 2)
        self.assertEqual(purged, 1)

    def test_select_chunks_within_budget(self):
        chunks = [
            {"text": "Chunk 1: Overview of deep learning architectures and neural networks.", "tokens": 10},
            {"text": "Chunk 2: Detailed explanation of attention mechanisms and query key value matrices.", "tokens": 12},
            {"text": "Chunk 3: Empirical evaluation results on translation benchmarks.", "tokens": 10},
        ]

        selected, total_tokens = self.optimizer.select_chunks_within_budget(chunks, target_budget=23)
        self.assertTrue(len(selected) >= 2)
        self.assertTrue(total_tokens <= 23)

    def test_reconstruct_markdown_context(self):
        chunks = [
            {"text": "Self-attention allows the model to jointly attend to information from different representation subspaces.", "metadata": {"section": "Model Architecture", "chunk_id": 1}, "relevance_score": 0.95},
            {"text": "Multi-head attention projects queries, keys, and values h times.", "metadata": {"section": "Attention", "chunk_id": 2}, "relevance_score": 0.88},
        ]

        md = self.optimizer.reconstruct_markdown_context(chunks, query="How does multi-head attention work?")
        self.assertIn("# Optimized Document Context", md)
        self.assertIn("## Section: Model Architecture", md)
        self.assertIn("Multi-head attention", md)

    def test_optimize_context_full_workflow(self):
        retrieved_chunks = [
            {"text": "Attention mechanisms allow dynamic context retrieval.", "tokens": 10, "metadata": {"section": "Intro", "chunk_id": 1}, "relevance_score": 0.9},
            {"text": "Attention mechanisms allow dynamic context retrieval.", "tokens": 10, "metadata": {"section": "Intro", "chunk_id": 2}, "relevance_score": 0.89},  # Redundant
            {"text": "Vector embeddings represent text in high-dimensional vector space.", "tokens": 12, "metadata": {"section": "Embeddings", "chunk_id": 3}, "relevance_score": 0.82},
        ]
        query_analysis = {"intent": "factual", "complexity": "moderate"}

        res = self.optimizer.optimize_context(
            retrieved_chunks=retrieved_chunks,
            query_analysis=query_analysis,
            total_analyzed_chunks=10,
            original_doc_tokens=1000,
            query="What are attention mechanisms?",
        )

        self.assertIn("reconstructed_markdown", res)
        self.assertIn("report", res)
        report = res["report"]

        self.assertEqual(report["original_document_tokens"], 1000)
        self.assertTrue(report["optimized_context_tokens"] > 0)
        self.assertTrue(report["tokens_saved"] > 0)
        self.assertTrue(report["token_reduction_percent"] > 0)
        self.assertEqual(report["chunks_analyzed"], 10)
        self.assertEqual(report["redundancy_purged_count"], 1)


if __name__ == "__main__":
    unittest.main()
