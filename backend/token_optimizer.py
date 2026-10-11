"""
TokenWise - Member 3: Token Optimization Engine
Implements Intent-Aware and Token-Budget-Adaptive Document Context Optimization.
Provides:
- Dynamic token budget estimation based on query intent & complexity.
- Redundancy removal and semantic deduplication.
- Budget-adaptive chunk ranking and selection.
- Structured Markdown context reconstruction.
- Comprehensive Context Efficiency Report generation.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import tiktoken

# ── Tokenizer Setup ──────────────────────────────────────────────────────────
try:
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKENIZER = None


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base) or approximate fallback."""
    if not text:
        return 0
    if _TOKENIZER:
        try:
            return len(_TOKENIZER.encode(text))
        except Exception:
            pass
    return max(1, int(len(text.split()) * 1.3))


class TokenOptimizer:
    """
    Member 3 Core Optimization Engine:
    Handles token budget estimation, redundancy removal, budget-bounded chunk selection,
    markdown context reconstruction, and context efficiency report generation.
    """

    # Base token budgets per query complexity level
    BASE_BUDGETS = {
        "simple": 500,
        "moderate": 1200,
        "complex": 2500,
    }

    # Intent multipliers for budget scaling
    INTENT_MULTIPLIERS = {
        "summary": 1.4,
        "comparison": 1.4,
        "extraction": 1.2,
        "factual": 1.0,
    }

    def estimate_budget(
        self,
        query_intent: str,
        complexity: str,
        original_doc_tokens: int,
        max_budget_override: Optional[int] = None,
    ) -> int:
        """
        Dynamically estimate context token budget based on query intent and complexity.
        """
        if max_budget_override is not None and max_budget_override > 0:
            return min(max_budget_override, max(100, original_doc_tokens if original_doc_tokens > 0 else max_budget_override))

        base = self.BASE_BUDGETS.get(complexity.lower(), 1200)
        multiplier = self.INTENT_MULTIPLIERS.get(query_intent.lower(), 1.0)
        estimated_budget = int(base * multiplier)

        # Clamp between reasonable minimum (200 tokens) and full document tokens
        if original_doc_tokens > 0:
            estimated_budget = min(estimated_budget, original_doc_tokens)
        return max(200, estimated_budget)

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute word-level Jaccard similarity between two texts."""
        words1 = set(re.findall(r"\w+", text1.lower()))
        words2 = set(re.findall(r"\w+", text2.lower()))
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)

    def remove_redundancy(
        self,
        chunks: List[Dict[str, Any]],
        overlap_threshold: float = 0.65,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Filter out semantically redundant or textually overlapping chunks.
        Returns (deduplicated_chunks, redundancy_purged_count).
        """
        unique_chunks: List[Dict[str, Any]] = []
        seen_texts: List[str] = []
        purged_count = 0

        for chunk in chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue

            # Check for overlap with already selected chunks
            is_redundant = False
            for prev_text in seen_texts:
                sim = self._jaccard_similarity(text, prev_text)
                if sim >= overlap_threshold:
                    is_redundant = True
                    break

            if is_redundant:
                purged_count += 1
            else:
                seen_texts.append(text)
                unique_chunks.append(chunk)

        return unique_chunks, purged_count

    def select_chunks_within_budget(
        self,
        chunks: List[Dict[str, Any]],
        target_budget: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Select highest-ranked chunks that fit strictly within target_budget tokens.
        Returns (selected_chunks, total_selected_tokens).
        """
        selected: List[Dict[str, Any]] = []
        accumulated_tokens = 0

        for chunk in chunks:
            text = chunk.get("text", "")
            tokens = chunk.get("tokens", 0) or count_tokens(text)

            if accumulated_tokens + tokens <= target_budget:
                selected.append(chunk)
                accumulated_tokens += tokens
            else:
                # Truncate final chunk if space remains and it's large enough to be useful
                remaining_budget = target_budget - accumulated_tokens
                if remaining_budget >= 50 and len(text) > 100:
                    words = text.split()
                    truncated_words = []
                    curr = 0
                    for w in words:
                        w_tokens = count_tokens(w)
                        if curr + w_tokens <= remaining_budget:
                            truncated_words.append(w)
                            curr += w_tokens
                        else:
                            break
                    truncated_text = " ".join(truncated_words) + "..."
                    truncated_chunk = dict(chunk)
                    truncated_chunk["text"] = truncated_text
                    truncated_chunk["tokens"] = count_tokens(truncated_text)
                    truncated_chunk["is_truncated"] = True
                    selected.append(truncated_chunk)
                    accumulated_tokens += truncated_chunk["tokens"]
                break

        return selected, accumulated_tokens

    def reconstruct_markdown_context(
        self,
        selected_chunks: List[Dict[str, Any]],
        query: str = "",
    ) -> str:
        """
        Reconstruct selected chunks into a clean, structured Markdown context document
        optimized for LLM prompt injection.
        """
        if not selected_chunks:
            return "<!-- No relevant context chunks selected within budget -->"

        lines: List[str] = [
            "# Optimized Document Context",
            f"<!-- Generated by TokenWise | Query: '{query}' -->\n",
        ]

        for idx, chunk in enumerate(selected_chunks, 1):
            meta = chunk.get("metadata", {})
            section = meta.get("section", chunk.get("section", "General"))
            score = chunk.get("relevance_score", 0.0)
            score_str = f" | Relevance: {score * 100:.1f}%" if score > 0 else ""

            lines.append(f"## Section: {section} (Chunk {idx}{score_str})")
            lines.append(f"<!-- Metadata: chunk_id={meta.get('chunk_id', idx)} -->")
            lines.append(chunk.get("text", "").strip())
            lines.append("")

        return "\n".join(lines).strip()

    def optimize_context(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        query_analysis: Dict[str, Any],
        total_analyzed_chunks: int,
        original_doc_tokens: int,
        max_budget_override: Optional[int] = None,
        query: str = "",
    ) -> Dict[str, Any]:
        """
        Master workflow for Member 3:
        1. Estimates dynamic token budget based on query intent & complexity.
        2. Applies semantic deduplication to remove redundancy.
        3. Ranks & selects optimal chunks within token budget.
        4. Reconstructs structured Markdown context.
        5. Generates Context Efficiency Report metrics.
        """
        intent = query_analysis.get("intent", "factual")
        complexity = query_analysis.get("complexity", "moderate")

        # 1. Budget estimation
        target_budget = self.estimate_budget(
            query_intent=intent,
            complexity=complexity,
            original_doc_tokens=original_doc_tokens,
            max_budget_override=max_budget_override,
        )

        # Calculate initial retrieved tokens from M2
        retrieved_tokens = sum(
            c.get("tokens", 0) or count_tokens(c.get("text", ""))
            for c in retrieved_chunks
        )

        # 2. Redundancy Removal & Deduplication
        clean_chunks, redundancy_purged = self.remove_redundancy(retrieved_chunks)

        # 3. Budget-Adaptive Chunk Selection
        selected_chunks, final_context_tokens = self.select_chunks_within_budget(
            clean_chunks, target_budget
        )

        # 4. Structured Markdown Reconstruction
        reconstructed_markdown = self.reconstruct_markdown_context(
            selected_chunks, query=query
        )
        actual_markdown_tokens = count_tokens(reconstructed_markdown)

        # 5. Efficiency Report Metrics calculation
        tokens_saved = max(0, original_doc_tokens - actual_markdown_tokens)
        reduction_percent = (
            round((tokens_saved / original_doc_tokens) * 100, 1)
            if original_doc_tokens > 0
            else 0.0
        )
        compression_ratio = (
            round(original_doc_tokens / actual_markdown_tokens, 2)
            if actual_markdown_tokens > 0
            else 1.0
        )

        return {
            "reconstructed_markdown": reconstructed_markdown,
            "selected_chunks": selected_chunks,
            "report": {
                "original_document_tokens": original_doc_tokens,
                "retrieved_chunks_tokens": retrieved_tokens,
                "optimized_context_tokens": actual_markdown_tokens,
                "tokens_saved": tokens_saved,
                "token_reduction_percent": reduction_percent,
                "compression_ratio": compression_ratio,
                "target_budget": target_budget,
                "chunks_analyzed": total_analyzed_chunks,
                "chunks_retrieved": len(retrieved_chunks),
                "chunks_selected": len(selected_chunks),
                "redundancy_purged_count": redundancy_purged,
                "query_intent": intent,
                "query_complexity": complexity,
            },
        }


# Global singleton instance for Member 3
token_optimizer = TokenOptimizer()
