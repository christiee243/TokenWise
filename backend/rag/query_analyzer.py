"""
TokenWise RAG Module — Query Analyzer
Classifies user queries by intent, complexity, and extracts keywords
to dynamically tune retrieval parameters.
"""

import re
from dataclasses import dataclass, field
from typing import List

from .config import STOP_WORDS, TOP_K_BY_COMPLEXITY


@dataclass
class QueryAnalysis:
    """Result of analyzing a user query."""

    intent: str                    # "factual", "summary", "comparison", "extraction"
    complexity: str                # "simple", "moderate", "complex"
    suggested_top_k: int           # Dynamic top_k based on complexity
    keywords: List[str] = field(default_factory=list)  # Key terms from query


class QueryAnalyzer:
    """
    Heuristic query analyzer that determines:
    - Intent: what kind of answer the user expects
    - Complexity: how many chunks are likely needed
    - Keywords: salient terms for potential re-ranking
    """

    # ── Complexity Signal Phrases ────────────────────────────────────────
    COMPLEXITY_SIGNALS = {
        "simple": [
            "what is", "define", "who is", "when did", "when was",
            "where is", "where was", "name the", "which",
        ],
        "moderate": [
            "explain", "describe", "how does", "how do", "how is",
            "why does", "why do", "why is", "what are the",
            "tell me about", "elaborate",
        ],
        "complex": [
            "compare", "contrast", "analyze", "evaluate", "assess",
            "relationship between", "impact of", "implications of",
            "advantages and disadvantages", "pros and cons",
            "critically", "in-depth", "comprehensive",
        ],
    }

    # ── Intent Signal Words ──────────────────────────────────────────────
    INTENT_SIGNALS = {
        "summary": [
            "summarize", "summary", "overview", "gist", "brief",
            "outline", "recap", "tldr", "tl;dr",
        ],
        "comparison": [
            "compare", "comparison", "difference", "differences",
            "versus", " vs ", " vs.", "distinguish", "contrast",
        ],
        "extraction": [
            "extract", "list", "find all", "give me", "enumerate",
            "show me", "what are all", "identify all", "gather",
        ],
    }

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze a user query and return structured classification.

        Args:
            query: The raw user query string.

        Returns:
            QueryAnalysis with intent, complexity, suggested_top_k, and keywords.
        """
        query_lower = query.lower().strip()

        complexity = self._classify_complexity(query_lower)
        intent = self._classify_intent(query_lower)
        suggested_top_k = TOP_K_BY_COMPLEXITY.get(complexity, 5)
        keywords = self._extract_keywords(query_lower)

        return QueryAnalysis(
            intent=intent,
            complexity=complexity,
            suggested_top_k=suggested_top_k,
            keywords=keywords,
        )

    def _classify_complexity(self, query: str) -> str:
        """Determine query complexity from signal phrases."""
        # Check complex first (most specific), then simple, default moderate
        for level in ("complex", "simple"):
            signals = self.COMPLEXITY_SIGNALS[level]
            if any(signal in query for signal in signals):
                return level
        # Check moderate signals explicitly
        if any(signal in query for signal in self.COMPLEXITY_SIGNALS["moderate"]):
            return "moderate"
        # Default: moderate for anything not matched
        return "moderate"

    def _classify_intent(self, query: str) -> str:
        """Classify the primary intent of the query."""
        for intent, signals in self.INTENT_SIGNALS.items():
            if any(signal in query for signal in signals):
                return intent
        return "factual"

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract meaningful keywords by removing stop words and
        keeping tokens with 3+ characters.
        """
        # Remove punctuation except hyphens within words
        cleaned = re.sub(r"[^\w\s-]", " ", query)
        tokens = cleaned.split()
        keywords = [
            token
            for token in tokens
            if token not in STOP_WORDS and len(token) > 2
        ]
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        return unique
