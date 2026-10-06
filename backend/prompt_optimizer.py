"""
TokenWise - Member 1: Prompt Optimizer Module
Refactors user prompts to eliminate conversational fluff, remove repetition,
and restructure into clear, token-efficient system/user instructions with token statistics.
"""

import re
from typing import Any, Dict, List, Tuple
import tiktoken

try:
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKENIZER = None


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base), or approximate fallback."""
    if not text:
        return 0
    if _TOKENIZER:
        try:
            return len(_TOKENIZER.encode(text))
        except Exception:
            pass
    return max(1, int(len(text.split()) * 1.3))


# Common conversational fluff patterns to strip
FLUFF_PATTERNS = [
    (r"\b(?:hello|hi|hey|dear)\s+(?:assistant|ai|chatgpt|model|bot|there)[,.]?\s*", ""),
    (r"\b(?:can you please|could you please|would you please|please)\s+", ""),
    (r"\bi would like you to\s+", ""),
    (r"\bi want you to\s+", ""),
    (r"\bact as an? expert in\b", "Role: Expert in"),
    (r"\bact as an?\b", "Role:"),
    (r"\byou are an? expert in\b", "Role: Expert in"),
    (r"\bi am trying to\s+", "Goal: "),
    (r"\bwhat i need is for you to\s+", ""),
    (r"\bmake sure that you\s+", "Must "),
    (r"\bmake sure to\s+", "Must "),
    (r"\bbe sure to\s+", "Must "),
    (r"\bdo not forget to\s+", "Include: "),
    (r"\bin order to\b", "to"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bat this point in time\b", "now"),
    (r"\bfor the purpose of\b", "for"),
    (r"\bwith reference to\b", "regarding"),
    (r"\bin addition to this\b", "also"),
    (r"\bas a matter of fact\b", ""),
    (r"\bto be honest\b", ""),
    (r"\bneedless to say\b", ""),
    (r"\bkindly\s+", ""),
]


def optimize_prompt(raw_prompt: str) -> Dict[str, Any]:
    """
    Refactor user prompt:
    1. Removes conversational padding and redundant phrases.
    2. Identifies key tasks, constraints, and desired output formats.
    3. Restructures into concise, structured prompt directives.
    4. Computes exact token reduction metrics.
    """
    text = raw_prompt.strip()
    original_tokens = count_tokens(text)
    if not text:
        return {
            "optimized": "",
            "original_tokens": 0,
            "optimized_tokens": 0,
            "tokens_saved": 0,
            "reduction_percent": 0.0,
            "changes": [],
        }

    changes_applied: List[str] = []
    cleaned = text

    # Strip conversational fluff
    for pattern, replacement in FLUFF_PATTERNS:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
            changes_applied.append(f"Removed fluff / verbose phrasing: '{match.group(0).strip()}'")

    # Deduplicate consecutive repeated words (e.g. "very very", "the the")
    cleaned = re.sub(r'\b(\w+)\s+\1\b', r'\1', cleaned, flags=re.IGNORECASE)

    # Clean whitespace
    cleaned = re.sub(r'[ \t]+', ' ', cleaned).strip()

    # Identify constraints or requirements mentioned in prompt
    constraints = []
    task_parts = []

    sentences = re.split(r'(?<=[.!?\n])\s+', cleaned)
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        s_lower = s_clean.lower()
        if any(kw in s_lower for kw in ["must", "format as", "output format", "limit to", "within", "only include", "do not"]):
            constraints.append(s_clean.rstrip('.'))
        else:
            task_parts.append(s_clean)

    task_summary = " ".join(task_parts).strip()
    if not task_summary and constraints:
        task_summary = constraints.pop(0)

    # Build structured, high-signal prompt
    output_lines = [f"Task: {task_summary}"]

    # Add core guidelines / requirements
    reqs = [
        "Be concise, accurate, and direct.",
        "Omit conversational filler and unnecessary preamble."
    ]
    if constraints:
        reqs.extend(constraints)

    output_lines.append("\nRequirements:")
    for r in reqs:
        clean_r = r.lstrip('-*• ').strip()
        output_lines.append(f"- {clean_r}")

    optimized_text = "\n".join(output_lines)
    optimized_tokens = count_tokens(optimized_text)

    # Calculate token reduction
    if optimized_tokens >= original_tokens and original_tokens > 10:
        # If the structured format is longer than a short 1-line query, provide direct condensed command
        optimized_text = f"Direct instruction: {task_summary}\nFormat: Concise, structured output."
        optimized_tokens = count_tokens(optimized_text)

    tokens_saved = max(0, original_tokens - optimized_tokens)
    reduction_percent = (
        round((tokens_saved / original_tokens) * 100, 1) if original_tokens > 0 else 0.0
    )

    if not changes_applied:
        changes_applied.append("Restructured query into imperative prompt with explicit constraints.")

    return {
        "optimized": optimized_text,
        "original_tokens": original_tokens,
        "optimized_tokens": optimized_tokens,
        "tokens_saved": tokens_saved,
        "reduction_percent": reduction_percent,
        "changes": changes_applied[:4],
    }
