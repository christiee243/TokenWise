"""Conservative prompt cleanup with token-count and meaning safeguards."""

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

try:
    import tiktoken

    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKENIZER = None


def count_tokens(text: str) -> int:
    """Count tokens with cl100k_base, falling back to a labeled estimate."""
    if not text:
        return 0
    if _TOKENIZER is not None:
        try:
            return len(_TOKENIZER.encode(text))
        except Exception:
            pass
    return max(1, int(len(text.split()) * 1.3))


def _result(
    optimized: str,
    original_tokens: int,
    method: str,
    changes: List[str],
    semantic_warning: bool = False,
) -> Dict[str, Any]:
    optimized_tokens = count_tokens(optimized)
    tokens_saved = max(0, original_tokens - optimized_tokens)
    reduction_percent = (
        round(tokens_saved / original_tokens * 100, 1) if original_tokens else 0.0
    )
    unique_changes = list(dict.fromkeys(changes))
    return {
        "optimized": optimized,
        "original_tokens": original_tokens,
        "optimized_tokens": optimized_tokens,
        "tokens_saved": tokens_saved,
        "reduction_percent": reduction_percent,
        "changes": unique_changes[:5] or ["Prompt is already concise and direct."],
        "method": method,
        "semantic_warning": semantic_warning,
    }


def _original_result(
    text: str,
    method: str,
    changes: Optional[List[str]] = None,
    semantic_warning: bool = False,
) -> Dict[str, Any]:
    original = text.strip()
    return _result(
        original,
        count_tokens(original),
        method,
        changes or ["Prompt is already concise and direct."],
        semantic_warning,
    )


def mask_protected_spans(text: str) -> Tuple[str, List[str]]:
    """Mask code, quoted text, and URLs while applying lexical transformations."""
    spans: List[str] = []

    def stash(match: re.Match) -> str:
        spans.append(match.group(0))
        return f"\x00{len(spans) - 1}\x00"

    protected = re.compile(
        r"```.*?```|`[^`\n]*`|\"[^\"\n]*\"|(?<!\w)'[^'\n]*'(?!\w)|https?://\S+",
        re.DOTALL,
    )
    return protected.sub(stash, text), spans


def unmask_protected_spans(text: str, spans: List[str]) -> str:
    return re.sub(r"\x00(\d+)\x00", lambda match: spans[int(match.group(1))], text)


GREETING = re.compile(
    r"^(?:(?:hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening))\b[,!.\s]*"
    r"(?:(?:ai|assistant|chatgpt|model|bot|there)\b[,!.\s]*)?"
    r"|dear\s+(?:assistant|ai|chatgpt|model|bot)\b[,!.\s]*)",
    re.IGNORECASE,
)

FLUFF_OPENERS = [
    (r"^(?:can\s+you\s+(?:please\s+)?(?:kindly\s+)?(?:help\s+me\s+(?:to\s+)?)?)", "Removed polite request opener"),
    (r"^(?:could\s+you\s+(?:please\s+)?(?:kindly\s+)?(?:help\s+me\s+(?:to\s+)?)?)", "Removed polite request opener"),
    (r"^(?:would\s+you\s+(?:please\s+)?(?:kindly\s+)?(?:mind\s+)?)", "Removed polite request opener"),
    (r"^(?:will\s+you\s+(?:please\s+)?(?:kindly\s+)?)", "Removed polite request opener"),
    (r"^(?:i(?:\s+would|['’]d)\s+like\s+you\s+to\s+)", "Removed indirect request phrasing"),
    (r"^(?:i\s+(?:was\s+wondering|am\s+wondering)\s+if\s+you\s+(?:could|can)\s+)", "Removed indirect request phrasing"),
    (r"^(?:what\s+i\s+(?:need|want)\s+is\s+(?:for\s+you\s+to\s+)?)", "Removed indirect request phrasing"),
    (r"^(?:please\s+(?:kindly\s+)?)", "Removed polite filler"),
    (r"^(?:kindly\s+)", "Removed polite filler"),
    (r"^(?:help\s+me\s+(?:to\s+)?)", "Removed conversational preamble"),
]

WORDY_REPLACEMENTS = [
    (r"\bin order to\b", "to", "Replaced 'in order to' with 'to'"),
    (r"\bdue to the fact that\b", "because", "Replaced 'due to the fact that' with 'because'"),
    (r"\bat this point in time\b|\bat the present time\b", "now", "Replaced wordy time phrase with 'now'"),
    (r"\bfor the purpose of\b", "to", "Replaced 'for the purpose of' with 'to'"),
    (r"\bwith reference to\b", "regarding", "Replaced 'with reference to' with 'regarding'"),
    (r"\bin addition to this\b", "also", "Replaced 'in addition to this' with 'also'"),
    (r"\bas a matter of fact\b|\bto be honest\b|\bneedless to say\b", "", "Removed conversational filler"),
    (r"\bit is important to note that\b", "", "Removed conversational filler"),
    (r"\ba large number of\b", "many", "Replaced 'a large number of' with 'many'"),
    (r"\ba majority of\b", "most", "Replaced 'a majority of' with 'most'"),
    (r"\bhas the ability to\b|\bis able to\b", "can", "Replaced wordy ability phrase with 'can'"),
    (r"\bin the event that\b", "if", "Replaced 'in the event that' with 'if'"),
    (r"\bprior to\b", "before", "Replaced 'prior to' with 'before'"),
    (r"\bsubsequent to\b", "after", "Replaced 'subsequent to' with 'after'"),
    (r"\btake into consideration\b", "consider", "Replaced 'take into consideration' with 'consider'"),
    (r"\bgive an explanation of\b", "explain", "Replaced wordy explanation phrase"),
    (r"\bmake a decision\b", "decide", "Replaced 'make a decision' with 'decide'"),
    (r"\beach and every\b", "every", "Replaced 'each and every' with 'every'"),
    (r"\bfirst and foremost\b", "first", "Replaced 'first and foremost' with 'first'"),
]

CLOSER_PATTERNS = [
    re.compile(r"(?:(?<=[.!?,])\s+|^)(?:thanks?(?:\s+a\s+lot|\s+in\s+advance)?|thank\s+you(?:\s+very\s+much)?)[!.]?$", re.IGNORECASE),
    re.compile(r"(?:(?<=[.!?,])\s+|^)let\s+me\s+know(?:\s+if\s+you\s+need\s+anything(?:\s+else)?)?[!.]?$", re.IGNORECASE),
    re.compile(r"(?:(?<=[.!?,])\s+|^)hope\s+this\s+helps[!.]?$", re.IGNORECASE),
]

QUESTION_DIRECTIVES = [
    (r"^what is\s+(.+?)\??$", r"Explain \1."),
    (r"^what are\s+(.+?)\??$", r"Describe \1."),
    (r"^how does\s+(.+?)\s+work\??$", r"Explain how \1 works."),
    (r"^how do I\s+(.+?)\??$", r"Explain how to \1."),
    (r"^how to\s+(.+?)\??$", r"Explain how to \1."),
    (r"^why\s+(.+?)\??$", r"Explain why \1."),
]


def _is_constraint(sentence: str) -> bool:
    if sentence.rstrip().endswith("?"):
        return False
    return bool(
        re.match(
            r"^(?:do\s+not|don['’]t|never|avoid|limit|keep|must|use|include|only|"
            r"make\s+sure|ensure|be\s+sure|format|output)\b",
            sentence.strip(),
            re.IGNORECASE,
        )
    )


def _clean_constraint(sentence: str) -> str:
    cleaned = re.sub(
        r"^(?:make\s+sure\s+(?:that\s+)?|be\s+sure\s+(?:that\s+)?|"
        r"ensure\s+(?:that\s+)?|please\s+)(?:you\s+)?",
        "",
        sentence.strip(),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^[,;:]+\s*", "", cleaned)
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned.rstrip(".?! ")


def _has_list_or_paragraphs(text: str) -> bool:
    if "\n" in text:
        return True
    return bool(re.search(r"(?:^|\s)(?:[-*•]|\d+[.)])\s+", text))


def _capitalise_sentence_starts(text: str) -> str:
    def protect(match: re.Match) -> str:
        return match.group(0).replace(".", "\ue001")

    text = re.sub(r"\b(?:e\.g|i\.e|mr|mrs|ms|dr|prof|etc)\.", protect, text, flags=re.IGNORECASE)
    text = re.sub(r"^([a-z])", lambda match: match.group(1).upper(), text)
    text = re.sub(
        r"(?<=[.!?]\s)([a-z])",
        lambda match: match.group(1).upper(),
        text,
    )
    return text.replace("\ue001", ".")


def _critical_signature(text: str) -> Counter:
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    negations = re.findall(r"\b(?:not|never|no|without|only|don['’]t)\b", text, re.IGNORECASE)
    quoted = re.findall(r'"[^"\n]*"|(?<!\w)\'[^\'\n]*\'(?!\w)', text)
    return Counter([number.lower() for number in numbers + negations + quoted])


def _split_sentences(text: str) -> List[str]:
    protected = re.sub(
        r"\b(e\.g|i\.e|mr|mrs|ms|dr|prof|etc)\.",
        lambda match: match.group(0).replace(".", "\ue000"),
        text,
        flags=re.IGNORECASE,
    )
    sentences = re.split(r"(?<=[.!?])\s+", protected)
    return [sentence.replace("\ue000", ".").strip() for sentence in sentences if sentence.strip()]


def _clean_phrases(text: str, changes: List[str]) -> str:
    cleaned = GREETING.sub("", text, count=1)
    if cleaned != text:
        changes.append("Removed conversational greeting")

    for pattern, label in FLUFF_OPENERS:
        new_text = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE)
        if new_text != cleaned:
            cleaned = new_text
            changes.append(label)
            break

    for closer in CLOSER_PATTERNS:
        new_text = closer.sub("", cleaned)
        if new_text != cleaned:
            cleaned = new_text
            changes.append("Removed sentence-final conversational sign-off")

    for pattern, replacement, label in WORDY_REPLACEMENTS:
        new_text = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        if new_text != cleaned:
            cleaned = new_text
            changes.append(label)

    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r",[ \t]*,", ",", cleaned)
    cleaned = re.sub(r"[ \t]+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"^[ \t,;:]+|[ \t,;:]+$", "", cleaned)
    return cleaned


def optimize_prompt(raw_prompt: str, include_structure: bool = False) -> Dict[str, Any]:
    """Remove safe conversational padding; structure is optional and never costs tokens."""
    text = raw_prompt.strip()
    method = "tiktoken" if _TOKENIZER is not None else "approx"
    original_tokens = count_tokens(text)
    if not text:
        return _result("", 0, method, [], False)

    changes: List[str] = []
    masked, spans = mask_protected_spans(text)
    flat_cleaned = _clean_phrases(masked, changes)
    multiline_or_list = _has_list_or_paragraphs(text)

    # Lists and paragraphs receive only local phrase cleanup: preserve their layout and wording.
    if not multiline_or_list:
        sentences = _split_sentences(flat_cleaned)
        flat_cleaned = " ".join(sentences)
        flat_cleaned = _capitalise_sentence_starts(flat_cleaned)

        if len(sentences) == 1:
            for pattern, replacement in QUESTION_DIRECTIVES:
                rewritten = re.sub(pattern, replacement, flat_cleaned, flags=re.IGNORECASE)
                if rewritten != flat_cleaned:
                    flat_cleaned = rewritten
                    changes.append("Rewrote question as a direct instruction")
                    break
            if flat_cleaned.endswith("?") and re.match(
                r"^(?:summarize|describe|list|compare|analyze|write|generate|provide|show|draft|create|explain|answer|return)\b",
                flat_cleaned,
                re.IGNORECASE,
            ):
                flat_cleaned = flat_cleaned[:-1] + "."

    flat_cleaned = unmask_protected_spans(flat_cleaned, spans).strip()
    if not flat_cleaned:
        return _original_result(text, method)

    candidates = [flat_cleaned]
    if include_structure and not multiline_or_list:
        structured_changes: List[str] = []
        role = None
        body = masked
        role_pattern = re.compile(
            r"^(?:act as|you are)\s+(?:an?|the)\s+(?:expert in\s+)?([^,.]+?)"
            r"(?:\s+and\s+(?:tell me|explain|write|help|give|show|create|provide)\b|[,.;]|$)",
            re.IGNORECASE,
        )
        role_match = role_pattern.search(body)
        if role_match:
            role = role_match.group(1).strip()
            body = body[role_match.end():].strip()

        task: List[str] = []
        requirements: List[str] = []
        for sentence in _split_sentences(body):
            if _is_constraint(sentence):
                requirements.append(_clean_constraint(sentence))
            else:
                task.append(sentence)

        sections: List[str] = []
        if role:
            sections.append(f"Role: {role[0].upper() + role[1:]}")
        if task:
            sections.append(" ".join(task))
        if requirements:
            sections.append("Requirements:\n" + "\n".join(f"- {item}" for item in requirements))
        structured = unmask_protected_spans("\n\n".join(sections).strip(), spans)
        if structured:
            candidates.append(structured)
            if structured != flat_cleaned:
                structured_changes.append("Structured role and explicit requirements")
                changes.extend(structured_changes)

    best = min(candidates, key=count_tokens)
    best_tokens = count_tokens(best)
    if best_tokens >= original_tokens:
        return _original_result(text, method)

    warning = _critical_signature(text) != _critical_signature(best)
    if warning:
        return _original_result(
            text,
            method,
            changes + ["Kept original because a meaning-sensitive detail changed."],
            semantic_warning=True,
        )

    return _result(best, original_tokens, method, changes, semantic_warning=False)