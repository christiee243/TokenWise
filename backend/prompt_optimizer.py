"""
TokenWise - Member 1: Prompt Optimizer Module
Refactors user prompts to eliminate conversational fluff, strip repetitive padding,
and restructure into clear, token-efficient directives with accurate token statistics.
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


# Salutations & Greetings to strip from start
GREETING_PATTERNS = [
    (r"^(?:hello|hi|hey|dear)\s+(?:assistant|ai|chatgpt|model|bot|there)[,.]?\s*", "greeting"),
    (r"^(?:hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening))[,!.]?\s*", "greeting"),
]

# Fluff / conversational openers to strip
FLUFF_OPENERS = [
    (r"^(?:can\s+you\s+(?:please\s+)?(?:kindly\s+)?(?:help\s+me\s+(?:to\s+)?)?)", "polite question opener"),
    (r"^(?:could\s+you\s+(?:please\s+)?(?:kindly\s+)?(?:help\s+me\s+(?:to\s+)?)?)", "polite question opener"),
    (r"^(?:would\s+you\s+(?:please\s+)?(?:kindly\s+)?(?:mind\s+)?)", "polite question opener"),
    (r"^(?:will\s+you\s+(?:please\s+)?(?:kindly\s+)?)", "polite question opener"),
    (r"^(?:i\s+(?:would\s+like|'d\s+like|want|need)\s+you\s+to\s+)", "indirect request phrasing"),
    (r"^(?:i\s+(?:was\s+wondering|am\s+wondering)\s+if\s+you\s+(?:could|can)\s+)", "indirect request phrasing"),
    (r"^(?:what\s+i\s+(?:need|want)\s+is\s+(?:for\s+you\s+to\s+)?)", "indirect request phrasing"),
    (r"^(?:do\s+your\s+best\s+to\s+)", "fluff preamble"),
    (r"^(?:try\s+your\s+best\s+to\s+)", "fluff preamble"),
    (r"^(?:please\s+(?:kindly\s+)?)", "polite filler"),
    (r"^(?:kindly\s+)", "polite filler"),
    (r"^(?:help\s+me\s+(?:to\s+)?)", "conversational preamble"),
]

# Verbose conversational phrases to compress
WORDY_REPLACEMENTS = [
    (r"\bin order to\b", "to", "in order to -> to"),
    (r"\bdue to the fact that\b", "because", "due to the fact that -> because"),
    (r"\bat this point in time\b", "now", "at this point in time -> now"),
    (r"\bat the present time\b", "now", "at the present time -> now"),
    (r"\bfor the purpose of\b", "to", "for the purpose of -> to"),
    (r"\bwith reference to\b", "regarding", "with reference to -> regarding"),
    (r"\bin addition to this\b", "also", "in addition to this -> also"),
    (r"\bas a matter of fact\b", "", "removed filler 'as a matter of fact'"),
    (r"\bto be honest\b", "", "removed filler 'to be honest'"),
    (r"\bneedless to say\b", "", "removed filler 'needless to say'"),
    (r"\bit goes without saying that\b", "", "removed filler 'it goes without saying'"),
    (r"\bit is important to note that\b", "", "removed filler 'it is important to note that'"),
    (r"\bas much as possible\b", "", "removed filler 'as much as possible'"),
    (r"\ba large number of\b", "many", "a large number of -> many"),
    (r"\ba majority of\b", "most", "a majority of -> most"),
    (r"\bhas the ability to\b", "can", "has the ability to -> can"),
    (r"\bis able to\b", "can", "is able to -> can"),
    (r"\bin the event that\b", "if", "in the event that -> if"),
    (r"\bprior to\b", "before", "prior to -> before"),
    (r"\bsubsequent to\b", "after", "subsequent to -> after"),
    (r"\btake into consideration\b", "consider", "take into consideration -> consider"),
    (r"\bgive an explanation of\b", "explain", "give an explanation of -> explain"),
    (r"\bmake a decision\b", "decide", "make a decision -> decide"),
    (r"\beach and every\b", "every", "each and every -> every"),
    (r"\bfirst and foremost\b", "first", "first and foremost -> first"),
    (r"\bfeel free to\b", "", "removed 'feel free to'"),
]

# Conversational closers to strip
CLOSER_PATTERNS = [
    r"[,.]?\s*(?:thank\s+you(?:\s+very\s+much)?|thanks(?:\s+a\s+lot|\s+in\s+advance)?)[!.]?$",
    r"[,.]?\s*let\s+me\s+know(?:\s+if\s+you\s+need\s+anything(?:\s+else)?)?[!.]?$",
    r"[,.]?\s*hope\s+this\s+helps[!.]?$",
]

QUESTION_DIRECTIVES = [
    (r"^what is\s+(.+?)\??$", r"Explain \1.", "Converted question to direct explanation directive"),
    (r"^what are\s+(.+?)\??$", r"Describe \1.", "Converted question to direct description directive"),
    (r"^how does\s+(.+?)\s+work\??$", r"Explain how \1 works.", "Converted question to direct explanation directive"),
    (r"^how do I\s+(.+?)\??$", r"Explain how to \1.", "Converted question to direct instruction"),
    (r"^how to\s+(.+?)\??$", r"Explain how to \1.", "Converted question to direct instruction"),
    (r"^how\s+(.+?)\??$", r"Explain how \1.", "Converted question to direct instruction"),
    (r"^why\s+(.+?)\??$", r"Explain why \1.", "Converted question to direct explanation directive"),
    (r"^(?:tell me\s+(?:about|how to|what is)?)\s+(.+?)\??$", r"Explain \1.", "Converted conversational request to direct directive"),
    (r"^(?:give me\s+(?:a\s+list\s+of)?)\s+(.+?)\??$", r"List \1.", "Converted conversational request to direct list directive"),
]

CONSTRAINT_KEYWORDS = [
    "must", "should", "format as", "output as", "output format", "limit to",
    "within", "only include", "do not", "avoid", "make sure that", "make sure to",
    "be sure to", "ensure that", "ensure to", "strictly"
]


def optimize_prompt(raw_prompt: str) -> Dict[str, Any]:
    """
    Refactor user prompt:
    1. Removes conversational padding, polite fillers, and repetitive phrases.
    2. Sharpens directives into concise, imperative instructions without artificial tags.
    3. Identifies explicit constraints/requirements and structures them cleanly.
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

    # 1. Strip greetings
    for pattern, label in GREETING_PATTERNS:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            cleaned = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE).strip()
            changes_applied.append(f"Removed conversational {label}: '{match.group(0).strip()}'")

    # 2. Strip fluff openers (can you please, could you please, etc.)
    stripped_opener = True
    while stripped_opener:
        stripped_opener = False
        for pattern, label in FLUFF_OPENERS:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                matched_str = match.group(0).strip()
                cleaned = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE).strip()
                changes_applied.append(f"Removed {label}: '{matched_str}'")
                stripped_opener = True

    # 3. Extract Persona / Role if present
    role_extracted = None
    role_pattern = r"^(?:act as|you are)\s+(?:an?|the)\s+([^,.\n]+?)(?:\s+(?:and|,)\s+(?:tell me\s+|explain\s+|write\s+|help\s+|give\s+|show\s+|create\s+|provide\s+)|\s*[,.]\s*|\s*\n)"
    role_match = re.search(role_pattern, cleaned, flags=re.IGNORECASE)
    if role_match:
        role_name = role_match.group(1).strip()
        role_extracted = role_name[0].upper() + role_name[1:] if role_name else None
        cleaned = cleaned[role_match.end():].strip()
        changes_applied.append(f"Extracted system role: '{role_extracted}'")
    else:
        standalone_role = r"^(?:act as|you are)\s+(?:an?|the)\s+expert in\s+([^,.\n]+?)(?:\s+and\s+|\s*[,.]\s*)"
        standalone_match = re.search(standalone_role, cleaned, flags=re.IGNORECASE)
        if standalone_match:
            role_extracted = f"Expert in {standalone_match.group(1).strip()}"
            cleaned = cleaned[standalone_match.end():].strip()
            changes_applied.append(f"Extracted system role: '{role_extracted}'")

    # 4. Strip conversational closers
    for pattern in CLOSER_PATTERNS:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
            changes_applied.append("Removed conversational sign-off")

    # 5. Compress wordy phrases
    for pattern, replacement, label in WORDY_REPLACEMENTS:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
            changes_applied.append(label)

    # 6. Deduplicate consecutive repeated words (e.g. "very very", "the the")
    dup_match = re.search(r'\b(\w+)\s+\1\b', cleaned, flags=re.IGNORECASE)
    if dup_match:
        cleaned = re.sub(r'\b(\w+)\s+\1\b', r'\1', cleaned, flags=re.IGNORECASE)
        changes_applied.append("Removed consecutive duplicate words")

    # Clean multi-spaces
    cleaned = re.sub(r'[ \t]+', ' ', cleaned).strip()

    # 7. Identify constraints vs main instructions across sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', cleaned) if s.strip()]
    task_sentences = []
    extracted_constraints = []

    for s in sentences:
        s_lower = s.lower()
        is_constraint = False
        for kw in CONSTRAINT_KEYWORDS:
            if kw in s_lower:
                is_constraint = True
                break

        if is_constraint and len(sentences) > 1:
            c_clean = s.rstrip('.!?')
            c_clean = re.sub(r'^(?:it\s+must|must\s+be|must|make sure that you|make sure to|be sure to|ensure that you|ensure that|ensure to|please)\s+', '', c_clean, flags=re.IGNORECASE)
            c_clean = c_clean[0].upper() + c_clean[1:] if c_clean else c_clean
            extracted_constraints.append(c_clean)
        else:
            task_sentences.append(s)

    main_instruction = " ".join(task_sentences).strip()
    if not main_instruction and extracted_constraints:
        main_instruction = extracted_constraints.pop(0)

    # 8. Convert questions or polite requests into direct imperative commands
    for pattern, replacement, label in QUESTION_DIRECTIVES:
        rewritten = re.sub(pattern, replacement, main_instruction, flags=re.IGNORECASE)
        if rewritten != main_instruction:
            main_instruction = rewritten
            changes_applied.append(label)
            break

    # If instruction was a polite question ending in '?' (e.g. "write a python script to parse csv files?"),
    # convert trailing '?' to '.'
    if main_instruction.endswith("?") and not any(main_instruction.lower().startswith(q) for q in ["what", "how", "why", "who", "where", "when"]):
        main_instruction = main_instruction.rstrip("?").strip() + "."
        changes_applied.append("Reframed question into direct imperative sentence")

    # Capitalize the first letter of main instruction
    if main_instruction:
        main_instruction = main_instruction[0].upper() + main_instruction[1:]
        if not main_instruction.endswith((".", "!", "?")):
            main_instruction += "."

    # 9. Format final optimized text (NO arbitrary "Task:" prefix!)
    output_sections: List[str] = []

    if role_extracted:
        output_sections.append(f"Role: {role_extracted}")

    if main_instruction:
        output_sections.append(main_instruction)

    if extracted_constraints:
        changes_applied.append("Extracted explicit constraints into structured requirements")
        req_lines = ["Requirements:"]
        for c in extracted_constraints:
            req_lines.append(f"- {c.lstrip('-*• ').strip()}")
        output_sections.append("\n".join(req_lines))

    optimized_text = "\n\n".join(output_sections).strip() or cleaned

    # 10. Check if prompt was already concise
    optimized_tokens = count_tokens(optimized_text)
    tokens_saved = max(0, original_tokens - optimized_tokens)
    reduction_percent = (
        round((tokens_saved / original_tokens) * 100, 1) if original_tokens > 0 else 0.0
    )

    if not changes_applied or (optimized_text.lower().rstrip('. ') == text.lower().rstrip('. ') and tokens_saved == 0):
        changes_applied = ["Prompt is already concise and direct."]
        optimized_text = text[0].upper() + text[1:] if text else text
        optimized_tokens = original_tokens
        tokens_saved = 0
        reduction_percent = 0.0

    return {
        "optimized": optimized_text,
        "original_tokens": original_tokens,
        "optimized_tokens": optimized_tokens,
        "tokens_saved": tokens_saved,
        "reduction_percent": reduction_percent,
        "changes": changes_applied[:5],
    }
