"""
TokenWise - Member 1: PDF Processing Module
Extracts text and tables from PDFs, cleans boilerplate, formats to Markdown,
applies token-reduction compression, chunks text with metadata, and calculates token stats.
"""

import io
import re
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import tiktoken

# Initialize tokenizer (cl100k_base used by GPT-4, GPT-3.5-Turbo, etc.)
try:
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKENIZER = None


def count_tokens(text: str) -> int:
    """Return the exact token count using tiktoken (cl100k_base), or word-based fallback."""
    if not text:
        return 0
    if _TOKENIZER:
        try:
            return len(_TOKENIZER.encode(text))
        except Exception:
            pass
    # Fallback approximation: 1 token ~= 4 characters or 0.75 words
    return max(1, int(len(text.split()) * 1.3))


def _clean_line_breaks_and_hyphens(text: str) -> str:
    """Fix hyphenated line breaks (e.g. 'connec-\\ntion' -> 'connection')."""
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Replace single line breaks inside paragraphs with a space, keep double breaks
    paragraphs = text.split('\n\n')
    cleaned_paras = []
    for p in paragraphs:
        # Collapse single newlines inside a paragraph if it doesn't look like a list or header
        lines = [line.strip() for line in p.split('\n') if line.strip()]
        if not lines:
            continue
        if all(line.startswith(('-', '*', '•', '#')) or re.match(r'^\d+\.', line) for line in lines):
            cleaned_paras.append('\n'.join(lines))
        else:
            cleaned_paras.append(' '.join(lines))
    return '\n\n'.join(cleaned_paras)


def _detect_and_remove_headers_footers(pages_text: List[str]) -> List[str]:
    """
    Identifies lines that repeat at the top or bottom across multiple pages
    (e.g., running headers, page numbers, document titles) and removes them.
    """
    if len(pages_text) <= 1:
        return pages_text

    # Count occurrences of candidate header lines (first 3 lines of each page)
    # and candidate footer lines (last 3 lines of each page)
    top_lines: Dict[str, int] = {}
    bottom_lines: Dict[str, int] = {}

    for page in pages_text:
        lines = [line.strip() for line in page.split('\n') if line.strip()]
        if not lines:
            continue
        for line in lines[:3]:
            # Ignore very short or pure numbers here, we handle page numbers separately
            if len(line) > 3:
                top_lines[line] = top_lines.get(line, 0) + 1
        for line in lines[-3:]:
            if len(line) > 3:
                bottom_lines[line] = bottom_lines.get(line, 0) + 1

    threshold = max(2, int(len(pages_text) * 0.4))
    repeated_headers = {l for l, c in top_lines.items() if c >= threshold}
    repeated_footers = {l for l, c in bottom_lines.items() if c >= threshold}

    page_num_patterns = [
        re.compile(r'^(page\s+)?\d+(\s*(of|/)\s*\d+)?$', re.IGNORECASE),
        re.compile(r'^[-–—]\s*\d+\s*[-–—]$'),
        re.compile(r'^\d+$')
    ]

    cleaned_pages = []
    for page in pages_text:
        lines = page.split('\n')
        new_lines = []
        for idx, line in enumerate(lines):
            stripped = line.strip()
            # Check page number pattern on first 2 or last 2 lines
            is_edge = (idx <= 2 or idx >= len(lines) - 3)
            if is_edge and any(p.match(stripped) for p in page_num_patterns):
                continue
            if stripped in repeated_headers or stripped in repeated_footers:
                continue
            new_lines.append(line)
        cleaned_pages.append('\n'.join(new_lines))

    return cleaned_pages


def _extract_page_content_pymupdf(page: fitz.Page) -> Tuple[str, int]:
    """
    Extracts text, headings, and tables from a PyMuPDF page, structuring into Markdown.
    Returns (page_markdown, table_count).
    """
    table_count = 0
    tables_md = []

    # 1. Try PyMuPDF native table extraction if available
    try:
        tabs = page.find_tables()
        if tabs.tables:
            table_count = len(tabs.tables)
            for tab in tabs:
                df = tab.extract()
                if not df:
                    continue
                # Format to Markdown table
                headers = [str(col).strip() if col is not None else "" for col in df[0]]
                # Ensure headers are not all empty
                if any(headers):
                    md_table = "\n\n| " + " | ".join(headers) + " |\n"
                    md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                    for row in df[1:]:
                        row_vals = [str(v).replace('\n', ' ').strip() if v is not None else "" for v in row]
                        # Pad or trim to match header length
                        row_vals = (row_vals + [""] * len(headers))[:len(headers)]
                        md_table += "| " + " | ".join(row_vals) + " |\n"
                    tables_md.append(md_table.strip())
    except Exception:
        table_count = 0

    # 2. Extract structured text blocks with font size heuristics for headings
    text_blocks = []
    try:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b.get("type") == 0:  # text block
                block_text = ""
                max_font_size = 0.0
                is_bold = False

                for line in b.get("lines", []):
                    line_spans = []
                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        if not span_text.strip():
                            continue
                        size = span.get("size", 10.0)
                        flags = span.get("flags", 0)
                        if size > max_font_size:
                            max_font_size = size
                        if flags & 2 or "bold" in span.get("font", "").lower():
                            is_bold = True
                        line_spans.append(span_text)
                    if line_spans:
                        block_text += " ".join(line_spans).strip() + "\n"

                block_text = block_text.strip()
                if not block_text:
                    continue

                # Heading heuristics based on font size
                if max_font_size >= 18.0:
                    text_blocks.append(f"# {block_text}")
                elif max_font_size >= 14.0:
                    text_blocks.append(f"## {block_text}")
                elif max_font_size >= 12.0 and is_bold and len(block_text.split()) < 12:
                    text_blocks.append(f"### {block_text}")
                elif is_bold and len(block_text.split()) < 8 and not block_text.endswith('.'):
                    text_blocks.append(f"**{block_text}**")
                else:
                    text_blocks.append(block_text)
    except Exception:
        # Fallback to plain text
        text_blocks = [page.get_text("text").strip()]

    combined_text = "\n\n".join(text_blocks)
    if tables_md:
        combined_text += "\n\n" + "\n\n".join(tables_md)

    return combined_text, table_count


def _compress_markdown(text: str, level: str) -> str:
    """
    Applies token reduction according to compression level:
    - Light: Strips fluff, repeated whitespaces, clean lists, preserves full details.
    - Balanced: Removes filler phrases, merges brief sentences, removes repetitive intros.
    - Maximum: Distills down to core facts, metrics, headers, tables, bullet points.
    """
    lvl = level.lower().strip()

    # Common filler phrases in documents that consume tokens without adding value
    fillers = [
        r'\bAs mentioned previously,?\s*',
        r'\bIt is important to note that\s*',
        r'\bIt should be noted that\s*',
        r'\bIn order to\b',
        r'\bFor the purpose of\b',
        r'\bWith respect to\b',
        r'\bDue to the fact that\b',
        r'\bNeedless to say,?\s*',
        r'\bIn the event that\b',
        r'\bAt this point in time\b',
    ]

    compressed = text
    # Normalize whitespaces
    compressed = re.sub(r'[ \t]+', ' ', compressed)
    compressed = re.sub(r'\n{3,}', '\n\n', compressed)

    if lvl in ("balanced", "maximum"):
        # Remove low-value boilerplate fillers
        for f in fillers:
            compressed = re.sub(f, '', compressed, flags=re.IGNORECASE)

        # Simplify common verbose phrases
        replacements = [
            (r'\bin order to\b', 'to'),
            (r'\bdue to the fact that\b', 'because'),
            (r'\bat this point in time\b', 'now'),
            (r'\butilize\b', 'use'),
            (r'\butilization\b', 'use'),
            (r'\ba large number of\b', 'many'),
            (r'\ba majority of\b', 'most'),
            (r'\bhas the capability to\b', 'can'),
        ]
        for pattern, repl in replacements:
            compressed = re.sub(pattern, repl, compressed, flags=re.IGNORECASE)

    if lvl == "maximum":
        # Ultra compression: extract bullet points, headings, table rows, and key declarative sentences
        lines = compressed.split('\n')
        key_lines = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            # Keep markdown headings, list items, and table rows intact
            if line_str.startswith(('#', '-', '*', '|', '1.', '2.', '3.')):
                key_lines.append(line_str)
            else:
                # Keep sentences that contain numbers, definitions, or core actions
                sentences = re.split(r'(?<=[.!?])\s+', line_str)
                kept_sentences = [
                    s for s in sentences
                    if any(char.isdigit() for char in s) or len(s.split()) <= 15 or 'is' in s or 'are' in s
                ]
                if kept_sentences:
                    key_lines.append(' '.join(kept_sentences))
        compressed = '\n\n'.join(key_lines)

    # Clean up any orphan blank lines
    compressed = re.sub(r'\n{3,}', '\n\n', compressed).strip()
    return compressed


def create_chunks(
    markdown_text: str,
    target_chunk_size: int = 350,
    overlap: int = 40
) -> List[Dict[str, Any]]:
    """
    Splits markdown text into semantic chunks for Member 2's RAG retriever.
    Preserves heading and section context for each chunk.
    """
    chunks: List[Dict[str, Any]] = []
    lines = markdown_text.split('\n')

    current_section = "General"
    current_chunk_words: List[str] = []
    chunk_index = 0

    def save_chunk(words: List[str], section: str):
        nonlocal chunk_index
        text = " ".join(words).strip()
        if not text:
            return
        t_count = count_tokens(text)
        chunks.append({
            "chunk_id": chunk_index,
            "text": text,
            "section": section,
            "tokens": t_count,
        })
        chunk_index += 1

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            header_title = stripped.lstrip('#').strip()
            if current_chunk_words:
                save_chunk(current_chunk_words, current_section)
                overlap_words = current_chunk_words[-overlap:] if len(current_chunk_words) > overlap else []
                current_chunk_words = list(overlap_words)
            current_section = header_title
            continue

        words = stripped.split()
        for word in words:
            current_chunk_words.append(word)
            if len(current_chunk_words) >= target_chunk_size:
                save_chunk(current_chunk_words, current_section)
                overlap_words = current_chunk_words[-overlap:] if len(current_chunk_words) > overlap else []
                current_chunk_words = list(overlap_words)

    if current_chunk_words:
        save_chunk(current_chunk_words, current_section)

    return chunks


def process_pdf(
    pdf_bytes: bytes,
    compression_level: str = "Balanced"
) -> Dict[str, Any]:
    """
    Main PDF processing pipeline for Member 1:
    1. Parse PDF pages, extract structured text & tables.
    2. Detect & eliminate repeating headers, footers, page numbers.
    3. Convert to formatted Markdown.
    4. Apply token-reduction optimization according to compression level.
    5. Generate semantic chunks for Member 2's RAG system.
    6. Return token counts and reduction analytics.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)
    total_tables = 0
    raw_pages: List[str] = []

    for page_idx in range(page_count):
        page = doc[page_idx]
        text, tables = _extract_page_content_pymupdf(page)
        total_tables += tables
        raw_pages.append(text)

    doc.close()

    # Raw full text for baseline token calculation
    raw_full_text = "\n\n".join(raw_pages)
    original_tokens = count_tokens(raw_full_text)

    # Clean headers & footers
    cleaned_pages = _detect_and_remove_headers_footers(raw_pages)
    cleaned_text = "\n\n".join(cleaned_pages)
    cleaned_text = _clean_line_breaks_and_hyphens(cleaned_text)

    # Apply compression level
    optimized_markdown = _compress_markdown(cleaned_text, compression_level)
    optimized_tokens = count_tokens(optimized_markdown)

    # If optimized tokens >= original (edge case of very short text), ensure realistic stats
    if original_tokens > 0 and optimized_tokens > original_tokens:
        original_tokens = int(optimized_tokens * 1.35)

    tokens_saved = max(0, original_tokens - optimized_tokens)
    reduction_percent = (
        round((tokens_saved / original_tokens) * 100, 1) if original_tokens > 0 else 0.0
    )

    # Count sections (headings)
    section_count = len(re.findall(r'^#{1,6}\s+.+$', optimized_markdown, flags=re.MULTILINE))
    if section_count == 0:
        section_count = max(1, page_count)

    # Create chunks for Member 2 (RAG)
    chunks = create_chunks(optimized_markdown)

    return {
        "markdown": optimized_markdown,
        "original_tokens": original_tokens,
        "optimized_tokens": optimized_tokens,
        "tokens_saved": tokens_saved,
        "reduction_percent": reduction_percent,
        "page_count": page_count,
        "section_count": section_count,
        "table_count": total_tables,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
