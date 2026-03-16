"""
utils/pdf_parser.py
────────────────────
WHY THIS FILE EXISTS:
  PDF files are binary. We can't feed raw bytes to an LLM.
  This module handles everything needed to turn a pitch deck PDF
  into clean, usable text:

    1. Extract text page by page using pdfplumber
    2. Clean up whitespace, headers/footers, junk characters
    3. Chunk the text into ~2000-token pieces (so large decks
       don't overflow the LLM context window)
    4. Return both the full text and the chunked list

HOW CHUNKING WORKS:
  We split on double newlines (paragraph boundaries) rather than
  fixed character counts. This preserves semantic units — a bullet
  point won't be split in the middle of a sentence.

FALLBACK:
  If pdfplumber can't extract text (e.g. scanned/image-based PDF),
  we return an empty string and log a warning. In a production system
  you'd add OCR here (e.g. Tesseract via pytesseract).
"""

import re
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("pdfplumber not installed. PDF parsing disabled.")


def extract_text_from_pdf(filepath: str) -> Tuple[str, int]:
    """
    Extract all text from a PDF file.

    Returns:
        (full_text, page_count)

    Raises:
        FileNotFoundError if the file doesn't exist
        RuntimeError if pdfplumber is not installed
    """
    if not PDF_AVAILABLE:
        raise RuntimeError("pdfplumber is required. Run: pip install pdfplumber")

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {filepath}")

    pages_text = []
    page_count = 0

    try:
        with pdfplumber.open(str(path)) as pdf:
            page_count = len(pdf.pages)
            # Limit to first 15 pages on production to save memory
            MAX_PAGES = 15
            pages_to_read = pdf.pages[:MAX_PAGES]
            for i, page in enumerate(pages_to_read):
                text = page.extract_text()
                if text:
                    pages_text.append(f"[PAGE {i+1}]\n{text}")
                else:
                    logger.warning(f"Page {i+1} of {filepath} has no extractable text (may be image-based)")
    except Exception as e:
        logger.error(f"PDF extraction failed for {filepath}: {e}")
        raise

    full_text = "\n\n".join(pages_text)
    cleaned = _clean_text(full_text)
    logger.info(f"Extracted {len(cleaned)} characters from {page_count} pages: {path.name}")
    return cleaned, page_count


def _clean_text(text: str) -> str:
    """
    Clean extracted PDF text:
      - Collapse 3+ consecutive newlines into 2
      - Remove non-printable characters (except newlines and tabs)
      - Strip trailing whitespace from each line
    """
    # Remove non-printable chars (keep \n \t)
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]', '', text)
    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split('\n')]
    return '\n'.join(lines).strip()


def chunk_text(text: str, max_chars: int = 6000) -> List[str]:
    """
    Split text into chunks of at most max_chars characters,
    splitting at paragraph boundaries (double newlines) to preserve
    semantic coherence.

    max_chars=6000 ≈ ~1500 tokens, leaving headroom in a 4k context window.

    Returns:
        List of text chunks
    """
    if not text:
        return []

    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        # If a single paragraph exceeds max_chars, split it by sentence
        if para_len > max_chars:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                if current_len + len(sentence) > max_chars and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [sentence]
                    current_len = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_len += len(sentence)
        elif current_len + para_len > max_chars and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_len = para_len
        else:
            current_chunk.append(para)
            current_len += para_len

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return [c for c in chunks if c.strip()]


def get_pdf_summary_chunk(text: str, max_chars: int = 8000) -> str:
    """
    Returns the first max_chars of text — usually enough to capture
    the key claims in a pitch deck (problem, solution, market, model).
    Used when we want a single chunk for claim extraction rather than
    iterating over all chunks.
    """
    return text[:max_chars] if len(text) > max_chars else text