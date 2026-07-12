"""Split OCR output text into individual pages on form-feed boundaries.

Form feeds (\\f) are the reliable page-break markers from OCRmyPDF/PyMuPDF
PDF text extraction. OCR chunk separators added by merge_text() are
synthetic and should not be used as page boundaries.
"""

from __future__ import annotations

import re

from farsi_book_ocr.models import PageRecord

# Matches "===== OCR CHUNK 0001: ... =====" and "===== PAGE 0001 =====" lines
_CHUNK_SEPARATOR_RE = re.compile(
    r"\n{0,2}===== (?:PAGE|OCR CHUNK) \d{4}.*?=====\n{0,2}"
)


def split_text_into_pages(text: str, document_id: str) -> list[PageRecord]:
    """Split OCR output text into individual pages on form feed boundaries.

    Each \\f (form feed, ASCII 12) marks a real page boundary from the PDF
    extraction.  OCR chunk separators are stripped from page content but
    do not serve as page boundaries.

    Args:
        text: Raw OCR output text.
        document_id: Stable identifier for the source document.

    Returns:
        List of PageRecord, one per page, in document order.
    """
    # Split on form feeds to get individual pages
    raw_pages = text.split("\f")

    records: list[PageRecord] = []
    seen_content: set[str] = set()  # track empty pages for index assignment

    for i, raw in enumerate(raw_pages):
        # Strip chunk separators and surrounding whitespace
        cleaned = _CHUNK_SEPARATOR_RE.sub("", raw).strip()

        page_id = PageRecord.make_page_id(i)
        record = PageRecord(
            document_id=document_id,
            page_id=page_id,
            page_index=i,
            source_text=cleaned,
            source_sha256=PageRecord.compute_sha256(cleaned),
        )
        records.append(record)

    return records


def count_pages(text: str) -> int:
    """Count the number of pages (form-feed-delimited segments) in the text."""
    return text.count("\f") + 1


def is_empty_page(page: PageRecord) -> bool:
    """Check if a page is effectively empty (no visible content)."""
    return len(page.source_text.strip()) == 0


def is_skipped_page(page: PageRecord) -> bool:
    """Check if a page was skipped by OCRmyPDF."""
    return page.source_text.startswith("[OCR skipped on page")
