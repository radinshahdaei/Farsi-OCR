"""Extract per-page text from a searchable PDF using PyMuPDF.

This is the canonical text extraction step — more reliable than OCR sidecars
because it includes pages that OCRmyPDF skipped (already had text layers)
and guarantees the page count matches the PDF.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import fitz  # PyMuPDF

from farsi_book_ocr.models import PageRecord


def compute_pdf_fingerprint(pdf_path: Path) -> str:
    """Compute a fast but reliable SHA-256 fingerprint for a PDF file.

    Uses the first 64KB + last 64KB + file size. This is fast for large
    files and sufficient to detect content changes in practice.
    """
    file_size = pdf_path.stat().st_size
    hasher = hashlib.sha256()
    hasher.update(str(file_size).encode())

    with pdf_path.open("rb") as f:
        hasher.update(f.read(65536))  # first 64KB
        if file_size > 131072:
            f.seek(-65536, 2)  # last 64KB
            hasher.update(f.read(65536))

    return hasher.hexdigest()


def extract_pages_from_pdf(
    pdf_path: Path,
    document_id: str | None = None,
) -> list[PageRecord]:
    """Extract per-page text from a searchable PDF.

    Args:
        pdf_path: Path to the searchable (OCR'd) PDF.
        document_id: Stable document identifier. If None, computed from the PDF fingerprint.

    Returns:
        List of PageRecord, one per page, in document order.
    """
    if document_id is None:
        document_id = compute_pdf_fingerprint(pdf_path)

    records: list[PageRecord] = []

    with fitz.open(pdf_path) as doc:
        for i in range(doc.page_count):
            page = doc[i]
            text = page.get_text("text")
            page_id = PageRecord.make_page_id(i)

            records.append(
                PageRecord(
                    document_id=document_id,
                    page_id=page_id,
                    page_index=i,
                    source_text=text,
                    source_sha256=PageRecord.compute_sha256(text),
                )
            )

    return records


def verify_page_count(pdf_path: Path, expected_range: tuple[int, int] | None = None) -> tuple[int, bool]:
    """Verify the page count of a PDF.

    Args:
        pdf_path: Path to the PDF.
        expected_range: Optional (min_pages, max_pages) to check against.

    Returns:
        Tuple of (actual_page_count, is_valid).
    """
    with fitz.open(pdf_path) as doc:
        count = doc.page_count

    if expected_range is not None:
        min_pages, max_pages = expected_range
        return count, min_pages <= count <= max_pages

    return count, count > 0


def flag_problematic_pages(pages: list[PageRecord]) -> dict[str, list[str]]:
    """Flag pages with potential issues.

    Returns a dict mapping page_id → list of issue descriptions.
    """
    issues: dict[str, list[str]] = {}

    for page in pages:
        page_issues: list[str] = []

        # Empty or whitespace-only
        if len(page.source_text.strip()) == 0:
            page_issues.append("empty")

        # OCR-skipped placeholder
        if page.source_text.startswith("[OCR skipped on page"):
            page_issues.append("ocr_skipped")

        # Anomalously short (less than 10 chars, but not legitimately empty)
        if 0 < len(page.source_text.strip()) < 10:
            page_issues.append("anomalously_short")

        if page_issues:
            issues[page.page_id] = page_issues

    return issues
