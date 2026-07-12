"""Tests for farsi_book_ocr.page_splitter."""

from farsi_book_ocr.models import PageRecord
from farsi_book_ocr.page_splitter import (
    count_pages,
    is_empty_page,
    is_skipped_page,
    split_text_into_pages,
)


class TestSplitTextIntoPages:
    def test_single_page_no_form_feed(self):
        pages = split_text_into_pages("just one page", "doc-1")
        assert len(pages) == 1
        assert pages[0].page_id == "page-000001"
        assert pages[0].source_text == "just one page"

    def test_multiple_pages(self):
        text = "page one\fpage two\fpage three"
        pages = split_text_into_pages(text, "doc-1")
        assert len(pages) == 3
        assert [p.page_id for p in pages] == [
            "page-000001",
            "page-000002",
            "page-000003",
        ]

    def test_page_indices_are_sequential(self):
        text = "a\fb\fc\fd\fe"
        pages = split_text_into_pages(text, "doc-1")
        assert [p.page_index for p in pages] == [0, 1, 2, 3, 4]

    def test_empty_text(self):
        pages = split_text_into_pages("", "doc-1")
        assert len(pages) == 1
        assert pages[0].source_text == ""

    def test_only_form_feeds(self):
        pages = split_text_into_pages("\f\f\f", "doc-1")
        assert len(pages) == 4  # 3 form feeds = 4 segments

    def test_document_id_preserved(self):
        pages = split_text_into_pages("text", "my-document-42")
        assert all(p.document_id == "my-document-42" for p in pages)

    def test_source_sha256_computed(self):
        pages = split_text_into_pages("hello", "doc-1")
        assert len(pages[0].source_sha256) == 64  # SHA-256 hex

    def test_sha256_deterministic(self):
        p1 = split_text_into_pages("same text", "doc-1")
        p2 = split_text_into_pages("same text", "doc-1")
        assert p1[0].source_sha256 == p2[0].source_sha256

    def test_sha256_different_for_different_content(self):
        p1 = split_text_into_pages("text A", "doc-1")
        p2 = split_text_into_pages("text B", "doc-1")
        assert p1[0].source_sha256 != p2[0].source_sha256


class TestChunkSeparatorStripping:
    def test_chunk_separator_stripped(self):
        text = "\n\n===== OCR CHUNK 0001: chunk_0001_p0001-0025 =====\n\nactual content"
        pages = split_text_into_pages(text, "doc-1")
        assert pages[0].source_text == "actual content"

    def test_page_separator_stripped(self):
        text = "===== PAGE 0001 =====\n\nactual content"
        pages = split_text_into_pages(text, "doc-1")
        assert pages[0].source_text == "actual content"

    def test_separator_within_form_feed_pages(self):
        text = (
            "===== OCR CHUNK 0001: test =====\n\ncontent one"
            "\f"
            "===== OCR CHUNK 0002: test =====\n\ncontent two"
        )
        pages = split_text_into_pages(text, "doc-1")
        assert len(pages) == 2
        assert pages[0].source_text == "content one"
        assert pages[1].source_text == "content two"


class TestCountPages:
    def test_single_page(self):
        assert count_pages("one page") == 1

    def test_two_pages(self):
        assert count_pages("page1\fpage2") == 2

    def test_empty(self):
        assert count_pages("") == 1


class TestIsEmptyPage:
    def test_empty(self):
        page = PageRecord(
            document_id="d", page_id="p-001", page_index=0,
            source_text="", source_sha256="abc",
        )
        assert is_empty_page(page)

    def test_whitespace_only(self):
        page = PageRecord(
            document_id="d", page_id="p-001", page_index=0,
            source_text="   \n  ", source_sha256="abc",
        )
        assert is_empty_page(page)

    def test_not_empty(self):
        page = PageRecord(
            document_id="d", page_id="p-001", page_index=0,
            source_text="content", source_sha256="abc",
        )
        assert not is_empty_page(page)


class TestIsSkippedPage:
    def test_skipped(self):
        page = PageRecord(
            document_id="d", page_id="p-001", page_index=0,
            source_text="[OCR skipped on page(s) 1]",
            source_sha256="abc",
        )
        assert is_skipped_page(page)

    def test_not_skipped(self):
        page = PageRecord(
            document_id="d", page_id="p-001", page_index=0,
            source_text="actual content",
            source_sha256="abc",
        )
        assert not is_skipped_page(page)
