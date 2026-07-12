"""Characterization tests for farsi_book_ocr.ocr_book."""

import pytest

from farsi_book_ocr.ocr_book import Chunk, build_chunks


class TestChunk:
    def test_human_start_is_one_based(self):
        c = Chunk(index=1, start_page_zero_based=0, end_page_zero_based_exclusive=10)
        assert c.human_start == 1

    def test_human_end_matches_exclusive_bound(self):
        c = Chunk(index=1, start_page_zero_based=0, end_page_zero_based_exclusive=10)
        assert c.human_end == 10

    def test_label_format(self):
        c = Chunk(index=3, start_page_zero_based=50, end_page_zero_based_exclusive=75)
        assert c.label == "chunk_0003_p0051-0075"

    def test_single_page_chunk(self):
        c = Chunk(index=1, start_page_zero_based=0, end_page_zero_based_exclusive=1)
        assert c.human_start == 1
        assert c.human_end == 1
        assert c.label == "chunk_0001_p0001-0001"

    def test_frozen(self):
        c = Chunk(index=1, start_page_zero_based=0, end_page_zero_based_exclusive=10)
        with pytest.raises(Exception):
            c.index = 2  # type: ignore[misc]


class TestBuildChunks:
    def test_single_chunk_exact_fit(self):
        chunks = build_chunks(total_pages=25, pages_per_chunk=25, first_page=1, last_page=None)
        assert len(chunks) == 1
        assert chunks[0].start_page_zero_based == 0
        assert chunks[0].end_page_zero_based_exclusive == 25

    def test_two_chunks(self):
        chunks = build_chunks(total_pages=50, pages_per_chunk=25, first_page=1, last_page=None)
        assert len(chunks) == 2
        assert chunks[0].start_page_zero_based == 0
        assert chunks[0].end_page_zero_based_exclusive == 25
        assert chunks[1].start_page_zero_based == 25
        assert chunks[1].end_page_zero_based_exclusive == 50

    def test_partial_last_chunk(self):
        chunks = build_chunks(total_pages=40, pages_per_chunk=25, first_page=1, last_page=None)
        assert len(chunks) == 2
        assert chunks[0].end_page_zero_based_exclusive == 25
        assert chunks[1].start_page_zero_based == 25
        assert chunks[1].end_page_zero_based_exclusive == 40

    def test_no_gaps_between_chunks(self):
        chunks = build_chunks(total_pages=100, pages_per_chunk=30, first_page=1, last_page=None)
        for i in range(len(chunks) - 1):
            assert chunks[i].end_page_zero_based_exclusive == chunks[i + 1].start_page_zero_based

    def test_no_overlap_between_chunks(self):
        chunks = build_chunks(total_pages=100, pages_per_chunk=30, first_page=1, last_page=None)
        for i in range(len(chunks) - 1):
            assert chunks[i].end_page_zero_based_exclusive <= chunks[i + 1].start_page_zero_based

    def test_first_page_mid_document(self):
        chunks = build_chunks(total_pages=100, pages_per_chunk=25, first_page=51, last_page=None)
        assert chunks[0].human_start == 51
        assert chunks[0].start_page_zero_based == 50
        assert chunks[-1].human_end == 100

    def test_last_page_limited(self):
        chunks = build_chunks(total_pages=100, pages_per_chunk=25, first_page=1, last_page=50)
        assert chunks[-1].human_end == 50
        assert len(chunks) == 2

    def test_page_range_single_page(self):
        chunks = build_chunks(total_pages=100, pages_per_chunk=25, first_page=33, last_page=33)
        assert len(chunks) == 1
        assert chunks[0].human_start == 33
        assert chunks[0].human_end == 33

    def test_first_page_must_be_one_or_greater(self):
        with pytest.raises(SystemExit, match="first-page"):
            build_chunks(total_pages=10, pages_per_chunk=5, first_page=0, last_page=None)

    def test_last_page_must_be_gte_first_page(self):
        with pytest.raises(SystemExit, match="last-page"):
            build_chunks(total_pages=10, pages_per_chunk=5, first_page=5, last_page=3)

    def test_last_page_cannot_exceed_total(self):
        with pytest.raises(SystemExit, match="page count"):
            build_chunks(total_pages=10, pages_per_chunk=5, first_page=1, last_page=20)

    def test_indices_are_sequential(self):
        chunks = build_chunks(total_pages=77, pages_per_chunk=25, first_page=1, last_page=None)
        for i, c in enumerate(chunks, start=1):
            assert c.index == i

    def test_large_document_many_chunks(self):
        chunks = build_chunks(total_pages=1000, pages_per_chunk=25, first_page=1, last_page=None)
        assert len(chunks) == 40
        assert chunks[-1].human_end == 1000

    def test_total_pages_covered_no_holes(self):
        """Every page from first to last is covered by exactly one chunk."""
        chunks = build_chunks(total_pages=77, pages_per_chunk=25, first_page=1, last_page=None)
        all_pages = set()
        for c in chunks:
            for p in range(c.start_page_zero_based, c.end_page_zero_based_exclusive):
                all_pages.add(p)
        assert len(all_pages) == 77
        assert min(all_pages) == 0
        assert max(all_pages) == 76
