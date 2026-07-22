"""Tests for farsi_book_ocr.validator — minimal validation."""

from farsi_book_ocr.models import PageRecord, ProviderResponse, ProviderUsage
from farsi_book_ocr.validator import validate_correction_response


def make_page(text: str, index: int = 0) -> PageRecord:
    return PageRecord(
        document_id="test-doc",
        page_id=PageRecord.make_page_id(index),
        page_index=index,
        source_text=text,
        source_sha256=PageRecord.compute_sha256(text),
    )


def make_response(
    text: str,
    finish_reason: str = "stop",
    input_tokens: int = 10,
    output_tokens: int = 10,
) -> ProviderResponse:
    return ProviderResponse(
        text=text,
        finish_reason=finish_reason,
        usage=ProviderUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        request_id="test-req-001",
        raw_status_code=200,
    )


class TestFinishReason:
    def test_stop_passes(self):
        page = make_page("hello")
        resp = make_response("hello", finish_reason="stop")
        result = validate_correction_response(page, resp)
        assert result.passed

    def test_end_turn_passes(self):
        page = make_page("hello")
        resp = make_response("hello", finish_reason="end_turn")
        result = validate_correction_response(page, resp)
        assert result.passed

    def test_max_tokens_fails(self):
        page = make_page("hello")
        resp = make_response("hel", finish_reason="max_tokens")
        result = validate_correction_response(page, resp)
        assert not result.passed
        assert any("finish_reason" in name for name, _, _ in result.failed_checks)

    def test_length_fails(self):
        page = make_page("hello")
        resp = make_response("hel", finish_reason="length")
        result = validate_correction_response(page, resp)
        assert not result.passed

    def test_truncated_fails(self):
        page = make_page("hello")
        resp = make_response("hel", finish_reason="truncated")
        result = validate_correction_response(page, resp)
        assert not result.passed


class TestEmptyResponse:
    def test_empty_when_source_nonempty_fails(self):
        page = make_page("non-empty source")
        resp = make_response("")
        result = validate_correction_response(page, resp)
        assert not result.passed

    def test_empty_when_source_empty_passes(self):
        page = make_page("")
        resp = make_response("", finish_reason="stop")
        result = validate_correction_response(page, resp)
        assert result.passed


class TestPersianUnicode:
    def test_persian_text_validates(self):
        page = make_page("سلام دنیا این یک متن فارسی است")
        resp = make_response("سلام دنیا این یک متن فارسی است")
        result = validate_correction_response(page, resp)
        assert result.passed

    def test_zwnj_preserved(self):
        """Zero-width non-joiners should not cause validation failure."""
        page = make_page("م‌ی‌خواهم")
        resp = make_response("م‌ی‌خواهم")
        result = validate_correction_response(page, resp)
        assert result.passed

    def test_mixed_persian_english(self):
        page = make_page("The word کتاب means book in Persian")
        resp = make_response("The word کتاب means book in Persian")
        result = validate_correction_response(page, resp)
        assert result.passed


class TestEdgeCases:
    def test_empty_page_to_empty_response(self):
        page = make_page("")
        resp = make_response("", finish_reason="stop")
        result = validate_correction_response(page, resp)
        assert result.passed

    def test_very_long_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 250
        page = make_page(text)
        resp = make_response(text.replace("fox", "cat"))
        result = validate_correction_response(page, resp)
        assert result.passed

    def test_html_in_text(self):
        page = make_page("<div>text</div>")
        resp = make_response("<div>text</div>")
        result = validate_correction_response(page, resp)
        assert result.passed
