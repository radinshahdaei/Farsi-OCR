"""Tests for farsi_book_ocr.providers."""

import pytest

from farsi_book_ocr.models import CorrectionRequest
from farsi_book_ocr.providers.base import CorrectionProvider
from farsi_book_ocr.providers.fake import FakeProvider


class TestFakeProvider:
    def test_returns_corrected_text_from_lookup(self):
        provider = FakeProvider(corrections={"wrong": "right"})
        req = CorrectionRequest(
            page_id="page-000001",
            source_text="wrong",
            system_prompt="fix",
            model="test",
            max_tokens=100,
        )
        resp = provider.correct(req)
        assert resp.text == "right"
        assert resp.finish_reason == "stop"

    def test_returns_source_when_no_mapping(self):
        provider = FakeProvider()
        req = CorrectionRequest(
            page_id="page-000001",
            source_text="unchanged text",
            system_prompt="fix",
            model="test",
            max_tokens=100,
        )
        resp = provider.correct(req)
        assert resp.text == "unchanged text"

    def test_call_count_increments(self):
        provider = FakeProvider()
        req = CorrectionRequest(
            page_id="page-000001",
            source_text="test",
            system_prompt="fix",
            model="test",
            max_tokens=100,
        )
        assert provider.call_count == 0
        provider.correct(req)
        assert provider.call_count == 1
        provider.correct(req)
        assert provider.call_count == 2

    def test_clear_resets_counter(self):
        provider = FakeProvider()
        req = CorrectionRequest(
            page_id="page-000001",
            source_text="test",
            system_prompt="fix",
            model="test",
            max_tokens=100,
        )
        provider.correct(req)
        provider.clear()
        assert provider.call_count == 0

    def test_simulate_http_error(self):
        provider = FakeProvider(simulate_http_error=True)
        req = CorrectionRequest(
            page_id="page-000001",
            source_text="test",
            system_prompt="fix",
            model="test",
            max_tokens=100,
        )
        with pytest.raises(RuntimeError, match="Simulated HTTP 500"):
            provider.correct(req)

    def test_simulate_timeout(self):
        provider = FakeProvider(simulate_timeout=True)
        req = CorrectionRequest(
            page_id="page-000001",
            source_text="test",
            system_prompt="fix",
            model="test",
            max_tokens=100,
        )
        with pytest.raises(TimeoutError, match="Simulated timeout"):
            provider.correct(req)

    def test_simulate_truncation(self):
        provider = FakeProvider(simulate_truncation=True)
        req = CorrectionRequest(
            page_id="page-000001",
            source_text="a" * 100,
            system_prompt="fix",
            model="test",
            max_tokens=100,
        )
        resp = provider.correct(req)
        assert len(resp.text) < len("a" * 100)
        assert resp.finish_reason == "max_tokens"

    def test_simulate_commentary(self):
        provider = FakeProvider(simulate_commentary=True)
        req = CorrectionRequest(
            page_id="page-000001",
            source_text="real text",
            system_prompt="fix",
            model="test",
            max_tokens=100,
        )
        resp = provider.correct(req)
        assert "```" in resp.text
        assert "Here is" in resp.text

    def test_simulate_refusal(self):
        provider = FakeProvider(simulate_refusal=True)
        req = CorrectionRequest(
            page_id="page-000001",
            source_text="some text",
            system_prompt="fix",
            model="test",
            max_tokens=100,
        )
        resp = provider.correct(req)
        assert "cannot process" in resp.text.lower()

    def test_simulate_page_loss(self):
        provider = FakeProvider(simulate_page_loss=True)
        req = CorrectionRequest(
            page_id="page-000001",
            source_text="text\fwith\fform\ffeeds",
            system_prompt="fix",
            model="test",
            max_tokens=100,
        )
        resp = provider.correct(req)
        assert "\f" not in resp.text

    def test_provider_name(self):
        provider = FakeProvider()
        assert provider.provider_name == "fake"

    def test_usage_is_reported(self):
        provider = FakeProvider(extra_tokens_per_call=10)
        req = CorrectionRequest(
            page_id="page-000001",
            source_text="test text",
            system_prompt="fix",
            model="test",
            max_tokens=100,
        )
        resp = provider.correct(req)
        assert resp.usage is not None
        assert resp.usage.input_tokens > 0
        assert resp.usage.output_tokens > 0

    def test_add_correction(self):
        provider = FakeProvider()
        provider.add_correction("old", "new")
        req = CorrectionRequest(
            page_id="page-000001",
            source_text="old",
            system_prompt="fix",
            model="test",
            max_tokens=100,
        )
        resp = provider.correct(req)
        assert resp.text == "new"

    def test_is_abstract_provider(self):
        """FakeProvider must be a valid CorrectionProvider."""
        assert isinstance(FakeProvider(), CorrectionProvider)
