"""Tests for farsi_book_ocr.correct_text (clean wrapper)."""

from farsi_book_ocr.correct_text import parse_args


class TestCLI:
    def test_pages_per_request_default(self):
        args = parse_args(["input.txt"])
        assert args.pages_per_request == 20

    def test_context_pages_default(self):
        args = parse_args(["input.txt"])
        assert args.context_pages == 3

    def test_output_path(self):
        args = parse_args(["input.txt", "--output", "out.txt"])
        assert args.output.name == "out.txt"

    def test_fallback_flag(self):
        args = parse_args(["input.txt", "--fallback"])
        assert args.fallback is True

    def test_redo_flag(self):
        args = parse_args(["input.txt", "--redo"])
        assert args.redo is True

    def test_model_override(self):
        args = parse_args(["input.txt", "--model", "gpt-4"])
        assert args.model == "gpt-4"

    def test_log_flag(self):
        args = parse_args(["input.txt", "--log"])
        assert args.log is True

    def test_log_short_flag(self):
        args = parse_args(["input.txt", "-v"])
        assert args.log is True

    def test_default_no_log(self):
        args = parse_args(["input.txt"])
        assert args.log is False
