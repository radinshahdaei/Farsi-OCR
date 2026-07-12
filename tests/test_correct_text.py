"""Tests for farsi_book_ocr.correct_text (clean wrapper)."""

from farsi_book_ocr.correct_text import estimate_cost, estimate_tokens, parse_args


class TestEstimateTokens:
    def test_approximate_count(self):
        assert estimate_tokens("a" * 100) == 50

    def test_minimum_one(self):
        assert estimate_tokens("") == 1
        assert estimate_tokens("a") == 1


class TestEstimateCost:
    def test_returns_three_values(self):
        in_cost, out_cost, total = estimate_cost(1000, 500)
        assert in_cost > 0
        assert out_cost > 0
        assert total == in_cost + out_cost

    def test_zero_chars(self):
        in_cost, out_cost, total = estimate_cost(0, 0)
        assert in_cost >= 0
        assert out_cost >= 0


class TestCLI:
    def test_estimate_only_flag(self):
        args = parse_args(["input.txt", "--estimate-only"])
        assert args.estimate_only is True

    def test_default_no_estimate(self):
        args = parse_args(["input.txt"])
        assert args.estimate_only is False

    def test_pages_per_request_default(self):
        args = parse_args(["input.txt"])
        assert args.pages_per_request == 1

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
