"""Quality metrics for evaluating OCR and correction output.

Computes character error rate (CER), word error rate (WER), page coverage,
and protected-token preservation against human-transcribed ground truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Levenshtein distance (no dependency needed)
# ---------------------------------------------------------------------------


def levenshtein(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


# ---------------------------------------------------------------------------
# Character Error Rate
# ---------------------------------------------------------------------------


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute CER: Levenshtein distance / reference length.

    Returns a float between 0.0 (perfect) and potentially >1.0 (very bad).
    """
    if len(reference) == 0:
        return float(len(hypothesis) > 0)
    return levenshtein(reference, hypothesis) / len(reference)


# ---------------------------------------------------------------------------
# Word Error Rate (space-based tokenization)
# ---------------------------------------------------------------------------


def _tokenize_persian(text: str) -> list[str]:
    """Simple space-based tokenization for Persian text.

    Note: Persian word boundaries are not perfectly captured by spaces
    (due to clitics like را, ها, etc.), but this is the standard
    approach for rapid evaluation.
    """
    return text.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute WER using space-based tokenization."""
    ref_words = _tokenize_persian(reference)
    hyp_words = _tokenize_persian(hypothesis)

    if len(ref_words) == 0:
        return float(len(hyp_words) > 0)

    return levenshtein(" ".join(ref_words), " ".join(hyp_words)) / len(" ".join(ref_words))


# ---------------------------------------------------------------------------
# Page coverage
# ---------------------------------------------------------------------------


def page_coverage(expected_count: int, actual_count: int) -> float:
    """Ratio of actual page count to expected page count.

    Returns 1.0 when all pages are present, <1.0 when pages are missing.
    """
    if expected_count == 0:
        return 1.0
    return min(actual_count / expected_count, 1.0)


# ---------------------------------------------------------------------------
# Protected-token preservation
# ---------------------------------------------------------------------------


_DIGIT_RE = re.compile(r"\d")
_URL_RE = re.compile(r"https?://[^\s<>\"{}|\\^`\[\]]+")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


def digit_preservation_rate(reference: str, hypothesis: str) -> float:
    """Ratio of digits preserved from reference to hypothesis."""
    ref_digits = len(_DIGIT_RE.findall(reference))
    if ref_digits == 0:
        return 1.0
    hyp_digits = len(_DIGIT_RE.findall(hypothesis))
    return min(hyp_digits / ref_digits, 1.0)


def url_preservation_rate(reference: str, hypothesis: str) -> float:
    """Ratio of URLs preserved from reference to hypothesis."""
    ref_urls = set(_URL_RE.findall(reference))
    if len(ref_urls) == 0:
        return 1.0
    hyp_urls = set(_URL_RE.findall(hypothesis))
    return len(ref_urls & hyp_urls) / len(ref_urls)


def email_preservation_rate(reference: str, hypothesis: str) -> float:
    """Ratio of emails preserved from reference to hypothesis."""
    ref_emails = set(_EMAIL_RE.findall(reference))
    if len(ref_emails) == 0:
        return 1.0
    hyp_emails = set(_EMAIL_RE.findall(hypothesis))
    return len(ref_emails & hyp_emails) / len(ref_emails)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Aggregate quality metrics for a correction run."""

    cer: float
    """Character error rate (0.0 = perfect)."""

    wer: float
    """Word error rate (0.0 = perfect)."""

    page_coverage: float
    """Ratio of actual pages to expected pages (1.0 = perfect)."""

    digit_preservation: float
    """Ratio of digits preserved (1.0 = all preserved)."""

    url_preservation: float
    """Ratio of URLs preserved."""

    email_preservation: float
    """Ratio of email addresses preserved."""

    reference_char_count: int
    hypothesis_char_count: int

    reference_page_count: int
    hypothesis_page_count: int

    def to_dict(self) -> dict:
        return {
            "cer": round(self.cer, 6),
            "wer": round(self.wer, 6),
            "page_coverage": round(self.page_coverage, 4),
            "digit_preservation": round(self.digit_preservation, 4),
            "url_preservation": round(self.url_preservation, 4),
            "email_preservation": round(self.email_preservation, 4),
            "reference_char_count": self.reference_char_count,
            "hypothesis_char_count": self.hypothesis_char_count,
            "reference_page_count": self.reference_page_count,
            "hypothesis_page_count": self.hypothesis_page_count,
        }


def compute_benchmark(
    reference: str,
    hypothesis: str,
    expected_page_count: int | None = None,
) -> BenchmarkResult:
    """Compute all quality metrics between reference and hypothesis text.

    Args:
        reference: Ground truth (human transcription).
        hypothesis: OCR or corrected output.
        expected_page_count: Expected number of pages in the hypothesis.
            If None, derived from form-feed count.

    Returns:
        BenchmarkResult with all metrics.
    """
    hyp_page_count = hypothesis.count("\f") + 1
    ref_page_count = expected_page_count or (reference.count("\f") + 1)

    return BenchmarkResult(
        cer=character_error_rate(reference, hypothesis),
        wer=word_error_rate(reference, hypothesis),
        page_coverage=page_coverage(ref_page_count, hyp_page_count),
        digit_preservation=digit_preservation_rate(reference, hypothesis),
        url_preservation=url_preservation_rate(reference, hypothesis),
        email_preservation=email_preservation_rate(reference, hypothesis),
        reference_char_count=len(reference),
        hypothesis_char_count=len(hypothesis),
        reference_page_count=ref_page_count,
        hypothesis_page_count=hyp_page_count,
    )


def compare_benchmarks(
    baseline: BenchmarkResult,
    current: BenchmarkResult,
    *,
    cer_regression_threshold: float = 0.05,
    wer_regression_threshold: float = 0.05,
    page_loss_threshold: int = 1,
) -> tuple[bool, list[str]]:
    """Compare two benchmarks and flag regressions.

    Returns:
        (has_regression, list_of_regression_descriptions).
    """
    regressions: list[str] = []

    cer_delta = current.cer - baseline.cer
    if cer_delta > cer_regression_threshold:
        regressions.append(
            f"CER regression: +{cer_delta:.4f} (baseline {baseline.cer:.4f} → {current.cer:.4f})"
        )

    wer_delta = current.wer - baseline.wer
    if wer_delta > wer_regression_threshold:
        regressions.append(
            f"WER regression: +{wer_delta:.4f} (baseline {baseline.wer:.4f} → {current.wer:.4f})"
        )

    page_loss = baseline.hypothesis_page_count - current.hypothesis_page_count
    if page_loss >= page_loss_threshold:
        regressions.append(
            f"Page loss: {page_loss} pages (baseline {baseline.hypothesis_page_count} → {current.hypothesis_page_count})"
        )

    return len(regressions) > 0, regressions
