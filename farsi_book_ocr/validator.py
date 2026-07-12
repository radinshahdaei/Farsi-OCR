"""Validate LLM correction responses against source pages.

Pure functions — no network, no state. Each check examines one aspect of
the response and returns a (name, passed, detail) tuple.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from farsi_book_ocr.models import PageRecord, ProviderResponse

# Phrases that indicate the LLM is commenting instead of returning raw text
_COMMENTARY_PREFIXES_EN = [
    "here is the corrected",
    "here is your corrected",
    "here's the corrected",
    "i've corrected",
    "i have corrected",
    "certainly",
    "sure",
    "below is",
    "the corrected text is",
    "here you go",
    "no problem",
]

_COMMENTARY_PREFIXES_FA = [
    "متن تصحیح",
    "متن اصلاح",
    "در ادامه",
    "تصحیح شده",
]

# Regex patterns for protected content
_URL_RE = re.compile(r"https?://[^\s<>\"{}|\\^`\[\]]+")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_DIGIT_RE = re.compile(r"\d")


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating a correction response."""

    passed: bool
    """True if all checks passed."""

    checks: list[tuple[str, bool, str]]
    """List of (check_name, passed, detail) tuples."""

    @property
    def failed_checks(self) -> list[tuple[str, bool, str]]:
        return [(n, p, d) for n, p, d in self.checks if not p]

    @property
    def passed_checks(self) -> list[tuple[str, bool, str]]:
        return [(n, p, d) for n, p, d in self.checks if p]


def validate_correction_response(
    source: PageRecord,
    response: ProviderResponse,
    *,
    max_length_ratio: float = 3.0,
    min_length_ratio: float = 0.1,
    max_edit_ratio: float = 0.6,
    digit_tolerance: float = 0.02,
) -> ValidationResult:
    """Validate a correction response against its source page.

    Args:
        source: The source page record.
        response: The provider's response.
        max_length_ratio: Maximum allowed ratio of corrected/source length.
        min_length_ratio: Minimum allowed ratio of corrected/source length.
        max_edit_ratio: Maximum allowed character-level edit distance ratio.
        digit_tolerance: Fraction of digits allowed to differ (0.02 = 2%).

    Returns:
        ValidationResult with all check outcomes.
    """
    checks: list[tuple[str, bool, str]] = []

    source_text = source.source_text
    corrected_text = response.text
    source_len = max(len(source_text), 1)
    corrected_len = max(len(corrected_text), 1)

    # 1. Finish reason — must not be truncated
    truncated = response.finish_reason in ("max_tokens", "length", "truncated")
    checks.append((
        "finish_reason_not_truncated",
        not truncated,
        f"finish_reason={response.finish_reason}"
        if truncated
        else "ok",
    ))

    # 2. Not empty when source is not empty
    source_nonempty = len(source_text.strip()) > 0
    response_nonempty = len(corrected_text.strip()) > 0
    checks.append((
        "text_not_empty",
        not source_nonempty or response_nonempty,
        "response is empty but source is not" if source_nonempty and not response_nonempty else "ok",
    ))

    # 3. Length ratio within bounds
    length_ratio = corrected_len / max(source_len, 1)
    length_ok = min_length_ratio <= length_ratio <= max_length_ratio
    checks.append((
        "length_in_bounds",
        length_ok,
        f"ratio={length_ratio:.2f} (allowed: [{min_length_ratio}, {max_length_ratio}])"
        if not length_ok
        else f"ratio={length_ratio:.2f}",
    ))

    # 4. No form feed injection (single-page correction)
    has_ff = "\f" in corrected_text
    checks.append((
        "no_form_feed_injection",
        not has_ff,
        "response contains form feed character"
        if has_ff
        else "ok",
    ))

    # 5. No chunk separator injection
    has_sep = "===== OCR CHUNK" in corrected_text or "===== PAGE" in corrected_text
    checks.append((
        "no_separator_injection",
        not has_sep,
        "response contains OCR chunk or page separator"
        if has_sep
        else "ok",
    ))

    # 6. No markdown fences
    has_fences = "```" in corrected_text or "~~~" in corrected_text
    checks.append((
        "no_markdown_fences",
        not has_fences,
        "response contains markdown code fences" if has_fences else "ok",
    ))

    # 7. No commentary prefixes (check first 100 chars, case-insensitive)
    lower_start = corrected_text.strip()[:100].lower()
    commentary = False
    matched_prefix = ""
    for prefix in _COMMENTARY_PREFIXES_EN + _COMMENTARY_PREFIXES_FA:
        if lower_start.startswith(prefix.lower()):
            commentary = True
            matched_prefix = prefix
            break
    checks.append((
        "no_commentary_prefix",
        not commentary,
        f"response starts with commentary: '{matched_prefix}'"
        if commentary
        else "ok",
    ))

    # 8. Edit distance check (simple character-level ratio)
    if source_len > 0 and corrected_len > 0:
        # Fast approximation: compare character set overlap
        source_chars = set(source_text)
        corrected_chars = set(corrected_text)
        if source_chars:
            overlap = len(source_chars & corrected_chars) / len(source_chars)
            edit_ok = overlap >= (1.0 - max_edit_ratio)
        else:
            edit_ok = True
    else:
        edit_ok = True
    checks.append((
        "character_overlap_ok",
        edit_ok,
        "character overlap too low — possible rewrite" if not edit_ok else "ok",
    ))

    # 9. Digit preservation (Persian ۰۱۲۳۴۵۶۷۸۹, Arabic ٠١٢٣٤٥٦٧٨٩, Latin 0-9)
    source_digits = len(_DIGIT_RE.findall(source_text))
    corrected_digits = len(_DIGIT_RE.findall(corrected_text))
    digit_ok = True
    if source_digits > 0:
        digit_diff_ratio = abs(corrected_digits - source_digits) / max(source_digits, 1)
        digit_ok = digit_diff_ratio <= digit_tolerance
    checks.append((
        "digit_preservation",
        digit_ok,
        f"source={source_digits}, corrected={corrected_digits}"
        if not digit_ok
        else f"digits preserved ({source_digits})",
    ))

    # 10. URL/email preservation
    source_urls = set(_URL_RE.findall(source_text))
    source_emails = set(_EMAIL_RE.findall(source_text))
    corrected_urls = set(_URL_RE.findall(corrected_text))
    corrected_emails = set(_EMAIL_RE.findall(corrected_text))
    missing_urls = source_urls - corrected_urls
    missing_emails = source_emails - corrected_emails
    url_email_ok = len(missing_urls) == 0 and len(missing_emails) == 0
    checks.append((
        "url_email_preservation",
        url_email_ok,
        f"missing URLs: {missing_urls}, missing emails: {missing_emails}"
        if not url_email_ok
        else "ok",
    ))

    passed = all(p for _, p, _ in checks)
    return ValidationResult(passed=passed, checks=checks)
