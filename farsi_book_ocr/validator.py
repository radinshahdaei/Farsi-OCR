"""Minimal validation for LLM correction responses.

Only checks for actual data loss — no cosmetic or statistical checks.
"""

from __future__ import annotations

from dataclasses import dataclass

from farsi_book_ocr.models import PageRecord, ProviderResponse


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
) -> ValidationResult:
    """Validate a correction response against its source page.

    Only checks for actual data loss: was the response truncated, and
    is it empty when the source had content. Everything else is trusted.
    """
    checks: list[tuple[str, bool, str]] = []

    source_text = source.source_text
    corrected_text = response.text

    # 1. Finish reason — must not be truncated
    truncated = response.finish_reason in ("max_tokens", "length", "truncated")
    checks.append((
        "finish_reason_not_truncated",
        not truncated,
        f"finish_reason={response.finish_reason}" if truncated else "ok",
    ))

    # 2. Not empty when source is not empty
    source_nonempty = len(source_text.strip()) > 0
    response_nonempty = len(corrected_text.strip()) > 0
    checks.append((
        "text_not_empty",
        not source_nonempty or response_nonempty,
        "response is empty but source is not" if source_nonempty and not response_nonempty else "ok",
    ))

    passed = all(p for _, p, _ in checks)
    return ValidationResult(passed=passed, checks=checks)
