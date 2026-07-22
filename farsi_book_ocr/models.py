"""Core data records for the page-safe correction pipeline.

These are frozen dataclasses used throughout the correction pipeline to
track per-page identity, correction status, and provider responses.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Page identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageRecord:
    """A single page extracted from an OCR'd document."""

    document_id: str
    """Stable identifier for the source document (SHA-256 of source PDF)."""

    page_id: str
    """Stable per-page identifier, e.g. 'page-000042'."""

    page_index: int
    """Zero-based index of this page within the document."""

    source_text: str
    """Raw OCR text for this page."""

    source_sha256: str
    """SHA-256 of source_text, used for cache validation."""

    @staticmethod
    def compute_sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def make_page_id(index: int) -> str:
        """Generate a stable page ID from a zero-based index."""
        return f"page-{index + 1:06d}"


# ---------------------------------------------------------------------------
# Correction result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectionRecord:
    """The result of correcting a single page via an LLM provider."""

    page_id: str
    """Matches PageRecord.page_id."""

    source_sha256: str
    """SHA-256 of the source text that was corrected."""

    corrected_text: str | None
    """The corrected text, or None if correction failed entirely."""

    status: str
    """One of: 'accepted', 'fallback_raw', 'failed'."""

    attempts: int
    """Number of API attempts made (including retries)."""

    provider: str
    """Provider name, e.g. 'deepseek', 'fake'."""

    model: str
    """Model identifier, e.g. 'deepseek-v4-pro'."""

    prompt_version: str
    """SHA-256 of the system prompt text used."""

    finish_reason: str | None = None
    """The finish_reason from the provider response, if available."""

    input_tokens: int | None = None
    """Estimated or reported input token count."""

    output_tokens: int | None = None
    """Estimated or reported output token count."""

    validation_results: list[str] = field(default_factory=list)
    """List of validation check names that passed/failed."""

    output_sha256: str | None = None
    """SHA-256 of corrected_text, or None if no text was produced."""

    error: str | None = None
    """Error message if status is 'failed'."""


# ---------------------------------------------------------------------------
# Provider request / response
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectionRequest:
    """A request to an LLM provider for text correction."""

    page_id: str
    """The page being corrected."""

    source_text: str
    """The OCR text to correct."""

    system_prompt: str
    """The system prompt to send."""

    model: str
    """Model identifier."""

    max_tokens: int
    """Maximum output tokens."""

    temperature: float = 0.0
    """Sampling temperature (0.0 = deterministic when supported)."""

    context_before: str | None = None
    """Optional read-only context from the preceding page."""

    context_after: str | None = None
    """Optional read-only context from the following page."""


@dataclass(frozen=True)
class ProviderUsage:
    """Token usage reported by a provider."""

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized response from a correction provider.

    This decouples the pipeline from SDK-specific response types.
    """

    text: str
    """The corrected text from the provider."""

    finish_reason: str
    """Why the provider stopped: 'stop', 'end_turn', 'max_tokens', etc."""

    usage: ProviderUsage | None = None
    """Token usage if reported by the provider."""

    request_id: str | None = None
    """Provider-assigned request identifier for debugging."""

    raw_status_code: int = 200
    """HTTP status code from the provider response."""


# ---------------------------------------------------------------------------
# Correction configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectionConfig:
    """Complete configuration for a correction run."""

    provider: str
    """Provider name, e.g. 'deepseek'."""

    model: str
    """Model identifier."""

    base_url: str
    """API base URL."""

    api_key: str
    """API authentication key (never logged)."""

    prompt_version: str
    """SHA-256 of the system prompt."""

    max_output_tokens: int = 65536
    """Hard ceiling on output tokens."""

    max_retries: int = 1
    """Maximum API attempts including initial call."""

    pages_per_request: int = 1
    """Pages per API call (1 = most reliable)."""

    temperature: float = 0.0
    """Sampling temperature."""

    timeout_seconds: int = 180
    """HTTP request timeout."""

    fallback_policy: str = "strict"
    """'strict': fail if any page unresolved. 'fallback_raw': use source text."""


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectionRunResult:
    """Summary of a completed correction run."""

    pages: list[CorrectionRecord]
    """Per-page results, in page_index order."""

    status: str
    """'completed', 'completed_with_fallbacks', or 'failed'."""

    total_input_tokens: int = 0
    """Sum of input tokens across all API calls."""

    total_output_tokens: int = 0
    """Sum of output tokens across all API calls."""

    total_cost_estimate: float | None = None
    """Estimated cost in USD if pricing is known."""

    @property
    def accepted_count(self) -> int:
        return sum(1 for p in self.pages if p.status == "accepted")

    @property
    def fallback_count(self) -> int:
        return sum(1 for p in self.pages if p.status == "fallback_raw")

    @property
    def failed_count(self) -> int:
        return sum(1 for p in self.pages if p.status == "failed")

    @property
    def all_resolved(self) -> bool:
        return self.failed_count == 0
