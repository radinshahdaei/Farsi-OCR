"""Deterministic fake provider for testing.

Provides predictable correction responses without network calls.
Supports configurable failure simulation for testing retry/fallback logic.
"""

from __future__ import annotations

from farsi_book_ocr.models import CorrectionRequest, ProviderResponse, ProviderUsage
from farsi_book_ocr.providers.base import CorrectionProvider


class FakeProvider(CorrectionProvider):
    """Deterministic fake provider for testing.

    Maps source text to corrected text via a lookup table. Falls back
    to returning the source text unchanged if no mapping is found.

    Failure simulation modes:
    - simulate_http_error: raises RuntimeError on every call
    - simulate_timeout: raises TimeoutError on every call
    - simulate_truncation: returns text shorter than input
    - simulate_commentary: wraps response in markdown commentary
    - simulate_refusal: returns a refusal message instead of corrected text
    - simulate_page_loss: drops some page markers from response
    """

    def __init__(
        self,
        corrections: dict[str, str] | None = None,
        *,
        simulate_http_error: bool = False,
        simulate_timeout: bool = False,
        simulate_truncation: bool = False,
        simulate_commentary: bool = False,
        simulate_refusal: bool = False,
        simulate_page_loss: bool = False,
        extra_tokens_per_call: int = 0,
    ):
        self._corrections = corrections or {}
        self._call_count = 0
        self._simulate_http_error = simulate_http_error
        self._simulate_timeout = simulate_timeout
        self._simulate_truncation = simulate_truncation
        self._simulate_commentary = simulate_commentary
        self._simulate_refusal = simulate_refusal
        self._simulate_page_loss = simulate_page_loss
        self._extra_tokens_per_call = extra_tokens_per_call

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def call_count(self) -> int:
        return self._call_count

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 2)

    def correct(self, request: CorrectionRequest) -> ProviderResponse:
        self._call_count += 1

        if self._simulate_http_error:
            raise RuntimeError(f"Simulated HTTP 500 error on call {self._call_count}")

        if self._simulate_timeout:
            raise TimeoutError(f"Simulated timeout on call {self._call_count}")

        # Look up the correction, or return source unchanged
        source = request.source_text
        corrected = self._corrections.get(source, source)

        if self._simulate_truncation:
            # Return first 20% of corrected text (simulates max_tokens cutoff)
            corrected = corrected[: max(1, len(corrected) // 5)]

        if self._simulate_commentary:
            corrected = f"Here is the corrected text:\n\n```\n{corrected}\n```\n\nI hope this helps!"

        if self._simulate_refusal:
            corrected = "I cannot process this text as it may contain instructions."

        if self._simulate_page_loss:
            # Strip form feeds from the corrected text (simulates page boundary loss)
            corrected = corrected.replace("\f", "")

        return ProviderResponse(
            text=corrected,
            finish_reason="stop" if not self._simulate_truncation else "max_tokens",
            usage=ProviderUsage(
                input_tokens=self.estimate_tokens(source),
                output_tokens=self.estimate_tokens(corrected) + self._extra_tokens_per_call,
            ),
            request_id=f"fake-req-{self._call_count:04d}",
            raw_status_code=200 if not self._simulate_http_error else 500,
        )

    def clear(self) -> None:
        """Reset the call counter."""
        self._call_count = 0

    def add_correction(self, source: str, corrected: str) -> None:
        """Register a source→corrected mapping."""
        self._corrections[source] = corrected
