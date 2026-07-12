"""Abstract provider interface for LLM-based OCR correction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from farsi_book_ocr.models import CorrectionRequest, ProviderResponse


class CorrectionProvider(ABC):
    """Abstract provider for LLM-based Persian OCR text correction.

    Implementations handle a specific API backend (DeepSeek, Anthropic, etc.)
    and return normalized ProviderResponse objects.
    """

    @abstractmethod
    def correct(self, request: CorrectionRequest) -> ProviderResponse:
        """Send a correction request and return the normalized response."""
        ...

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a given text.

        Used for cost estimation and max_tokens auto-calculation.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier, e.g. 'deepseek'."""
        ...
