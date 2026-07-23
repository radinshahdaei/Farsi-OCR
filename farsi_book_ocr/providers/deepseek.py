"""DeepSeek provider via an Anthropic-compatible API proxy.

Uses the Anthropic Messages API format. Configured via environment variables:
  ANTHROPIC_BASE_URL - API base URL
  ANTHROPIC_AUTH_TOKEN - API key
  ANTHROPIC_MODEL - Model name (default: deepseek-v4-pro)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from farsi_book_ocr.models import CorrectionRequest, ProviderResponse, ProviderUsage
from farsi_book_ocr.providers.base import CorrectionProvider

# Load .env from project root
_load_dotenv_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_load_dotenv_path)

# Token estimation: Persian ~ 1 token per 2 chars (conservative)
_CHARS_PER_TOKEN = 2.0


def _get_env_or_die(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"Environment variable {name} is not set.\n"
            f"Set it before running:\n"
            f"  export {name}=<value>"
        )
    return value


class DeepSeekProvider(CorrectionProvider):
    """LLM correction via DeepSeek using the Anthropic Messages API protocol."""

    def __init__(
        self,
        base_url: str | None = None,
        auth_token: str | None = None,
        model: str | None = None,
        timeout: int = 180,
        max_retries: int = 2,
    ):
        self._base_url = (base_url or _get_env_or_die("ANTHROPIC_BASE_URL")).rstrip("/")
        self._auth_token = auth_token or _get_env_or_die("ANTHROPIC_AUTH_TOKEN")
        self._model = model or os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-pro").replace("[1m]", "")
        self._timeout = timeout
        self._max_retries = max_retries

    @property
    def provider_name(self) -> str:
        return "deepseek"

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def estimate_tokens(self, text: str) -> int:
        return max(1, int(len(text) / _CHARS_PER_TOKEN))

    def correct(self, request: CorrectionRequest) -> ProviderResponse:
        payload = {
            "model": request.model if request.model else self._model,
            "max_tokens": request.max_tokens,
            "system": request.system_prompt,
            "messages": [
                {"role": "user", "content": self._build_user_message(request)},
            ],
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature

        last_error: Exception | None = None
        for attempt in range(1 + self._max_retries):
            try:
                response = self._send(payload)
                return response
            except (httpx.HTTPError, httpx.TimeoutException, RuntimeError) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    wait = 2**attempt
                    print(f"  API error: {exc}. Retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"API request failed after {self._max_retries + 1} attempts"
                    ) from last_error

        # Should be unreachable (except clause always raises on last attempt)
        raise RuntimeError("Unexpected: all retries exhausted") from last_error

    def _build_user_message(self, request: CorrectionRequest) -> str:
        """Build the user message with optional context pages."""
        parts: list[str] = []
        if request.context_before:
            parts.append(f"[[CONTEXT: preceding page — READ ONLY, DO NOT CORRECT]]\n{request.context_before}\n[[/CONTEXT]]")
        parts.append(f"[[PAGE {request.page_id} — CORRECT THIS PAGE]]\n{request.source_text}\n[[/PAGE {request.page_id}]]")
        if request.context_after:
            parts.append(f"[[CONTEXT: following page — READ ONLY, DO NOT CORRECT]]\n{request.context_after}\n[[/CONTEXT]]")
        return "\n\n".join(parts)

    def _send(self, payload: dict) -> ProviderResponse:
        with httpx.Client(
            base_url=self._base_url,
            headers={
                "x-api-key": self._auth_token,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=self._timeout,
        ) as client:
            http_response = client.post("/v1/messages", json=payload)
            http_response.raise_for_status()
            body = http_response.json()

        content = body.get("content", [])
        text_blocks = [b.get("text", "") for b in content if b.get("type") == "text"]
        if not text_blocks:
            raise RuntimeError("API returned no text content block")

        finish_reason = body.get("stop_reason", "unknown")
        usage_raw = body.get("usage", {})
        usage = ProviderUsage(
            input_tokens=usage_raw.get("input_tokens", 0),
            output_tokens=usage_raw.get("output_tokens", 0),
        )

        return ProviderResponse(
            text="".join(text_blocks),
            finish_reason=finish_reason,
            usage=usage,
            request_id=body.get("id"),
            raw_status_code=http_response.status_code,
        )

