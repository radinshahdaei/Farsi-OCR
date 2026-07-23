"""DeepSeek provider via an Anthropic-compatible API proxy.

Uses the Anthropic Messages API format. Configured via environment variables:
  ANTHROPIC_BASE_URL - API base URL
  ANTHROPIC_AUTH_TOKEN - API key
  ANTHROPIC_MODEL - Model name (default: deepseek-v4-pro)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from farsi_book_ocr.models import CorrectionRequest, ProviderResponse, ProviderUsage
from farsi_book_ocr.providers.base import CorrectionProvider

logger = logging.getLogger(__name__)

# Load .env from project root
_load_dotenv_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_load_dotenv_path)

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

    def correct(self, request: CorrectionRequest) -> ProviderResponse:
        user_message = self._build_user_message(request)
        system_len = len(request.system_prompt)
        user_len = len(user_message)

        payload = {
            "model": (request.model or self._model).replace("[1m]", ""),
            "max_tokens": request.max_tokens,
            "system": request.system_prompt,
            "messages": [
                {"role": "user", "content": user_message},
            ],
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature

        logger.info(
            "→ API request: page=%s model=%s system_len=%d user_len=%d max_tokens=%d",
            request.page_id, payload["model"],
            system_len, user_len, request.max_tokens,
        )

        last_error: Exception | None = None
        for attempt in range(1 + self._max_retries):
            try:
                response = self._send(payload)
                logger.info(
                    "← API response: page=%s status=%d finish=%s in=%d out=%d req_id=%s",
                    request.page_id,
                    response.raw_status_code,
                    response.finish_reason,
                    response.usage.input_tokens if response.usage else 0,
                    response.usage.output_tokens if response.usage else 0,
                    response.request_id or "-",
                )
                return response
            except (httpx.HTTPError, httpx.TimeoutException, RuntimeError) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    wait = 2**attempt
                    logger.warning(
                        "API error (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1, self._max_retries + 1, exc, wait,
                    )
                    print(f"  API error: {exc}. Retrying in {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    logger.error(
                        "API request failed after %d attempts: %s",
                        self._max_retries + 1, exc,
                    )
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
        payload_json = json.dumps(payload, ensure_ascii=False)
        payload_kb = len(payload_json) / 1024

        url = f"{self._base_url}/v1/messages"
        logger.info("HTTP POST %s (payload %.1f KB, timeout %ds)", url, payload_kb, self._timeout)

        t0 = time.monotonic()
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
            elapsed = time.monotonic() - t0
            http_response.raise_for_status()
            body = http_response.json()

        response_bytes = len(http_response.content)
        logger.info(
            "HTTP %d in %.1fs (response %.1f KB)",
            http_response.status_code, elapsed, response_bytes / 1024,
        )

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

        corrected_len = len("".join(text_blocks))
        logger.info(
            "response body: %d chars, %d text blocks, stop_reason=%s",
            corrected_len, len(text_blocks), finish_reason,
        )

        return ProviderResponse(
            text="".join(text_blocks),
            finish_reason=finish_reason,
            usage=usage,
            request_id=body.get("id"),
            raw_status_code=http_response.status_code,
        )

