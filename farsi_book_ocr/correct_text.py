"""LLM-based OCR correction for Persian text using DeepSeek API.

Uses the Anthropic Messages API format via a proxy to DeepSeek.
Reads ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN, and ANTHROPIC_MODEL from the environment.

Example:
    python -m farsi_book_ocr.correct_text output/book_ocr.txt --sample-pages 3
    python -m farsi_book_ocr.correct_text output/book_ocr.txt --estimate-only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

import httpx

# Load .env from project root (parent of this file's package directory)
_load_dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_load_dotenv_path)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Matches "===== PAGE 0001 =====" and "===== OCR CHUNK 0001: ... =====" lines
_SEPARATOR_RE = re.compile(
    r"\n{2,}===== (?:PAGE|OCR CHUNK) \d{4}.*?=====\n{2,}"
)

# Fallback for files without the standard separator
_PARAGRAPH_BREAK_RE = re.compile(r"\n{3,}")

# Token estimation: Persian ~ 1 token per 2 chars (conservative)
_CHARS_PER_TOKEN = 2.0

# Pricing in USD per 1M tokens (conservative estimates for DeepSeek V4)
# Verify at: https://api-docs.deepseek.com/quick_start/pricing
_DEFAULT_PRICE_PER_1M_INPUT = 0.14
_DEFAULT_PRICE_PER_1M_OUTPUT = 0.28

# API configuration
_DEFAULT_MAX_TOKENS = 0  # 0 = auto-calculate from input size
_MAX_OUTPUT_TOKENS = 65536  # hard ceiling for auto-calculation
_DEFAULT_TIMEOUT = 180  # seconds
_MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# System prompt for Persian OCR correction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a Persian OCR correction assistant.

Your task: Fix OCR errors in the provided Persian text.

Common Persian OCR errors to correct:
- Dotted character confusion: پ↔ب, ت↔ث, ج↔چ↔ح↔خ, ژ↔ز, ف↔ق, etc.
- Arabic character forms that should be Persian: ي→ی, ك→ک, ى→ی, ة→ه, etc.
- Incorrect word segmentation due to connected Persian script
- Spacing and line-break artifacts from OCR scanning
- Misrecognized numbers (Persian ۰۱۲۳۴۵۶۷۸۹ vs Arabic ٠١٢٣٤٥٦٧٨٩)
- Spurious tatweel/kashida (ـ) characters
- RTL/LTR mark characters left over from PDF extraction

IMPORTANT RULES:
1. Return ONLY the corrected text — no explanations, no commentary, no greetings, no markdown fences
2. Preserve ALL "===== PAGE XXXX =====" or "===== OCR CHUNK XXXX =====" separators exactly as they appear in the input
3. Preserve the original meaning and structure — do not summarize, paraphrase, or improve the writing
4. If a word is ambiguous, choose the most likely reading based on Persian spelling conventions
5. Keep the same paragraph breaks and overall layout"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_env_or_die(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"Environment variable {name} is not set.\n"
            "Set it before running:\n"
            f"  export {name}=<value>"
        )
    return value


def _estimate_tokens(text: str) -> int:
    """Rough token count for Persian text. ~2 chars per token is conservative."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


@dataclass
class _Batch:
    index: int
    segments: list[str]
    label: str

    @property
    def text(self) -> str:
        return "".join(self.segments)


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------

def _detect_separator(text: str) -> re.Pattern | None:
    """Check whether the text uses standard OCR chunk/page separators."""
    if _SEPARATOR_RE.search(text):
        return _SEPARATOR_RE
    return None


def _split_into_segments(text: str, sep_re: re.Pattern | None) -> list[str]:
    """Split text into segments, each starting with its separator (if any)."""
    if sep_re is not None:
        # Split on separators, keep them attached to the following segment
        parts = sep_re.split(text)
        # Re-attach separator to each segment (except the first which is before any separator)
        segments: list[str] = []
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            # Find the separator that was between this part and the previous
            if i > 0:
                m = sep_re.search(text)
                # We need a different approach — use finditer instead
                pass
        # Simpler: use finditer to get positions, then slice
        matches = list(sep_re.finditer(text))
        if not matches:
            return [text] if text.strip() else []

        result: list[str] = []
        # Text before first separator (usually empty / preamble)
        preamble = text[: matches[0].start()].strip()
        if preamble:
            result.append(preamble)

        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            # Combine separator + body
            segment = m.group() + (body if body else "")
            result.append(segment.strip())

        return result

    # No standard separator — fall back to paragraph breaks
    parts = _PARAGRAPH_BREAK_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def _group_into_batches(
    segments: list[str],
    pages_per_request: int,
) -> list[_Batch]:
    """Group all segments into batches for correction."""
    batches: list[_Batch] = []
    for i in range(0, len(segments), pages_per_request):
        batch_segs = segments[i : i + pages_per_request]
        start = i + 1
        end = min(i + len(batch_segs), len(segments))
        batches.append(
            _Batch(
                index=len(batches) + 1,
                segments=batch_segs,
                label=f"batch_{len(batches) + 1:04d}_seg{start:04d}-{end:04d}",
            )
        )
    return batches


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def _build_client() -> httpx.Client:
    base_url = _get_env_or_die("ANTHROPIC_BASE_URL")
    auth_token = _get_env_or_die("ANTHROPIC_AUTH_TOKEN")

    # Normalise: strip trailing slash so we can append /v1/messages cleanly
    base_url = base_url.rstrip("/")

    return httpx.Client(
        base_url=base_url,
        headers={
            "x-api-key": auth_token,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        timeout=_DEFAULT_TIMEOUT,
    )


def _call_api(
    client: httpx.Client,
    system_prompt: str,
    user_message: str,
    model: str,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> str:
    """Send a single correction request. Returns the corrected text."""
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_message},
        ],
    }

    last_error: Exception | None = None
    for attempt in range(1 + _MAX_RETRIES):
        try:
            response = client.post("/v1/messages", json=payload)
            response.raise_for_status()
            body = response.json()
            # Find the text content block (skip thinking/reasoning blocks)
            content = body.get("content", [])
            text_blocks = [b.get("text", "") for b in content if b.get("type") == "text"]
            if not text_blocks:
                raise RuntimeError("API returned no text content block")
            return "".join(text_blocks)
        except (httpx.HTTPError, httpx.TimeoutException, RuntimeError) as exc:
            last_error = exc
            if attempt < _MAX_RETRIES:
                wait = 2 ** attempt
                print(f"  API error: {exc}. Retrying in {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise RuntimeError(f"API request failed after {_MAX_RETRIES + 1} attempts") from last_error

    # Should be unreachable
    raise RuntimeError("Unexpected: all retries exhausted") from last_error


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def _print_estimate(
    input_text: str,
    pages_per_request: int,
    model: str,
) -> None:
    segments = _split_into_segments(input_text, _detect_separator(input_text))
    batches = _group_into_batches(segments, pages_per_request)

    # Work tokens: sum of all batch texts
    work_chars = sum(len(b.text) for b in batches)
    total_input_tokens = max(1, int(work_chars / _CHARS_PER_TOKEN))
    # Output: rough estimate — typically similar length to input
    total_output_tokens = total_input_tokens

    input_cost = (total_input_tokens / 1_000_000) * _DEFAULT_PRICE_PER_1M_INPUT
    output_cost = (total_output_tokens / 1_000_000) * _DEFAULT_PRICE_PER_1M_OUTPUT

    print(f"\n{'='*60}")
    print(f"Cost estimate for LLM OCR correction")
    print(f"{'='*60}")
    print(f"Model:            {model}")
    print(f"Input file:       {len(input_text):,} chars")
    print(f"Segments found:   {len(segments)}")
    print(f"Pages per batch:  {pages_per_request}")
    print(f"Number of batches:{len(batches)}")
    print(f"")
    print(f"Estimated input tokens:  {total_input_tokens:,}")
    print(f"Estimated output tokens: {total_output_tokens:,}")
    print(f"")
    print(f"Price rates used (per 1M tokens):")
    print(f"  Input:  ${_DEFAULT_PRICE_PER_1M_INPUT:.2f}")
    print(f"  Output: ${_DEFAULT_PRICE_PER_1M_OUTPUT:.2f}")
    print(f"")
    print(f"Estimated input cost:  ${input_cost:.4f}")
    print(f"Estimated output cost: ${output_cost:.4f}")
    print(f"Estimated total cost:  ${input_cost + output_cost:.4f}")
    print(f"")
    print(f"Note: Token counts are rough estimates (Persian ≈ {_CHARS_PER_TOKEN:.0f} chars/token).")
    print(f"      Verify pricing at https://api-docs.deepseek.com/quick_start/pricing")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Correction pipeline
# ---------------------------------------------------------------------------

def _correct_batch(
    client: httpx.Client,
    batch: _Batch,
    system_prompt: str,
    work_dir: Path,
    model: str,
    max_tokens: int,
) -> Path:
    """Correct one batch. Returns path to the corrected output file."""
    out_file = work_dir / f"{batch.label}.txt"

    if out_file.exists() and out_file.stat().st_size > 0:
        print(f"  Skipping completed batch: {batch.label}")
        return out_file

    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(".tmp.txt")

    # Auto-calculate max_tokens if not set: 2× estimated input tokens with a ceiling
    if max_tokens <= 0:
        est_input = _estimate_tokens(batch.text)
        max_tokens = max(8192, min(est_input * 2, _MAX_OUTPUT_TOKENS))
        print(f"  Processing {batch.label}: {len(batch.segments)} segment(s), "
              f"~{len(batch.text):,} chars (max_tokens={max_tokens:,})...", flush=True)
    else:
        print(f"  Processing {batch.label}: {len(batch.segments)} segment(s), "
              f"~{len(batch.text):,} chars...", flush=True)

    corrected = _call_api(
        client,
        system_prompt=system_prompt,
        user_message=batch.text,
        model=model,
        max_tokens=max_tokens,
    )

    tmp.write_text(corrected, encoding="utf-8")
    tmp.replace(out_file)
    return out_file


def _merge_batches(batch_files: list[Path], output_path: Path) -> None:
    """Merge corrected batch files into the final output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp.txt")

    with tmp.open("w", encoding="utf-8") as out:
        for bf in batch_files:
            text = bf.read_text(encoding="utf-8", errors="replace")
            out.write(text)
            out.write("\n\n")

    tmp.replace(output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correct Persian OCR text using an LLM (DeepSeek via Anthropic API proxy)."
    )
    parser.add_argument(
        "input", type=Path,
        help="OCR'd .txt file (with PAGE or OCR CHUNK separators)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Corrected output path (default: <input_stem>.corrected.txt)",
    )
    parser.add_argument(
        "--pages-per-request", type=int, default=2,
        help="Number of pages/chunks per API call",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=_DEFAULT_MAX_TOKENS,
        help="Max tokens per API response (0 = auto-calculate)",
    )
    parser.add_argument(
        "--model", type=str,
        default=os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-pro").replace("[1m]", ""),
        help="Model name (also reads ANTHROPIC_MODEL env var)",
    )
    parser.add_argument(
        "--estimate-only", action="store_true",
        help="Print token and cost estimate, then exit without calling the API",
    )
    parser.add_argument(
        "--redo", action="store_true",
        help="Delete previous correction work and start over",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("work"),
        help="Intermediate work directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    input_text = input_path.read_text(encoding="utf-8", errors="replace")
    if not input_text.strip():
        raise SystemExit("Input file is empty.")

    # ---- estimate-only mode ----
    if args.estimate_only:
        _print_estimate(
            input_text,
            args.pages_per_request,
            args.model,
        )
        return 0

    # ---- split into segments ----
    sep_re = _detect_separator(input_text)
    segments = _split_into_segments(input_text, sep_re)
    if sep_re is None:
        print("Warning: No standard PAGE/OCR CHUNK separators found. "
              "Treating as a single segment.")

    print(f"Input:    {input_path}")
    print(f"Segments: {len(segments)}")
    print(f"Model:    {args.model}")

    # ---- group into batches ----
    batches = _group_into_batches(segments, args.pages_per_request)
    print(f"Batches:  {len(batches)}")

    # ---- prepare work dir ----
    book_name = input_path.stem
    work_dir = args.work_dir / book_name / "corrected"
    if args.redo and work_dir.exists():
        import shutil
        shutil.rmtree(work_dir)

    # ---- build API client ----
    client = _build_client()

    # ---- process each batch ----
    batch_files: list[Path] = []
    failed: list[str] = []
    for batch in batches:
        try:
            bf = _correct_batch(client, batch, _SYSTEM_PROMPT, work_dir, args.model, args.max_tokens)
            batch_files.append(bf)
        except Exception as exc:
            print(f"  FAILED: {batch.label} — {exc}", flush=True)
            failed.append(batch.label)

    client.close()

    # ---- merge ----
    if not batch_files:
        raise SystemExit("No batches were processed successfully.")

    output_path = args.output or input_path.with_stem(input_path.stem + ".corrected")
    output_path = output_path.expanduser().resolve()
    _merge_batches(batch_files, output_path)
    print(f"\nCorrected text written to: {output_path}")

    if failed:
        print(f"\nWarning: {len(failed)} batch(es) failed: {', '.join(failed)}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
