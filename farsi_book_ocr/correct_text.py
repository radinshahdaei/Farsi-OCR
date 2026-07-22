"""LLM-based OCR correction for Persian text.

Uses the page-safe correction pipeline with response validation,
retry, caching, and deterministic assembly.

Example:
    python -m farsi_book_ocr.correct_text output/book_ocr.txt
    python -m farsi_book_ocr.correct_text output/book_ocr.txt --estimate-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from farsi_book_ocr.correction import assemble_corrected_text, run_correction_pipeline
from farsi_book_ocr.models import CorrectionConfig
from farsi_book_ocr.page_splitter import split_text_into_pages
from farsi_book_ocr.providers.deepseek import DeepSeekProvider

# Load .env from project root
_load_dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_load_dotenv_path)

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 2.0

# DeepSeek V4 pricing per 1M tokens (verify at deepseek.com)
_PRICE_INPUT = 0.14
_PRICE_OUTPUT = 0.28


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def estimate_cost(input_chars: int, output_chars: int) -> tuple[float, float, float]:
    input_tokens = max(1, int(input_chars / _CHARS_PER_TOKEN))
    output_tokens = max(1, int(output_chars / _CHARS_PER_TOKEN))
    input_cost = (input_tokens / 1_000_000) * _PRICE_INPUT
    output_cost = (output_tokens / 1_000_000) * _PRICE_OUTPUT
    return input_cost, output_cost, input_cost + output_cost


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correct Persian OCR text using an LLM.")
    p.add_argument("input", type=Path, help="OCR'd .txt file")
    p.add_argument("--output", type=Path, default=None, help="Output path (default: <input>.corrected.txt)")
    p.add_argument("--pages-per-request", type=int, default=1, help="Pages per API call (1 = safest)")
    p.add_argument("--max-tokens", type=int, default=0, help="Max tokens per response (0 = auto)")
    p.add_argument("--model", type=str, default=os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-pro"))
    p.add_argument("--estimate-only", action="store_true", help="Print cost estimate and exit")
    p.add_argument("--fallback", action="store_true", help="Use source text for failed pages (now the default)")
    p.add_argument("--strict", action="store_true", help="Abort on any page failure instead of using source text")
    p.add_argument("--redo", action="store_true", help="Ignore cached results and re-correct all pages")
    p.add_argument("--work-dir", type=Path, default=Path("work"), help="Cache directory")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise SystemExit("Input file is empty.")

    # ---- estimate-only ----
    if args.estimate_only:
        pages = split_text_into_pages(text, input_path.stem)
        total_chars = sum(len(p.source_text) for p in pages)
        in_cost, out_cost, total = estimate_cost(total_chars, total_chars)
        est_tokens = max(1, int(total_chars / _CHARS_PER_TOKEN))
        print(f"Pages: {len(pages)}")
        print(f"Chars: {total_chars:,}")
        print(f"Est. tokens (input+output): ~{est_tokens:,} each")
        print(f"Est. cost: ${total:.4f} (input ${in_cost:.4f} + output ${out_cost:.4f})")
        return 0

    # ---- correction ----
    pages = split_text_into_pages(text, input_path.stem)
    print(f"Input:    {input_path}")
    print(f"Pages:    {len(pages)}")

    provider = DeepSeekProvider(model=args.model.replace("[1m]", ""))
    max_out = args.max_tokens if args.max_tokens > 0 else 65536

    config = CorrectionConfig(
        provider=provider.provider_name,
        model=args.model,
        base_url=os.environ.get("ANTHROPIC_BASE_URL", ""),
        api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN", ""),
        prompt_version="v1",
        max_output_tokens=max_out,
        max_retries=3,
        pages_per_request=args.pages_per_request,
        fallback_policy="strict" if args.strict else "fallback_raw",
    )

    book_name = input_path.stem
    work_dir = args.work_dir / book_name / "corrected"
    if args.redo and work_dir.exists():
        import shutil
        shutil.rmtree(work_dir)

    result = run_correction_pipeline(pages, provider, work_dir, config, resume=not args.redo)

    if result.status == "failed":
        print("\nCorrection failed — some pages could not be corrected.")
        print("Re-run with --fallback to use source text for failed pages.")
        return 1

    assembled = assemble_corrected_text(result, pages)
    output_path = args.output or input_path.with_stem(input_path.stem + ".corrected")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(assembled, encoding="utf-8")

    print(f"\nStatus:   {result.status}")
    print(f"Accepted: {result.accepted_count}")
    if result.fallback_count:
        print(f"Fallback: {result.fallback_count}")
    print(f"Output:   {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
