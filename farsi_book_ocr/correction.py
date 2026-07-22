"""Page-safe LLM correction pipeline.

Orchestrates the full correction flow: splitting, correction, validation,
retry, caching, and deterministic assembly.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from farsi_book_ocr.models import (
    CorrectionConfig,
    CorrectionRecord,
    CorrectionRequest,
    CorrectionRunResult,
    PageRecord,
)
from farsi_book_ocr.providers.base import CorrectionProvider
from farsi_book_ocr.validator import validate_correction_response


def _load_prompt() -> str:
    """Load the system prompt from the packaged prompt file."""
    prompt_path = Path(__file__).resolve().parent / "prompts" / "ocr_correction_v1.txt"
    return prompt_path.read_text(encoding="utf-8")


def _prompt_hash() -> str:
    """SHA-256 of the system prompt text (for version tracking)."""
    return hashlib.sha256(_load_prompt().encode("utf-8")).hexdigest()


def _build_user_message(page: PageRecord, context_before: str | None, context_after: str | None) -> str:
    """Build the user message for a single-page correction request."""
    parts: list[str] = []

    if context_before:
        parts.append(
            "[[CONTEXT: preceding page — READ ONLY, DO NOT CORRECT]]\n"
            f"{context_before}\n"
            "[[/CONTEXT]]"
        )

    parts.append(
        f"[[PAGE {page.page_id} — CORRECT THIS PAGE]]\n"
        f"{page.source_text}\n"
        f"[[/PAGE {page.page_id}]]"
    )

    if context_after:
        parts.append(
            "[[CONTEXT: following page — READ ONLY, DO NOT CORRECT]]\n"
            f"{context_after}\n"
            "[[/CONTEXT]]"
        )

    return "\n\n".join(parts)


def _compute_max_tokens(request: CorrectionRequest, config: CorrectionConfig) -> int:
    """Auto-calculate max_tokens if not set."""
    if request.max_tokens > 0:
        return request.max_tokens
    # Conservative: 2× input length, with a floor of 8192
    estimated = len(request.source_text) // 2
    return max(8192, min(estimated * 2, config.max_output_tokens))


def _correction_cache_path(page_id: str, work_dir: Path) -> Path:
    """Path to the cached correction record for a page."""
    return work_dir / "corrections" / f"{page_id}.json"


def _read_cached_record(
    page: PageRecord, work_dir: Path
) -> CorrectionRecord | None:
    """Read a cached correction record if it exists and matches the source hash."""
    cache_path = _correction_cache_path(page.page_id, work_dir)
    if not cache_path.exists() or cache_path.stat().st_size == 0:
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("source_sha256") != page.source_sha256:
            return None  # Source changed — invalidate cache
        return CorrectionRecord(**data)
    except (json.JSONDecodeError, TypeError):
        return None  # Corrupted cache


def _write_cached_record(record: CorrectionRecord, work_dir: Path) -> None:
    """Atomically write a correction record to the cache."""
    cache_path = _correction_cache_path(record.page_id, work_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp.json")
    data = {
        "page_id": record.page_id,
        "source_sha256": record.source_sha256,
        "corrected_text": record.corrected_text,
        "status": record.status,
        "attempts": record.attempts,
        "provider": record.provider,
        "model": record.model,
        "prompt_version": record.prompt_version,
        "finish_reason": record.finish_reason,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "validation_results": record.validation_results,
        "output_sha256": record.output_sha256,
        "error": record.error,
    }
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(cache_path)


def _correct_one_page(
    provider: CorrectionProvider,
    page: PageRecord,
    config: CorrectionConfig,
    context_before: str | None,
    context_after: str | None,
    work_dir: Path,
) -> CorrectionRecord:
    """Correct a single page with retry logic."""
    system_prompt = _load_prompt()
    prompt_version = _prompt_hash()

    user_message = _build_user_message(page, context_before, context_after)
    estimated_tokens = provider.estimate_tokens(user_message)
    max_tokens = max(8192, min(estimated_tokens * 2, config.max_output_tokens))

    request = CorrectionRequest(
        page_id=page.page_id,
        source_text=page.source_text,
        system_prompt=system_prompt,
        model=config.model,
        max_tokens=max_tokens,
        temperature=config.temperature,
        context_before=context_before,
        context_after=context_after,
    )

    for attempt in range(1, config.max_retries + 1):
        try:
            response = provider.correct(request)

            # Validate the response
            validation = validate_correction_response(page, response)

            check_names = [name for name, passed, _ in validation.checks if not passed]

            if validation.passed:
                corrected_text = response.text
                return CorrectionRecord(
                    page_id=page.page_id,
                    source_sha256=page.source_sha256,
                    corrected_text=corrected_text,
                    status="accepted",
                    attempts=attempt,
                    provider=provider.provider_name,
                    model=config.model,
                    prompt_version=prompt_version,
                    finish_reason=response.finish_reason,
                    input_tokens=response.usage.input_tokens if response.usage else None,
                    output_tokens=response.usage.output_tokens if response.usage else None,
                    validation_results=check_names,
                    output_sha256=PageRecord.compute_sha256(corrected_text),
                    error=None,
                )

            # Validation failed — log and retry
            detail = "; ".join(
                f"{name}: {detail}"
                for name, passed, detail in validation.checks
                if not passed
            )
            print(
                f"  Validation failed for {page.page_id} (attempt {attempt}): {detail}",
                flush=True,
            )

        except Exception as exc:
            print(
                f"  Error correcting {page.page_id} (attempt {attempt}): {exc}",
                flush=True,
            )
            if attempt < config.max_retries:
                wait = min(2 ** attempt, 60)
                print(f"  Retrying in {wait}s...", flush=True)
                time.sleep(wait)

    # All attempts exhausted
    if config.fallback_policy == "fallback_raw":
        return CorrectionRecord(
            page_id=page.page_id,
            source_sha256=page.source_sha256,
            corrected_text=page.source_text,
            status="fallback_raw",
            attempts=config.max_retries,
            provider=provider.provider_name,
            model=config.model,
            prompt_version=prompt_version,
            finish_reason=None,
            input_tokens=None,
            output_tokens=None,
            validation_results=[],
            output_sha256=page.source_sha256,
            error="All correction attempts failed — using source text",
        )
    else:
        return CorrectionRecord(
            page_id=page.page_id,
            source_sha256=page.source_sha256,
            corrected_text=None,
            status="failed",
            attempts=config.max_retries,
            provider=provider.provider_name,
            model=config.model,
            prompt_version=prompt_version,
            finish_reason=None,
            input_tokens=None,
            output_tokens=None,
            validation_results=[],
            output_sha256=None,
            error="All correction attempts failed",
        )


def run_correction_pipeline(
    pages: list[PageRecord],
    provider: CorrectionProvider,
    work_dir: Path,
    config: CorrectionConfig,
    *,
    resume: bool = True,
) -> CorrectionRunResult:
    """Run the page-safe correction pipeline.

    Args:
        pages: Pages to correct, in document order.
        provider: LLM provider instance.
        work_dir: Directory for cache and intermediate files.
        config: Correction configuration.
        resume: If True, reuse cached results with matching source hashes.

    Returns:
        CorrectionRunResult with per-page records and summary statistics.
    """
    results: list[CorrectionRecord] = []
    total_input_tokens = 0
    total_output_tokens = 0

    for i, page in enumerate(pages):
        # Try cache first
        if resume:
            cached = _read_cached_record(page, work_dir)
            if cached is not None:
                print(
                    f"  Using cached result for {page.page_id} (status={cached.status})",
                    flush=True,
                )
                results.append(cached)
                if cached.input_tokens:
                    total_input_tokens += cached.input_tokens
                if cached.output_tokens:
                    total_output_tokens += cached.output_tokens
                continue

        # Get context from neighboring pages (read-only)
        context_before = pages[i - 1].source_text if i > 0 else None
        context_after = pages[i + 1].source_text if i + 1 < len(pages) else None

        record = _correct_one_page(
            provider, page, config, context_before, context_after, work_dir
        )

        # Cache the result
        _write_cached_record(record, work_dir)

        results.append(record)
        if record.input_tokens:
            total_input_tokens += record.input_tokens
        if record.output_tokens:
            total_output_tokens += record.output_tokens

    # Determine overall status
    has_failed = any(r.status == "failed" for r in results)
    has_fallbacks = any(r.status == "fallback_raw" for r in results)

    if has_failed and config.fallback_policy == "strict":
        run_status = "failed"
    elif has_fallbacks:
        run_status = "completed_with_fallbacks"
    else:
        run_status = "completed"

    return CorrectionRunResult(
        pages=sorted(results, key=lambda r: int(r.page_id.split("-")[1])),
        status=run_status,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost_estimate=None,  # Set by caller if pricing is known
    )


def assemble_corrected_text(result: CorrectionRunResult, pages: list[PageRecord]) -> str:
    """Assemble the final corrected text from correction records.

    Pages are ordered by their original page_index. Each page is separated
    by exactly one form feed. Pages with status 'failed' have their source
    text used as fallback (to produce a complete document).

    Args:
        result: The correction run result.
        pages: Original page records (used for ordering and fallback).

    Returns:
        Complete corrected text with form-feed page separators.
    """
    # Index correction records by page_id
    records_by_id: dict[str, CorrectionRecord] = {
        r.page_id: r for r in result.pages
    }

    output_parts: list[str] = []
    for page in sorted(pages, key=lambda p: p.page_index):
        record = records_by_id.get(page.page_id)
        if record is None:
            # Page was never corrected — use source text
            output_parts.append(page.source_text)
        elif record.corrected_text is not None:
            output_parts.append(record.corrected_text)
        else:
            # Failed in strict mode, but we're assembling anyway
            output_parts.append(page.source_text)

    return "\f".join(output_parts) + "\n"
