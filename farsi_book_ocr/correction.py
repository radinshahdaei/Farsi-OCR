"""Page-safe LLM correction pipeline.

Orchestrates the full correction flow: splitting, correction, validation,
retry, caching, and deterministic assembly.
"""

from __future__ import annotations

import hashlib
import json
import logging
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

logger = logging.getLogger(__name__)


def _load_prompt() -> str:
    """Load the system prompt from the packaged prompt file."""
    prompt_path = Path(__file__).resolve().parent / "prompts" / "ocr_correction_v1.txt"
    return prompt_path.read_text(encoding="utf-8")


def _prompt_hash() -> str:
    """SHA-256 of the system prompt text (for version tracking)."""
    return hashlib.sha256(_load_prompt().encode("utf-8")).hexdigest()


def _build_context_block(pages_info: list[tuple[str, str]], label: str) -> str | None:
    """Build a multi-page context block.

    Each entry is (page_id, text). Uses --- page-XXXXXX --- sub-markers
    so the model can distinguish individual context pages.
    """
    if not pages_info:
        return None

    lines = [f"[[CONTEXT: {label} — READ ONLY, DO NOT CORRECT]]"]
    for pid, text in pages_info:
        lines.append(f"--- {pid} ---")
        lines.append(text)
    lines.append("[[/CONTEXT]]")
    return "\n".join(lines)


def _build_user_message(
    page: PageRecord,
    context_before: list[tuple[str, str]] | None,
    context_after: list[tuple[str, str]] | None,
) -> str:
    """Build the user message for a single-page correction request."""
    parts: list[str] = []

    ctx_before_block = _build_context_block(context_before or [], "preceding pages")
    if ctx_before_block:
        parts.append(ctx_before_block)

    parts.append(
        f"[[PAGE {page.page_id} — CORRECT THIS PAGE]]\n"
        f"{page.source_text}\n"
        f"[[/PAGE {page.page_id}]]"
    )

    ctx_after_block = _build_context_block(context_after or [], "following pages")
    if ctx_after_block:
        parts.append(ctx_after_block)

    return "\n\n".join(parts)


def _build_batch_message(
    pages: list[PageRecord],
    context_before: list[tuple[str, str]] | None,
    context_after: list[tuple[str, str]] | None,
) -> str:
    """Build a user message that corrects multiple pages in one request.

    Each page is wrapped in [[PAGE ...]] markers. The model must preserve
    these markers exactly in its response.

    Context before/after: list of (page_id, text) tuples. Multiple context
    pages are joined into a single [[CONTEXT ... /CONTEXT]] block with
    --- page-XXXXXX --- sub-markers.
    """
    parts: list[str] = []

    ctx_before_block = _build_context_block(context_before or [], "preceding pages")
    if ctx_before_block:
        parts.append(ctx_before_block)

    for page in pages:
        parts.append(
            f"[[PAGE {page.page_id} — CORRECT THIS PAGE]]\n"
            f"{page.source_text}\n"
            f"[[/PAGE {page.page_id}]]"
        )

    ctx_after_block = _build_context_block(context_after or [], "following pages")
    if ctx_after_block:
        parts.append(ctx_after_block)

    return "\n\n".join(parts)


def _split_batch_response(response_text: str, pages: list[PageRecord]) -> dict[str, str]:
    """Split a multi-page correction response back into per-page text.

    Uses [[/PAGE page-XXXXXX]] markers as delimiters. If a page is missing
    from the response, its entry will be empty.

    Args:
        response_text: The full model response containing all corrected pages.
        pages: The pages that were sent in the batch.

    Returns:
        Dict mapping page_id -> corrected text for that page.
    """
    result: dict[str, str] = {}

    for page in pages:
        end_marker = f"[[/PAGE {page.page_id}]]"
        start_marker = f"[[PAGE {page.page_id}"

        # Find this page's section in the response
        start_idx = response_text.find(start_marker)
        if start_idx == -1:
            result[page.page_id] = ""
            continue

        end_idx = response_text.find(end_marker, start_idx)
        if end_idx == -1:
            # No end marker — take everything after start marker
            # Skip past the start marker line itself
            content_start = response_text.find("\n", start_idx)
            if content_start == -1:
                result[page.page_id] = ""
            else:
                result[page.page_id] = response_text[content_start + 1:].strip()
            continue

        # Extract content between the start marker line and end marker
        content_start = response_text.find("\n", start_idx)
        if content_start == -1 or content_start > end_idx:
            result[page.page_id] = ""
        else:
            result[page.page_id] = response_text[content_start + 1:end_idx].strip()

    return result


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


def _correct_single_page(
    provider: CorrectionProvider,
    page: PageRecord,
    config: CorrectionConfig,
    context_before: list[tuple[str, str]] | None,
    context_after: list[tuple[str, str]] | None,
    work_dir: Path,
) -> CorrectionRecord:
    """Correct a single page with retry logic."""
    system_prompt = _load_prompt()
    prompt_version = _prompt_hash()

    user_message = _build_user_message(page, context_before, context_after)
    estimated_tokens = max(1, len(user_message) // 2)
    # DeepSeek thinking tokens eat ~half the budget — be generous
    max_tokens = max(32768, min(estimated_tokens * 4, config.max_output_tokens))

    # Flatten context for the CorrectionRequest (kept for backward compat)
    ctx_before_str = "\f".join(t for _, t in (context_before or [])) or None
    ctx_after_str = "\f".join(t for _, t in (context_after or [])) or None

    request = CorrectionRequest(
        page_id=page.page_id,
        source_text=page.source_text,
        system_prompt=system_prompt,
        model=config.model,
        max_tokens=max_tokens,
        temperature=config.temperature,
        context_before=ctx_before_str,
        context_after=ctx_after_str,
    )

    for attempt in range(1, config.max_retries + 1):
        try:
            t0 = time.monotonic()
            response = provider.correct(request)
            elapsed = time.monotonic() - t0

            logger.info(
                "%s API call #%d: %d→%d tokens, finish=%s, %.1fs",
                page.page_id, attempt,
                response.usage.input_tokens if response.usage else 0,
                response.usage.output_tokens if response.usage else 0,
                response.finish_reason, elapsed,
            )

            validation = validate_correction_response(page, response)
            check_names = [name for name, passed, _ in validation.checks if not passed]

            if validation.passed:
                return CorrectionRecord(
                    page_id=page.page_id,
                    source_sha256=page.source_sha256,
                    corrected_text=response.text,
                    status="accepted",
                    attempts=attempt,
                    provider=provider.provider_name,
                    model=config.model,
                    prompt_version=prompt_version,
                    finish_reason=response.finish_reason,
                    input_tokens=response.usage.input_tokens if response.usage else None,
                    output_tokens=response.usage.output_tokens if response.usage else None,
                    validation_results=check_names,
                    output_sha256=PageRecord.compute_sha256(response.text),
                    error=None,
                )

            detail = "; ".join(
                f"{name}: {detail}"
                for name, passed, detail in validation.checks
                if not passed
            )
            logger.warning(
                "%s validation failed (attempt %d/%d): %s",
                page.page_id, attempt, config.max_retries, detail,
            )
            print(
                f"  Validation failed for {page.page_id} (attempt {attempt}): {detail}",
                flush=True,
            )

        except Exception as exc:
            logger.warning(
                "%s error (attempt %d/%d): %s",
                page.page_id, attempt, config.max_retries, exc,
            )
            print(
                f"  Error correcting {page.page_id} (attempt {attempt}): {exc}",
                flush=True,
            )
            if attempt < config.max_retries:
                wait = min(2 ** attempt, 60)
                print(f"  Retrying in {wait}s...", flush=True)
                time.sleep(wait)

    logger.warning("%s: all %d attempts exhausted, falling back", page.page_id, config.max_retries)
    return _fallback_or_fail(page, config, prompt_version, provider.provider_name)


def _correct_batch(
    provider: CorrectionProvider,
    pages: list[PageRecord],
    config: CorrectionConfig,
    context_before: list[tuple[str, str]] | None,
    context_after: list[tuple[str, str]] | None,
    work_dir: Path,
) -> list[CorrectionRecord]:
    """Correct multiple pages in a single API call, then split and validate each.

    On retry, individual pages that failed validation are retried solo to avoid
    re-sending pages that were already corrected successfully.
    """
    system_prompt = _load_prompt()
    prompt_version = _prompt_hash()

    user_message = _build_batch_message(pages, context_before, context_after)
    estimated_tokens = max(1, len(user_message) // 2)
    # Batch mode: use 4× input as output budget (thinking tokens eat ~half)
    max_tokens = max(32768, min(estimated_tokens * 4, config.max_output_tokens))

    # Flatten context for the CorrectionRequest (kept for backward compat)
    ctx_before_str = "\f".join(t for _, t in (context_before or [])) or None
    ctx_after_str = "\f".join(t for _, t in (context_after or [])) or None

    # Build a single request covering all pages in the batch
    batch_request = CorrectionRequest(
        page_id="batch",
        source_text=user_message,
        system_prompt=system_prompt,
        model=config.model,
        max_tokens=max_tokens,
        temperature=config.temperature,
        context_before=ctx_before_str,
        context_after=ctx_after_str,
    )

    records: dict[str, CorrectionRecord] = {}
    pending = {p.page_id: p for p in pages}

    for attempt in range(1, config.max_retries + 1):
        if not pending:
            break

        try:
            t0 = time.monotonic()
            response = provider.correct(batch_request)
            elapsed = time.monotonic() - t0

            logger.info(
                "batch API call #%d: %d→%d tokens, finish=%s, req=%s, %.1fs",
                attempt,
                response.usage.input_tokens if response.usage else 0,
                response.usage.output_tokens if response.usage else 0,
                response.finish_reason,
                response.request_id or "-",
                elapsed,
            )
            corrected_texts = _split_batch_response(response.text, pages)

            # Validate each page individually
            still_pending: dict[str, PageRecord] = {}
            for page_id, page in pending.items():
                corrected = corrected_texts.get(page_id, "")
                finish_reason = response.finish_reason
                input_tokens = response.usage.input_tokens if response.usage else None
                output_tokens = response.usage.output_tokens if response.usage else None

                if not corrected:
                    if attempt < config.max_retries:
                        still_pending[page_id] = page
                    else:
                        records[page_id] = _fallback_or_fail(
                            page, config, prompt_version, provider.provider_name
                        )
                    continue

                # Validate
                from farsi_book_ocr.models import ProviderResponse, ProviderUsage

                pv_resp = ProviderResponse(
                    text=corrected,
                    finish_reason=finish_reason,
                    usage=ProviderUsage(
                        input_tokens=input_tokens or 0,
                        output_tokens=output_tokens or 0,
                    ) if (input_tokens or output_tokens) else None,
                    request_id=response.request_id,
                    raw_status_code=response.raw_status_code,
                )
                validation = validate_correction_response(page, pv_resp)
                check_names = [n for n, p, _ in validation.checks if not p]

                if validation.passed:
                    src_len = len(page.source_text)
                    corr_len = len(corrected)
                    delta_pct = abs(corr_len - src_len) / max(1, src_len) * 100
                    logger.info(
                        "%s: accepted (%.0f%% length change)",
                        page_id, delta_pct,
                    )
                    records[page_id] = CorrectionRecord(
                        page_id=page.page_id,
                        source_sha256=page.source_sha256,
                        corrected_text=corrected,
                        status="accepted",
                        attempts=attempt,
                        provider=provider.provider_name,
                        model=config.model,
                        prompt_version=prompt_version,
                        finish_reason=finish_reason,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        validation_results=check_names,
                        output_sha256=PageRecord.compute_sha256(corrected),
                        error=None,
                    )
                    print(f"  {page_id}: accepted", flush=True)
                else:
                    detail = "; ".join(
                        f"{n}: {d}" for n, _, d in validation.checks if not _
                    )
                    print(
                        f"  {page_id}: validation failed (attempt {attempt}): {detail}",
                        flush=True,
                    )
                    if attempt < config.max_retries:
                        still_pending[page_id] = page
                    else:
                        records[page_id] = _fallback_or_fail(
                            page, config, prompt_version, provider.provider_name
                        )

            pending = still_pending

            # If pages still pending, retry with solo requests for each
            if pending and attempt < config.max_retries:
                for pid, p in list(pending.items()):
                    wait = min(2 ** attempt, 60)
                    time.sleep(wait)
                    solo = _correct_single_page(
                        provider, p, config, None, None, work_dir
                    )
                    records[pid] = solo
                    del pending[pid]
                    if solo.status == "accepted":
                        print(f"  {pid}: accepted on solo retry", flush=True)
                    else:
                        print(f"  {pid}: {solo.status} on solo retry", flush=True)

        except Exception as exc:
            print(
                f"  Error correcting batch (attempt {attempt}): {exc}",
                flush=True,
            )
            if attempt < config.max_retries:
                wait = min(2 ** attempt, 60)
                print(f"  Retrying in {wait}s...", flush=True)
                time.sleep(wait)

    # Any pages still pending after all retries
    for page_id, page in pending.items():
        records[page_id] = _fallback_or_fail(page, config, prompt_version, provider.provider_name)

    return [records[p.page_id] for p in pages]


def _fallback_or_fail(
    page: PageRecord,
    config: CorrectionConfig,
    prompt_version: str,
    provider_name: str,
) -> CorrectionRecord:
    """Return a fallback or failed record for a page that couldn't be corrected."""
    if config.fallback_policy == "fallback_raw":
        return CorrectionRecord(
            page_id=page.page_id,
            source_sha256=page.source_sha256,
            corrected_text=page.source_text,
            status="fallback_raw",
            attempts=config.max_retries,
            provider=provider_name,
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
            provider=provider_name,
            model=config.model,
            prompt_version=prompt_version,
            finish_reason=None,
            input_tokens=None,
            output_tokens=None,
            validation_results=[],
            output_sha256=None,
            error="All correction attempts failed",
        )


def _get_page_text(
    page_index: int,
    pages: list[PageRecord],
    corrected_context: dict[int, str],
) -> str | None:
    """Get the best available text for a page index.

    Returns corrected text if available, otherwise raw source text.
    Returns None if the index is out of bounds.
    """
    if page_index < 0 or page_index >= len(pages):
        return None
    return corrected_context.get(page_index, pages[page_index].source_text)


def _build_context_list(
    start_idx: int,
    direction: int,
    count: int,
    pages: list[PageRecord],
    corrected_context: dict[int, str],
) -> list[tuple[str, str]]:
    """Build a context list of (page_id, text) tuples.

    Walks from start_idx in the given direction (-1 for before, +1 for after),
    collecting up to `count` pages. Uses corrected text when available,
    raw source otherwise.
    """
    ctx: list[tuple[str, str]] = []
    idx = start_idx
    while 0 <= idx < len(pages) and len(ctx) < count:
        text = corrected_context.get(idx, pages[idx].source_text)
        if text:
            ctx.append((pages[idx].page_id, text))
        idx += direction
    # Context before should be in document order (reverse the backward walk)
    if direction == -1:
        ctx.reverse()
    return ctx


def run_correction_pipeline(
    pages: list[PageRecord],
    provider: CorrectionProvider,
    work_dir: Path,
    config: CorrectionConfig,
    *,
    resume: bool = True,
) -> CorrectionRunResult:
    """Run the page-safe correction pipeline.

    Groups pages into batches of config.pages_per_request. Each batch gets
    config.context_pages of surrounding context. Corrected text from completed
    batches is fed forward as context for subsequent batches, so the LLM sees
    its own spelling/style choices and maintains consistency across the document.

    Args:
        pages: Pages to correct, in document order.
        provider: LLM provider instance.
        work_dir: Directory for cache and intermediate files.
        config: Correction configuration.
        resume: If True, reuse cached results with matching source hashes.

    Returns:
        CorrectionRunResult with per-page records and summary statistics.
    """
    results: list[tuple[int, CorrectionRecord]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    batch_size = max(1, config.pages_per_request)
    ctx_pages = max(0, config.context_pages)

    # Track corrected text by page_index — used to feed forward as context.
    # Maps page_index -> corrected_text (or source_text for fallback pages).
    corrected_context: dict[int, str] = {}

    # Phase 1: collect cached results and populate corrected_context from them
    uncached: list[tuple[int, PageRecord]] = []
    for i, page in enumerate(pages):
        if resume:
            cached = _read_cached_record(page, work_dir)
            if cached is not None:
                print(
                    f"  Using cached result for {page.page_id} (status={cached.status})",
                    flush=True,
                )
                results.append((i, cached))
                if cached.input_tokens:
                    total_input_tokens += cached.input_tokens
                if cached.output_tokens:
                    total_output_tokens += cached.output_tokens
                # Seed corrected_context from cache for forward-feed
                if cached.corrected_text:
                    corrected_context[i] = cached.corrected_text
                elif cached.status == "fallback_raw":
                    corrected_context[i] = page.source_text
                logger.info(
                    "cache hit: %s (status=%s, in=%d, out=%d)",
                    page.page_id, cached.status,
                    cached.input_tokens or 0, cached.output_tokens or 0,
                )
                continue
        uncached.append((i, page))

    if not uncached:
        # Everything was cached — no API calls needed
        results.sort(key=lambda item: item[0])
        final_records = [r for _, r in results]
        return CorrectionRunResult(
            pages=final_records,
            status="completed",
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
        )

    if batch_size == 1:
        # Single-page mode
        for i, page in uncached:
            ctx_before = _build_context_list(
                i - 1, -1, ctx_pages, pages, corrected_context
            ) if ctx_pages > 0 else None
            ctx_after = _build_context_list(
                i + 1, 1, ctx_pages, pages, corrected_context
            ) if ctx_pages > 0 else None

            ctx_before_ids = [pid for pid, _ in (ctx_before or [])]
            ctx_after_ids = [pid for pid, _ in (ctx_after or [])]
            logger.info(
                "%s: correcting, ctx_before=%s, ctx_after=%s",
                page.page_id, ctx_before_ids, ctx_after_ids,
            )

            record = _correct_single_page(
                provider, page, config,
                ctx_before or None, ctx_after or None,
                work_dir,
            )
            _write_cached_record(record, work_dir)
            results.append((i, record))
            if record.input_tokens:
                total_input_tokens += record.input_tokens
            if record.output_tokens:
                total_output_tokens += record.output_tokens
            # Feed forward
            if record.corrected_text:
                corrected_context[i] = record.corrected_text
            elif record.status == "fallback_raw":
                corrected_context[i] = page.source_text
    else:
        # Multi-page batch mode
        batch_count = (len(uncached) + batch_size - 1) // batch_size
        for b in range(batch_count):
            batch_start = b * batch_size
            batch_slice = uncached[batch_start:batch_start + batch_size]
            batch_indices, batch_pages = zip(*batch_slice)
            batch_pages = list(batch_pages)

            first_idx = batch_indices[0]
            last_idx = batch_indices[-1]

            # Build context: before = corrected from previous batches,
            # after = raw source (not yet corrected)
            ctx_before = _build_context_list(
                first_idx - 1, -1, ctx_pages, pages, corrected_context
            ) if ctx_pages > 0 else None
            ctx_after = _build_context_list(
                last_idx + 1, 1, ctx_pages, pages, corrected_context
            ) if ctx_pages > 0 else None

            page_range = f"{batch_pages[0].page_id}–{batch_pages[-1].page_id}"
            ctx_info = ""
            if ctx_before:
                ctx_info += f", ctx_before={len(ctx_before)}p"
            if ctx_after:
                ctx_info += f", ctx_after={len(ctx_after)}p"
            print(
                f"  Batch {b + 1}/{batch_count} ({page_range}, "
                f"{len(batch_pages)} pages{ctx_info})...",
                flush=True,
            )

            ctx_before_ids = [pid for pid, _ in (ctx_before or [])]
            ctx_after_ids = [pid for pid, _ in (ctx_after or [])]
            logger.info(
                "batch %d/%d: %d pages [%s–%s], ctx_before=%s, ctx_after=%s",
                b + 1, batch_count, len(batch_pages),
                batch_pages[0].page_id, batch_pages[-1].page_id,
                ctx_before_ids, ctx_after_ids,
            )

            t_batch = time.monotonic()
            batch_records = _correct_batch(
                provider, batch_pages, config,
                ctx_before or None, ctx_after or None,
                work_dir,
            )

            for (i, _), record in zip(batch_slice, batch_records):
                _write_cached_record(record, work_dir)
                results.append((i, record))
                if record.input_tokens:
                    total_input_tokens += record.input_tokens
                if record.output_tokens:
                    total_output_tokens += record.output_tokens
                # Feed forward for subsequent batches
                if record.corrected_text:
                    corrected_context[i] = record.corrected_text
                elif record.status == "fallback_raw":
                    corrected_context[i] = pages[i].source_text
                logger.info("cached %s → %s", record.page_id, work_dir)

            batch_elapsed = time.monotonic() - t_batch
            accepted = sum(1 for r in batch_records if r.status == "accepted")
            fallback = sum(1 for r in batch_records if r.status == "fallback_raw")
            logger.info(
                "batch %d/%d done: %d accepted, %d fallback, %.1fs",
                b + 1, batch_count, accepted, fallback, batch_elapsed,
            )

    # Sort by original page index and strip index
    results.sort(key=lambda item: item[0])
    final_records = [r for _, r in results]

    # Determine overall status
    has_failed = any(r.status == "failed" for r in final_records)
    has_fallbacks = any(r.status == "fallback_raw" for r in final_records)

    if has_failed and config.fallback_policy == "strict":
        run_status = "failed"
    elif has_fallbacks:
        run_status = "completed_with_fallbacks"
    else:
        run_status = "completed"

    accepted = sum(1 for r in final_records if r.status == "accepted")
    fallback = sum(1 for r in final_records if r.status == "fallback_raw")
    failed = sum(1 for r in final_records if r.status == "failed")
    logger.info(
        "run complete: status=%s, accepted=%d, fallback=%d, failed=%d, "
        "tokens in=%d out=%d",
        run_status, accepted, fallback, failed,
        total_input_tokens, total_output_tokens,
    )

    return CorrectionRunResult(
        pages=final_records,
        status=run_status,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
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
