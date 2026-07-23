# Farsi Book OCR — Developer Guide

## Overview

Three-stage pipeline for OCR-ing Persian/Farsi scanned books:

```
PDF → OCR (ocr_book.py) → Normalize (normalize_text.py) → Correct (correct_text.py)
```

| Stage | Module | Input | Output |
|-------|--------|-------|--------|
| OCR | `ocr_book.py` | Scanned PDF | Searchable PDF + plain text (`.txt` with `\f` page breaks) |
| Normalize | `normalize_text.py` | Raw OCR text | Unicode-normalized text |
| Correct | `correct_text.py` | Normalized text | LLM-corrected text (experimental) |

## Setup (development)

```bash
# System dependencies
brew install ocrmypdf tesseract-lang

# Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# API access (only needed for correction stage)
cp .env.example .env   # then edit with your API key
```

## Running the pipeline

### Stage 1: OCR

```bash
python -m farsi_book_ocr.ocr_book input/book.pdf
```

Full flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--lang` | `fas+ara+eng` | Tesseract language string |
| `--jobs` | `min(4, cpu/2)` | OCRmyPDF parallel jobs per chunk |
| `--pages-per-chunk` | `25` | Pages per chunk — lower = more resumable, higher = faster |
| `--first-page` | `1` | First 1-based page to OCR |
| `--last-page` | _end of PDF_ | Last 1-based page to OCR |
| `--output-dir` | `output` | Where final merged files land |
| `--work-dir` | `work` | Intermediate chunk storage (resumable) |
| `--deskew` | off | Straighten crooked scans |
| `--rotate-pages` | off | Auto-detect and fix page orientation |
| `--clean-final` | off | Clean page images (can hurt fine Persian marks) |
| `--force-ocr` | off | OCR even if existing text layer detected |
| `--output-type` | `pdf` | OCRmyPDF output type: `pdf`, `pdfa`, `pdfa-1`, `pdfa-2`, `pdfa-3` |
| `--tesseract-timeout` | `300` | Seconds before Tesseract gives up on a page |
| `--redo` | off | Delete existing work and start over |

Output files (in `--output-dir`):
- `<book>_ocr.pdf` — searchable PDF
- `<book>_ocr.txt` — plain text with `\f` (form feed) page separators

### Stage 2: Normalize

```bash
python -m farsi_book_ocr.normalize_text output/book_ocr.txt output/book_normalized.txt
```

| Flag | Description |
|------|-------------|
| `--persian` | Aggressive: convert Arabic→Persian letters. Use only for pure Persian texts (no Quran/Arabic quotes) |
| `--preserve-layout` | Preserve table spacing and line breaks (now the default behavior) |
| `--arabic-safe` | Preserve Arabic letter forms and collapse whitespace |

Without any flag, the default normalization applies: NFC normalization, strip kashida/bidi marks, normalize line endings — layout is preserved.

### Stage 3: LLM Correction (experimental)

```bash
python -m farsi_book_ocr.correct_text output/book_normalized.txt
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `<input>.corrected.txt` | Output path |
| `--pages-per-request` | `20` | Pages per API call (1 = safest, 20 = batching) |
| `--context-pages` | `3` | Context pages sent on each side of a batch |
| `--max-tokens` | `65536` | Max output tokens per response |
| `--model` | from `.env` | Model identifier |
| `--strict` | off | Abort on any page failure instead of using source text |
| `--redo` | off | Ignore cached results and re-correct everything |
| `--work-dir` | `work` | Cache directory |
| `--log` / `-v` | off | Verbose: show timing, tokens, and request details |

## Recommended workflow (production)

The automated LLM pipeline is still in development. For real projects, use:

1. **OCR** the book with Stage 1
2. **Normalize** with Stage 2
3. **Feed the output** into ChatGPT, Claude, or Gemini for manual correction

This is cheaper than API-based correction and works better for long books.

## Testing

```bash
# All tests
pytest

# Single file
pytest tests/test_correction_pipeline.py

# Single test
pytest tests/test_correction_pipeline.py::TestBasicCorrection::test_single_page_correction

# Lint
ruff check .

# Type check
mypy farsi_book_ocr/
```

Tests use `FakeProvider` — deterministic, no network, no API key needed.

## Architecture

### Key data types (`models.py`)

All frozen dataclasses:

- **`PageRecord`** — a single page: `document_id`, `page_id`, `page_index`, `source_text`, `source_sha256`
- **`CorrectionRecord`** — result of correcting one page: `corrected_text`, `status` (`accepted`/`fallback_raw`/`failed`), token counts, validation results
- **`CorrectionConfig`** — everything needed for a correction run: provider, model, token limits, retries, fallback policy
- **`CorrectionRequest`** — what gets sent to the LLM: page text, system prompt, model, context pages
- **`ProviderResponse`** — normalized LLM response: text, finish reason, token usage
- **`CorrectionRunResult`** — summary of a completed run: per-page records, aggregate counts, status

### Correction pipeline flow (`correction.py`)

```
Pages → Cache check → Batch → Build context → LLM call → Split response
                                                      ↓
                                            Validate each page
                                                      ↓
                                          Accept / Retry (solo) / Fallback
                                                      ↓
                                              Feed forward as context
                                                      ↓
                                              Assemble with \f joins
```

1. **Split** — pages are split on `\f` boundaries; chunk separators are stripped
2. **Cache check** — pages with matching `source_sha256` skip re-correction
3. **Batch** — pages grouped into `pages_per_request`-sized batches
4. **Context** — each batch gets `context_pages` of surrounding text (read-only)
5. **Correct** — single API call per batch, response split by page markers
6. **Validate** — each page checked for truncation and empty-when-nonempty
7. **Retry** — failed pages retried solo with exponential backoff
8. **Fallback** — exhausted pages use source text (`fallback_raw`) or fail (`strict`)
9. **Feed forward** — corrected text used as context for subsequent batches
10. **Assemble** — pages sorted by `page_index`, joined with `\f`

### Provider abstraction (`providers/`)

```
CorrectionProvider (ABC)
├── DeepSeekProvider   ← talks to DeepSeek via Anthropic-compatible Messages API
└── FakeProvider       ← deterministic test double
```

The `DeepSeekProvider` uses the Anthropic Messages API format through a proxy (`ANTHROPIC_BASE_URL`). The provider layer is decoupled from the pipeline — swap in a new provider by implementing `CorrectionProvider`.

### Caching (`cache.py`)

Content-addressed layout:

```
<work_dir>/<source_sha256[:16]>/<ocr_config[:16]>/corrections/<corr_config[:16]>/
```

Each correction run writes a `manifest.json` tracking source identity, config, tool versions, and artifact hashes.

### Validation (`validator.py`)

Minimal by design — only two checks:
1. **Not truncated** — `finish_reason` is not `max_tokens`/`length`/`truncated`
2. **Not empty** — response has text when source had text

No statistical or cosmetic checks. The philosophy: under-correction is better than over-correction, and false rejections waste API calls.

## Key invariants

- **Form feeds are the only page boundary.** Chunk separators (`===== OCR CHUNK ... =====`) are synthetic and stripped. Never treat them as page breaks.
- **Context pages are read-only.** Marked with `[[CONTEXT: ... /CONTEXT]]` in prompts. They provide continuity but are never corrected.
- **Validation is deliberately relaxed.** Only data-loss checks — no cosmetic or statistical gates.
- **Caching is invalidated by source hash change.** If a page's `source_sha256` doesn't match the cache, it's a miss.
- **Atomic writes only.** All file output uses write-to-tmp-then-atomic-rename.
- **Assembly is deterministic.** Pages sorted by `page_index`, not API response order.
