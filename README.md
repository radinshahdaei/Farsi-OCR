# Farsi Book OCR Pipeline

OCR and LLM-correct Persian/Farsi scanned books on macOS.

## Setup

```bash
# 1. Install system tools
brew install ocrmypdf tesseract-lang

# 2. Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. API credentials (for LLM correction)
cp .env.example .env
# Edit .env — add your DeepSeek API key
```

## Run

```bash
# Activate
source .venv/bin/activate

# OCR the book
python -m farsi_book_ocr.ocr_book input/book.pdf \
  --lang fas+eng --jobs 4 --pages-per-chunk 25 \
  --deskew --rotate-pages

# Estimate LLM correction cost
python -m farsi_book_ocr.correct_text output/book_ocr.txt --estimate-only

# LLM correction
python -m farsi_book_ocr.correct_text output/book_ocr.txt
```

Output: `output/book_ocr.pdf` (searchable), `output/book_ocr.txt` (text), `output/book_ocr.corrected.txt` (LLM-corrected).

## OCR Pipeline

| Flag | Default | What it does |
| --- | --- | --- |
| `--lang fas+eng` | `fas` | Tesseract language(s) |
| `--jobs 4` | auto | Parallel chunks |
| `--pages-per-chunk 25` | 25 | Pages per OCR chunk |
| `--deskew` | off | Straighten crooked scans |
| `--rotate-pages` | off | Fix page orientation |
| `--first-page / --last-page` | 1 / end | OCR a page range |
| `--redo` | off | Start fresh |

Resumable — re-run the same command to pick up where you left off.

## LLM Correction

Uses DeepSeek to fix residual Persian OCR errors (dot confusion, Arabic characters, spacing). Credentials are loaded from `.env`.

| Flag | Default | What it does |
| --- | --- | --- |
| `--pages-per-request 2` | 2 | Chunks per API call |
| `--estimate-only` | off | Show cost, no API calls |
| `--redo` | off | Re-correct all pages |

Resumable — completed batches are skipped on re-run.

**Pricing:** token estimates are approximate (Persian ≈ 2 chars/token). Hardcoded at DeepSeek V4 rates: $0.14/M input, $0.28/M output. Verify at [deepseek.com](https://api-docs.deepseek.com/quick_start/pricing).

## Utilities

```bash
# Check tools are installed
python -m farsi_book_ocr.check_install

# Rule-based Persian text normalization (no LLM)
python -m farsi_book_ocr.normalize_text output/book_ocr.txt output/book_ocr.normalized.txt

# Extract text from any searchable PDF
python -m farsi_book_ocr.extract_text output/book_ocr.pdf output/extracted.txt
```

## Sample Output

See [`sample-output/`](sample-output/) — first 100 pages of "توسعه و تضاد" by فرامرز رفیع پور, before and after LLM correction.
