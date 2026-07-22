# Farsi Book OCR

Pipeline for OCR-ing Persian/Farsi scanned books and pamphlets (جزوه) — used for
digitizing printed Islamic studies and sociology texts. Produces searchable PDFs
and plain text, with optional LLM correction to fix residual OCR errors in
Persian script.

## Setup

```bash
brew install ocrmypdf tesseract-lang
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your API key
```

## Usage

```bash
# OCR a scanned PDF
python -m farsi_book_ocr.ocr_book input/book.pdf --lang fas+ara+eng

# Estimate correction cost
python -m farsi_book_ocr.correct_text output/book_ocr.txt --estimate-only

# LLM correction
python -m farsi_book_ocr.correct_text output/book_ocr.txt

# Rule-based normalization (no LLM) — safe default, preserves Arabic
python -m farsi_book_ocr.normalize_text output/book_ocr.txt output/normalized.txt

# Aggressive normalization for pure Persian (converts Arabic→Persian letters)
python -m farsi_book_ocr.normalize_text output/book_ocr.txt output/normalized.txt --persian

# Preserve table layout and line breaks
python -m farsi_book_ocr.normalize_text output/book_ocr.txt output/normalized.txt --preserve-layout
```

Output: searchable PDF + plain text. Correction uses the page-safe pipeline — every LLM response is validated before acceptance, and no pages are silently dropped.

## Pipeline

| Stage | Module | Description |
|---|---|---|
| OCR | `ocr_book.py` | Splits PDF into chunks, runs OCRmyPDF+Tesseract, merges results |
| Normalize | `normalize_text.py` | Unicode NFC, strip invisible chars, whitespace cleanup (Arabic-safe default) |
| Correct | `correct_text.py` | Page-safe LLM correction with validation, retry, and caching |

