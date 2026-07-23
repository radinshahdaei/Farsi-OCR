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
# 1. OCR a scanned PDF
python -m farsi_book_ocr.ocr_book input/book.pdf

# 2. Normalize the OCR output
python -m farsi_book_ocr.normalize_text output/book_ocr.txt output/book_normalized.txt
```

Output: searchable PDF + plain text.

## LLM Correction (in development)

The automated LLM correction pipeline is still under development, but you can try it:

```bash
# 3. LLM correction (experimental — requires .env with API key)
python -m farsi_book_ocr.correct_text output/book_normalized.txt
```
For now, the recommended workflow is:

1. Generate the OCR output (Stage 1 above)
2. Normalize the text (Stage 2 above)
3. Feed the normalized output into a chatbot (ChatGPT, etc.) for manual correction.

### Motivation

Very large scanned books with unreadable PDF text are difficult for chatbots to
process directly — the file sizes are too large and the raw scanned content is
not recognizable as text. This pipeline solves that by first running OCR to
extract readable plain text, then normalizing it for Persian-specific issues,
so the output is small enough and clean enough to feed into a chatbot for
correction.

## Pipeline

| Stage | Module | Description |
| --- | --- | --- |
| OCR | `ocr_book.py` | Splits PDF into chunks, runs OCRmyPDF+Tesseract, merges results |
| Normalize | `normalize_text.py` | Unicode NFC, strip invisible chars, preserve layout (Arabic-safe default) |
| Correct | `correct_text.py` | Page-safe LLM correction with validation, retry, and caching |
