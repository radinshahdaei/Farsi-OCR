# Farsi Book OCR

OCR and LLM-correct Persian/Farsi scanned books.

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
python -m farsi_book_ocr.ocr_book input/book.pdf --lang fas+eng

# Estimate correction cost
python -m farsi_book_ocr.correct_text output/book_ocr.txt --estimate-only

# LLM correction
python -m farsi_book_ocr.correct_text output/book_ocr.txt

# Rule-based normalization (no LLM)
python -m farsi_book_ocr.normalize_text output/book_ocr.txt output/normalized.txt
```

Output: searchable PDF + plain text. Correction uses the page-safe pipeline — every LLM response is validated before acceptance, and no pages are silently dropped.

## Pipeline

| Stage | Module | Description |
|---|---|---|
| OCR | `ocr_book.py` | Splits PDF into chunks, runs OCRmyPDF+Tesseract, merges results |
| Normalize | `normalize_text.py` | Unicode NFC, Arabic→Persian letters, whitespace cleanup |
| Correct | `correct_text.py` | Page-safe LLM correction with validation, retry, and caching |

## Tests

```bash
pip install pytest
pytest  # 184 tests, zero network calls
```

## License

MIT
