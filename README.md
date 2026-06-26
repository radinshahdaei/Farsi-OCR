# Farsi/Persian Book OCR Pipeline for macOS Apple Silicon

This project OCRs a long Persian/Farsi scanned book PDF on a Mac M1/M2/M3.

It uses:

- **OCRmyPDF** to add a searchable OCR layer to PDF pages.
- **Tesseract** with the Persian language model `fas`.
- A Python wrapper that splits a long PDF into chunks, OCRs each chunk, resumes after failures, and merges the output back into one searchable PDF plus one `.txt` file.

## 1. Install system tools

Install Homebrew first if you do not already have it.

Then:

```bash
brew update
brew install ocrmypdf
brew install tesseract-lang
```

Check that Persian/Farsi is installed:

```bash
tesseract --list-langs | grep fas
```

You should see:

```text
fas
```

## 2. Create a Python virtual environment

From this project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Put your book in the project folder

Example:

```bash
mkdir -p input
cp /path/to/your/book.pdf input/book.pdf
```

## 4. Run OCR

For a Persian-only book:

```bash
python -m farsi_book_ocr.ocr_book input/book.pdf --lang fas --jobs 4 --pages-per-chunk 25
```

For Persian + English:

```bash
python -m farsi_book_ocr.ocr_book input/book.pdf --lang fas+eng --jobs 4 --pages-per-chunk 25
```

Outputs will be written to:

```text
output/book_ocr.pdf
output/book_ocr.txt
work/book/
```

## 5. Resume after interruption

Just run the same command again. Finished chunks are skipped automatically.

```bash
python -m farsi_book_ocr.ocr_book input/book.pdf --lang fas+eng --jobs 4 --pages-per-chunk 25
```

## 6. Start over

Use `--redo` to delete previous chunk results for this book and rerun all chunks.

```bash
python -m farsi_book_ocr.ocr_book input/book.pdf --lang fas+eng --jobs 4 --pages-per-chunk 25 --redo
```

## Recommended settings for a 600-page book

Start with:

```bash
python -m farsi_book_ocr.ocr_book input/book.pdf \
  --lang fas+eng \
  --jobs 4 \
  --pages-per-chunk 25 \
  --deskew \
  --rotate-pages
```

For an M1 MacBook Air, keep `--jobs 3` or `--jobs 4` to avoid overheating. For an M1 Pro/Max or desktop Mac, try `--jobs 6` or `--jobs 8`.

## Accuracy tips

- Best input: 300 DPI grayscale or black-and-white scans.
- Use `--deskew` for crooked scans.
- Use `--rotate-pages` if some pages are sideways or upside down.
- Use `--lang fas+eng` if the book has page headers, footnotes, citations, or numbers in English.
- Avoid aggressive cleaning unless needed. It can sometimes damage dots/diacritics in Persian text.

## Useful extra commands

Run only selected pages first, for a test:

```bash
python -m farsi_book_ocr.ocr_book input/book.pdf --first-page 1 --last-page 10 --lang fas+eng --jobs 4
```

Extract plain text from an already OCRed PDF:

```bash
python -m farsi_book_ocr.extract_text output/book_ocr.pdf output/book_extracted.txt
```

Normalize Persian text after OCR:

```bash
python -m farsi_book_ocr.normalize_text output/book_ocr.txt output/book_ocr.normalized.txt
```

## Troubleshooting

### `fas` not found

Run:

```bash
brew install tesseract-lang
brew reinstall tesseract-lang
```

Then check:

```bash
tesseract --list-langs | grep fas
```

### OCRmyPDF says a PDF already has text

This project passes `--skip-text` by default. If you want OCR to replace existing text, use:

```bash
python -m farsi_book_ocr.ocr_book input/book.pdf --force-ocr
```

### A chunk keeps failing

Try smaller chunks:

```bash
python -m farsi_book_ocr.ocr_book input/book.pdf --pages-per-chunk 10 --lang fas+eng --jobs 3
```

Then inspect the failed chunk in `work/<book-name>/chunks/`.

