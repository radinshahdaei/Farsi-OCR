#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

# OCR the book
python -m farsi_book_ocr.ocr_book input/book.pdf \
  --lang fas+eng --jobs 4 --pages-per-chunk 25 \
  --deskew --rotate-pages

# Estimate cost
python -m farsi_book_ocr.correct_text output/book_ocr.txt --estimate-only

# LLM correction
python -m farsi_book_ocr.correct_text output/book_ocr.txt
