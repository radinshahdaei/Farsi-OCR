#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
python -m farsi_book_ocr.ocr_book input/book.pdf --lang fas+eng --jobs 4 --pages-per-chunk 25 --deskew --rotate-pages
