#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

echo "=== Step 1/3: OCR ==="
echo "Started at $(date)"
python -m farsi_book_ocr.ocr_book sample-output-new/book.pdf \
  --lang fas+ara+eng \
  --output-dir sample-output-new \
  --work-dir sample-output-new/work

echo ""
echo "=== Step 2/3: Normalize ==="
echo "Started at $(date)"
python -m farsi_book_ocr.normalize_text \
  sample-output-new/book_ocr.txt \
  sample-output-new/book_normalized.txt

echo ""
echo "=== Step 3/3: LLM Correction ==="
echo "Started at $(date)"
python -m farsi_book_ocr.correct_text \
  sample-output-new/book_normalized.txt \
  --output sample-output-new/book_corrected.txt

echo ""
echo "=== Done ==="
echo "Finished at $(date)"
echo ""
echo "Output files:"
ls -lh sample-output-new/
