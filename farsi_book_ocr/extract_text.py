"""Extract text from a searchable PDF using PyMuPDF."""
from __future__ import annotations

import argparse
from pathlib import Path

import fitz


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract text from a searchable PDF.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(args.pdf) as doc, args.output.open("w", encoding="utf-8") as out:
        for i, page in enumerate(doc, start=1):
            out.write(f"\n\n===== PAGE {i:04d} =====\n\n")
            out.write(page.get_text("text"))
            out.write("\n")

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
