"""Small Persian text normalization helper for OCR output.

This does not try to be linguistically perfect. It applies safe Unicode-level
normalizations that commonly help Persian OCR text.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

REPLACEMENTS = {
    "ي": "ی",  # Arabic yeh -> Persian yeh
    "ى": "ی",  # Alef maksura -> Persian yeh
    "ك": "ک",  # Arabic kaf -> Persian kaf
    "ؤ": "ؤ",
    "ة": "ه",
    "ۀ": "هٔ",
    "ـ": "",   # tatweel/kashida
    "\u200f": "",  # RTL mark
    "\u200e": "",  # LTR mark
}

MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
MULTI_BLANK_RE = re.compile(r"\n{4,}")


def normalize(text: str) -> str:
    for src, dst in REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_BLANK_RE.sub("\n\n\n", text)
    return text.strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize Persian OCR text.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8", errors="replace")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(normalize(text), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
