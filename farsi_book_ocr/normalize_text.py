"""Persian text normalization for OCR output.

Applies safe Unicode-level normalizations that commonly help Persian OCR text.
Now supports staged configuration — each transformation can be toggled.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Character-level replacements
# ---------------------------------------------------------------------------

# Arabic-to-Persian letter mappings
ARABIC_TO_PERSIAN = {
    "ي": "ی",  # Arabic yeh → Persian yeh
    "ى": "ی",  # Alef maksura → Persian yeh
    "ك": "ک",  # Arabic kaf → Persian kaf
}

# Additional Persian-specific normalizations (may over-correct Arabic text)
PERSIAN_NORMALIZATIONS = {
    "ؤ": "ؤ",  # Waw with hamza above → Waw + standalone hamza
    "ة": "ه",   # Teh marbuta → Heh
    "ۀ": "هٔ",  # Heh yeh → Heh + standalone hamza
}

# Invisible / control characters to strip
STRIP_CHARACTERS = {
    "ـ": "",       # Tatweel / kashida
    "‏": "",  # RTL mark
    "‎": "",  # LTR mark
}

# Compiled regexes
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_BLANK_RE = re.compile(r"\n{4,}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NormalizationConfig:
    """Configuration for the normalization pipeline.

    All options default to True for backward compatibility with the
    original normalize() function. Disable specific transformations
    to preserve legitimate Arabic text, table layout, etc.
    """

    apply_nfc: bool = True
    """Apply Unicode NFC normalization (compose combining characters)."""

    arabic_to_persian: bool = False
    """Convert Arabic letter forms to Persian: ي→ی, ك→ک, ى→ی.
    Off by default — enable only for pure Persian texts (no Arabic quotations)."""

    persian_normalizations: bool = False
    """Apply Persian-specific fixes: ؤ→ؤ, ة→ه, ۀ→هٔ.
    Off by default — WARNING: may corrupt legitimate Arabic text. Only enable
    for pure Persian texts (no Quranic verses, Arabic quotations)."""

    remove_kashida: bool = True
    """Strip tatweel/kashida (ـ) characters."""

    remove_bidi_marks: bool = True
    """Strip RTL/LTR mark characters (U+200F, U+200E)."""

    normalize_whitespace: bool = True
    """Collapse multiple spaces/tabs and reduce excessive blank lines."""

    normalize_line_endings: bool = True
    """Convert \\r\\n and \\r to \\n."""

    strip_trailing: bool = True
    """Strip leading/trailing whitespace and ensure trailing newline."""


# Default configuration (matches original behavior)
DEFAULT_CONFIG = NormalizationConfig()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize(text: str, config: NormalizationConfig | None = None) -> str:
    """Normalize Persian OCR text.

    Args:
        text: Raw OCR text to normalize.
        config: Optional configuration. If None, uses DEFAULT_CONFIG
                which matches the original behavior.

    Returns:
        Normalized text.
    """
    if config is None:
        config = DEFAULT_CONFIG

    # 1. Unicode NFC
    if config.apply_nfc:
        text = unicodedata.normalize("NFC", text)

    # 2. Line endings
    if config.normalize_line_endings:
        text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Arabic-to-Persian letter mapping
    if config.arabic_to_persian:
        for src, dst in ARABIC_TO_PERSIAN.items():
            text = text.replace(src, dst)

    # 4. Persian-specific normalizations
    if config.persian_normalizations:
        for src, dst in PERSIAN_NORMALIZATIONS.items():
            text = text.replace(src, dst)

    # 5. Strip kashida
    if config.remove_kashida:
        # Kashida is in STRIP_CHARACTERS
        text = text.replace("ـ", "")

    # 6. Strip bidi marks
    if config.remove_bidi_marks:
        text = text.replace("‏", "").replace("‎", "")

    # 7. Whitespace
    if config.normalize_whitespace:
        text = _MULTI_SPACE_RE.sub(" ", text)
        text = _MULTI_BLANK_RE.sub("\n\n\n", text)

    # 8. Final cleanup
    if config.strip_trailing:
        text = text.strip() + "\n"

    return text


def normalize_preserve_layout(text: str) -> str:
    """Normalize text while preserving layout (table spacing, line breaks).

    Useful for pages with tables, contents pages with dotted leaders, etc.
    """
    config = NormalizationConfig(
        apply_nfc=True,
        arabic_to_persian=True,
        persian_normalizations=False,  # Preserve Arabic forms
        remove_kashida=True,
        remove_bidi_marks=True,
        normalize_whitespace=False,  # Keep layout
        normalize_line_endings=True,
        strip_trailing=False,  # Keep exact whitespace
    )
    return normalize(text, config)


def normalize_arabic_safe(text: str) -> str:
    """Normalize text while preserving Arabic orthography.

    Use for texts that contain Quranic verses, Arabic quotations, or
    Arabic poetry alongside Persian text.

    Note: This is the same as the default normalize() behavior.
    Kept as a self-documenting alias for clarity.
    """
    config = NormalizationConfig(
        apply_nfc=True,
        arabic_to_persian=False,  # Preserve Arabic letters
        persian_normalizations=False,
        remove_kashida=True,
        remove_bidi_marks=True,
        normalize_whitespace=True,
        normalize_line_endings=True,
        strip_trailing=True,
    )
    return normalize(text, config)


def normalize_persian(text: str) -> str:
    """Aggressive normalization for pure Persian texts.

    Converts Arabic letter forms to Persian equivalents (ي→ی, ك→ک, etc.)
    and applies Persian-specific character fixes. Use only when you are
    certain the text contains no Arabic content (Quranic verses, quotes, etc.).

    For mixed Persian/Arabic texts, use the default normalize() instead.
    """
    config = NormalizationConfig(
        apply_nfc=True,
        arabic_to_persian=True,
        persian_normalizations=True,
        remove_kashida=True,
        remove_bidi_marks=True,
        normalize_whitespace=True,
        normalize_line_endings=True,
        strip_trailing=True,
    )
    return normalize(text, config)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize Persian OCR text.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--persian", action="store_true",
        help="Aggressive: convert Arabic→Persian letters (only for pure Persian texts)",
    )
    parser.add_argument(
        "--preserve-layout", action="store_true",
        help="Preserve table spacing and line breaks",
    )
    parser.add_argument(
        "--arabic-safe", action="store_true",
        help="Preserve Arabic letter forms (this is now the default; kept for compatibility)",
    )
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8", errors="replace")

    if args.persian:
        result = normalize_persian(text)
    elif args.preserve_layout:
        result = normalize_preserve_layout(text)
    elif args.arabic_safe:
        result = normalize_arabic_safe(text)
    else:
        result = normalize(text)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
