"""Shared fixtures for the Farsi-OCR test suite."""

from pathlib import Path

import pytest


@pytest.fixture
def sample_dir() -> Path:
    """Path to the sample-output directory."""
    return Path(__file__).resolve().parent.parent / "sample-output"


@pytest.fixture
def sample_raw_text(sample_dir: Path) -> str:
    """Contents of the raw OCR sample file."""
    path = sample_dir / "book_ocr.txt"
    if not path.exists():
        pytest.skip("sample-output/book_ocr.txt not available")
    return path.read_text(encoding="utf-8", errors="replace")


@pytest.fixture
def sample_corrected_text(sample_dir: Path) -> str:
    """Contents of the LLM-corrected sample file."""
    path = sample_dir / "book_ocr.corrected.txt"
    if not path.exists():
        pytest.skip("sample-output/book_ocr.corrected.txt not available")
    return path.read_text(encoding="utf-8", errors="replace")
