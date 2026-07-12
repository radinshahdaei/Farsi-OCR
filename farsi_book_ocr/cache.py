"""Content-addressed caching and manifest management.

Replaces filename-stem-based work directories with hash-based layouts.
Cache validity is determined by matching SHA-256 fingerprints of source
content and configuration — not by filename.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------

# Sentinel for unset values (None is a valid value in config dicts)
_UNSET = object()


def _stable_json_dumps(obj: Any) -> str:
    """Serialize to a canonical JSON string with sorted keys."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def compute_file_fingerprint(path: Path) -> str:
    """SHA-256 of file content (full file)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_pdf_fingerprint(path: Path) -> str:
    """Fast fingerprint for PDF: first 64KB + last 64KB + file size."""
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(65536))
        if size > 131072:
            f.seek(-65536, 2)
            h.update(f.read(65536))
    return h.hexdigest()


def compute_text_fingerprint(text: str) -> str:
    """SHA-256 of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_config_fingerprint(config: dict[str, Any]) -> str:
    """SHA-256 of a canonical JSON representation of the config dict."""
    return hashlib.sha256(_stable_json_dumps(config).encode()).hexdigest()


# ---------------------------------------------------------------------------
# OCR configuration fingerprint
# ---------------------------------------------------------------------------


def build_ocr_config_dict(
    lang: str,
    deskew: bool,
    rotate_pages: bool,
    clean_final: bool,
    force_ocr: bool,
    output_type: str,
    pages_per_chunk: int,
    first_page: int,
    last_page: int | None,
) -> dict[str, Any]:
    """Build a stable config dict for OCR fingerprinting."""
    return {
        "lang": lang,
        "deskew": deskew,
        "rotate_pages": rotate_pages,
        "clean_final": clean_final,
        "force_ocr": force_ocr,
        "output_type": output_type,
        "pages_per_chunk": pages_per_chunk,
        "first_page": first_page,
        "last_page": last_page,
    }


# ---------------------------------------------------------------------------
# Tool version detection
# ---------------------------------------------------------------------------


def _get_ocrmypdf_version() -> str | None:
    try:
        r = subprocess.run(
            ["ocrmypdf", "--version"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False,
        )
        return r.stdout.strip().split("\n")[0] if r.stdout else None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _get_tesseract_version() -> str | None:
    try:
        r = subprocess.run(
            ["tesseract", "--version"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False,
        )
        return r.stdout.strip().split("\n")[0] if r.stdout else None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _get_pymupdf_version() -> str:
    return fitz.version[0]


def get_tool_versions() -> dict[str, str | None]:
    """Collect versions of installed tools for the manifest."""
    return {
        "python": sys.version.split()[0],
        "ocrmypdf": _get_ocrmypdf_version(),
        "tesseract": _get_tesseract_version(),
        "PyMuPDF": _get_pymupdf_version(),
    }


# ---------------------------------------------------------------------------
# Work directory layout
# ---------------------------------------------------------------------------


def build_work_layout(
    base_work_dir: Path,
    source_fingerprint: str,
    ocr_config_fingerprint: str | None = None,
    correction_config_fingerprint: str | None = None,
) -> Path:
    """Build a hash-based work directory path.

    Layout:
        <base>/<source[:16]>/<ocr[:16]>/corrections/<corr[:16]>/

    Args:
        base_work_dir: Root work directory (e.g., Path("work")).
        source_fingerprint: SHA-256 of the source PDF.
        ocr_config_fingerprint: SHA-256 of OCR configuration.
        correction_config_fingerprint: SHA-256 of correction configuration.

    Returns:
        Path to the appropriate work subdirectory.
    """
    parts = [source_fingerprint[:16]]
    if ocr_config_fingerprint:
        parts.append(ocr_config_fingerprint[:16])
    if correction_config_fingerprint:
        parts.append("corrections")
        parts.append(correction_config_fingerprint[:16])
    return base_work_dir.joinpath(*parts)


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------


@dataclass
class RunManifest:
    """Complete record of an OCR or correction run."""

    source_fingerprint: str
    source_path: str  # display only
    source_page_count: int
    selected_page_range: tuple[int, int]

    ocr_config_fingerprint: str | None = None
    ocr_config: dict[str, Any] | None = None

    correction_config_fingerprint: str | None = None
    correction_config: dict[str, Any] | None = None

    tool_versions: dict[str, str | None] = field(default_factory=dict)
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    page_statuses: dict[str, str] = field(default_factory=dict)

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    status: str = "in_progress"
    """'in_progress', 'completed', 'completed_with_fallbacks', 'failed'."""


def write_manifest(manifest: RunManifest, work_dir: Path) -> None:
    """Atomically write a run manifest to the work directory."""
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "manifest.json"
    tmp = path.with_suffix(".tmp.json")

    data = {
        "source_fingerprint": manifest.source_fingerprint,
        "source_path": manifest.source_path,
        "source_page_count": manifest.source_page_count,
        "selected_page_range": list(manifest.selected_page_range),
        "ocr_config_fingerprint": manifest.ocr_config_fingerprint,
        "ocr_config": manifest.ocr_config,
        "correction_config_fingerprint": manifest.correction_config_fingerprint,
        "correction_config": manifest.correction_config,
        "tool_versions": manifest.tool_versions,
        "artifact_hashes": manifest.artifact_hashes,
        "page_statuses": manifest.page_statuses,
        "created_at": manifest.created_at,
        "status": manifest.status,
    }

    tmp.write_text(_stable_json_dumps(data), encoding="utf-8")
    tmp.replace(path)


def read_manifest(work_dir: Path) -> RunManifest | None:
    """Read a run manifest from the work directory."""
    path = work_dir / "manifest.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return RunManifest(
            source_fingerprint=data["source_fingerprint"],
            source_path=data["source_path"],
            source_page_count=data["source_page_count"],
            selected_page_range=tuple(data["selected_page_range"]),
            ocr_config_fingerprint=data.get("ocr_config_fingerprint"),
            ocr_config=data.get("ocr_config"),
            correction_config_fingerprint=data.get("correction_config_fingerprint"),
            correction_config=data.get("correction_config"),
            tool_versions=data.get("tool_versions", {}),
            artifact_hashes=data.get("artifact_hashes", {}),
            page_statuses=data.get("page_statuses", {}),
            created_at=data.get("created_at", ""),
            status=data.get("status", "unknown"),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def validate_cache(
    work_dir: Path,
    expected_source_fingerprint: str,
    expected_config_fingerprint: str | None = None,
) -> bool:
    """Check whether a cached work directory is still valid.

    Returns True if the cache matches the expected fingerprints.
    """
    manifest = read_manifest(work_dir)
    if manifest is None:
        return False

    if manifest.source_fingerprint != expected_source_fingerprint:
        return False

    if expected_config_fingerprint is not None:
        if manifest.ocr_config_fingerprint != expected_config_fingerprint:
            return False

    return True
