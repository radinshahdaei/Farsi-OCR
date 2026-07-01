"""Chunked OCR pipeline for long Farsi/Persian scanned books.

Example:
    python -m farsi_book_ocr.ocr_book input/book.pdf --lang fas+eng --jobs 4 --pages-per-chunk 25
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass(frozen=True)
class Chunk:
    index: int
    start_page_zero_based: int
    end_page_zero_based_exclusive: int

    @property
    def human_start(self) -> int:
        return self.start_page_zero_based + 1

    @property
    def human_end(self) -> int:
        return self.end_page_zero_based_exclusive

    @property
    def label(self) -> str:
        return f"chunk_{self.index:04d}_p{self.human_start:04d}-{self.human_end:04d}"


def run(cmd: list[str], *, log_file: Path | None = None) -> None:
    """Run a subprocess and stream output to the terminal and optional log."""
    printable = " ".join(cmd)
    print(f"\n$ {printable}\n", flush=True)

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        if proc.stdout is not None:
            if log_file is not None:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with log_file.open("a", encoding="utf-8") as log:
                    log.write(f"\n$ {printable}\n")
                    for line in proc.stdout:
                        print(line, end="")
                        log.write(line)
            else:
                for line in proc.stdout:
                    print(line, end="")
        code = proc.wait()

    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(
            f"Required executable not found: {name}\n"
            f"Install it first. On macOS, run: brew install ocrmypdf tesseract-lang"
        )


def check_tesseract_language(lang: str) -> None:
    """Warn early if requested Tesseract languages are unavailable."""
    require_executable("tesseract")
    result = subprocess.run(
        ["tesseract", "--list-langs"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    available = set(result.stdout.split())
    requested = set(lang.split("+"))
    missing = sorted(requested - available)
    if missing:
        raise SystemExit(
            "Missing Tesseract language data: " + ", ".join(missing) + "\n"
            "Install language data with:\n"
            "  brew install tesseract-lang\n"
            "Then verify with:\n"
            "  tesseract --list-langs | grep fas"
        )


def build_chunks(total_pages: int, pages_per_chunk: int, first_page: int, last_page: int | None) -> list[Chunk]:
    if first_page < 1:
        raise SystemExit("--first-page must be 1 or greater")
    if last_page is None:
        last_page = total_pages
    if last_page < first_page:
        raise SystemExit("--last-page must be greater than or equal to --first-page")
    if last_page > total_pages:
        raise SystemExit(f"--last-page {last_page} is beyond PDF page count {total_pages}")

    start = first_page - 1
    stop = last_page
    chunks: list[Chunk] = []
    idx = 1
    for chunk_start in range(start, stop, pages_per_chunk):
        chunk_stop = min(chunk_start + pages_per_chunk, stop)
        chunks.append(Chunk(idx, chunk_start, chunk_stop))
        idx += 1
    return chunks


def split_chunk(src_pdf: Path, chunk: Chunk, chunk_pdf: Path) -> None:
    if chunk_pdf.exists():
        return
    chunk_pdf.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(src_pdf) as src, fitz.open() as out:
        out.insert_pdf(
            src,
            from_page=chunk.start_page_zero_based,
            to_page=chunk.end_page_zero_based_exclusive - 1,
        )
        out.save(chunk_pdf, deflate=True, garbage=3)


def ocr_chunk(
    chunk_pdf: Path,
    ocr_pdf: Path,
    sidecar_txt: Path,
    *,
    lang: str,
    jobs: int,
    deskew: bool,
    rotate_pages: bool,
    clean_final: bool,
    force_ocr: bool,
    output_type: str,
    tesseract_timeout: int,
    log_file: Path,
) -> None:
    if ocr_pdf.exists() and sidecar_txt.exists() and ocr_pdf.stat().st_size > 0:
        print(f"Skipping completed chunk: {ocr_pdf.name}")
        return

    ocr_pdf.parent.mkdir(parents=True, exist_ok=True)
    sidecar_txt.parent.mkdir(parents=True, exist_ok=True)

    tmp_pdf = ocr_pdf.with_suffix(".tmp.pdf")
    tmp_txt = sidecar_txt.with_suffix(".tmp.txt")
    for p in (tmp_pdf, tmp_txt):
        if p.exists():
            p.unlink()

    cmd = [
        "ocrmypdf",
        "-l",
        lang,
        "--jobs",
        str(jobs),
        "--output-type",
        output_type,
        "--optimize",
        "1",
        "--tesseract-timeout",
        str(tesseract_timeout),
        "--sidecar",
        str(tmp_txt),
    ]

    if deskew:
        cmd.append("--deskew")
    if rotate_pages:
        cmd.append("--rotate-pages")
    if clean_final:
        cmd.append("--clean-final")
    if force_ocr:
        cmd.append("--force-ocr")
    else:
        cmd.append("--skip-text")

    cmd.extend([str(chunk_pdf), str(tmp_pdf)])
    run(cmd, log_file=log_file)

    tmp_pdf.replace(ocr_pdf)
    tmp_txt.replace(sidecar_txt)


def merge_pdfs(ocr_pdfs: list[Path], out_pdf: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_pdf.with_suffix(".tmp.pdf")
    if tmp.exists():
        tmp.unlink()
    with fitz.open() as merged:
        for pdf in ocr_pdfs:
            with fitz.open(pdf) as part:
                merged.insert_pdf(part)
        merged.save(tmp, deflate=True, garbage=3)
    tmp.replace(out_pdf)


def merge_text(sidecars: list[Path], out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_txt.with_suffix(".tmp.txt")
    with tmp.open("w", encoding="utf-8") as w:
        for i, txt in enumerate(sidecars, start=1):
            w.write(f"\n\n===== OCR CHUNK {i:04d}: {txt.stem} =====\n\n")
            w.write(txt.read_text(encoding="utf-8", errors="replace"))
            w.write("\n")
    tmp.replace(out_txt)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR a long Persian/Farsi scanned book PDF in resumable chunks.")
    parser.add_argument("pdf", type=Path, help="Input PDF path")
    parser.add_argument("--lang", default="fas", help="Tesseract language string, e.g. fas or fas+eng")
    parser.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 4) // 2)), help="OCRmyPDF parallel jobs per chunk")
    parser.add_argument("--pages-per-chunk", type=int, default=25, help="Pages per chunk")
    parser.add_argument("--first-page", type=int, default=1, help="First 1-based page to OCR")
    parser.add_argument("--last-page", type=int, default=None, help="Last 1-based page to OCR")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Final output directory")
    parser.add_argument("--work-dir", type=Path, default=Path("work"), help="Intermediate work directory")
    parser.add_argument("--deskew", action="store_true", help="Deskew crooked scans")
    parser.add_argument("--rotate-pages", action="store_true", help="Detect and rotate misoriented pages")
    parser.add_argument("--clean-final", action="store_true", help="Clean page images; can help noisy scans but may hurt fine Persian marks")
    parser.add_argument("--force-ocr", action="store_true", help="OCR even if an existing text layer is detected")
    parser.add_argument("--output-type", default="pdf", choices=["pdf", "pdfa", "pdfa-1", "pdfa-2", "pdfa-3"], help="OCRmyPDF output type")
    parser.add_argument("--tesseract-timeout", type=int, default=300, help="Seconds before Tesseract gives up on a page")
    parser.add_argument("--redo", action="store_true", help="Delete existing work for this book and start over")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    require_executable("ocrmypdf")
    check_tesseract_language(args.lang)

    input_pdf = args.pdf.expanduser().resolve()
    if not input_pdf.exists():
        raise SystemExit(f"Input PDF not found: {input_pdf}")

    book_name = input_pdf.stem
    book_work = args.work_dir / book_name
    chunks_dir = book_work / "chunks"
    ocr_dir = book_work / "ocr_chunks"
    text_dir = book_work / "text_chunks"
    log_file = book_work / "logs" / "ocr.log"

    if args.redo and book_work.exists():
        shutil.rmtree(book_work)

    with fitz.open(input_pdf) as doc:
        total_pages = doc.page_count

    chunks = build_chunks(total_pages, args.pages_per_chunk, args.first_page, args.last_page)
    print(f"Input: {input_pdf}")
    print(f"Total pages in PDF: {total_pages}")
    print(f"OCR page range: {chunks[0].human_start}-{chunks[-1].human_end}")
    print(f"Chunks: {len(chunks)} chunk(s) of about {args.pages_per_chunk} pages")
    print(f"Language: {args.lang}")

    expected_ocr_pdfs: list[Path] = []
    expected_sidecars: list[Path] = []

    for chunk in chunks:
        chunk_pdf = chunks_dir / f"{chunk.label}.pdf"
        ocr_pdf = ocr_dir / f"{chunk.label}.ocr.pdf"
        sidecar_txt = text_dir / f"{chunk.label}.txt"
        expected_ocr_pdfs.append(ocr_pdf)
        expected_sidecars.append(sidecar_txt)

        print(f"\n=== {chunk.label}: pages {chunk.human_start}-{chunk.human_end} ===")
        split_chunk(input_pdf, chunk, chunk_pdf)
        ocr_chunk(
            chunk_pdf,
            ocr_pdf,
            sidecar_txt,
            lang=args.lang,
            jobs=args.jobs,
            deskew=args.deskew,
            rotate_pages=args.rotate_pages,
            clean_final=args.clean_final,
            force_ocr=args.force_ocr,
            output_type=args.output_type,
            tesseract_timeout=args.tesseract_timeout,
            log_file=log_file,
        )

    missing = [p for p in expected_ocr_pdfs + expected_sidecars if not p.exists()]
    if missing:
        raise SystemExit("Some expected output files are missing:\n" + "\n".join(str(p) for p in missing))

    suffix = "_ocr"
    if args.first_page != 1 or args.last_page is not None:
        suffix += f"_p{chunks[0].human_start:04d}-{chunks[-1].human_end:04d}"

    out_pdf = args.output_dir / f"{book_name}{suffix}.pdf"
    out_txt = args.output_dir / f"{book_name}{suffix}.txt"

    print("\nMerging OCR PDF chunks...")
    merge_pdfs(expected_ocr_pdfs, out_pdf)
    print("Merging OCR text chunks...")
    merge_text(expected_sidecars, out_txt)

    print("\nDone.")
    print(f"Searchable PDF: {out_pdf.resolve()}")
    print(f"Plain text:      {out_txt.resolve()}")
    print(f"Work folder:     {book_work.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
