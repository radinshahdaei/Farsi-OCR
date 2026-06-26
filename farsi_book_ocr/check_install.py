"""Check whether OCRmyPDF and Persian Tesseract language data are installed."""
from __future__ import annotations

import shutil
import subprocess


def main() -> int:
    for exe in ["ocrmypdf", "tesseract"]:
        path = shutil.which(exe)
        print(f"{exe}: {path or 'NOT FOUND'}")

    if shutil.which("tesseract"):
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        langs = set(result.stdout.split())
        print("fas:", "OK" if "fas" in langs else "MISSING")
        print("eng:", "OK" if "eng" in langs else "MISSING")

    if shutil.which("ocrmypdf"):
        result = subprocess.run(
            ["ocrmypdf", "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print("ocrmypdf version:", result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
