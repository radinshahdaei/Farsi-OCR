"""Tests for farsi_book_ocr.cache — content-addressed caching."""

from pathlib import Path

from farsi_book_ocr.cache import (
    RunManifest,
    build_work_layout,
    compute_config_fingerprint,
    compute_file_fingerprint,
    compute_text_fingerprint,
    read_manifest,
    validate_cache,
    write_manifest,
)


class TestFingerprints:
    def test_file_fingerprint_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        a = compute_file_fingerprint(f)
        b = compute_file_fingerprint(f)
        assert a == b

    def test_file_fingerprint_different_content(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert compute_file_fingerprint(f1) != compute_file_fingerprint(f2)

    def test_text_fingerprint_deterministic(self):
        a = compute_text_fingerprint("hello")
        b = compute_text_fingerprint("hello")
        assert a == b

    def test_text_fingerprint_different(self):
        assert compute_text_fingerprint("hello") != compute_text_fingerprint("world")

    def test_config_fingerprint_deterministic(self):
        c1 = {"lang": "fas", "deskew": True, "foo": None}
        c2 = {"foo": None, "deskew": True, "lang": "fas"}  # different key order
        assert compute_config_fingerprint(c1) == compute_config_fingerprint(c2)

    def test_config_fingerprint_different(self):
        c1 = {"lang": "fas"}
        c2 = {"lang": "fas+eng"}
        assert compute_config_fingerprint(c1) != compute_config_fingerprint(c2)

    def test_is_sha256_hex(self):
        fp = compute_text_fingerprint("test")
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)


class TestWorkLayout:
    def test_source_only(self):
        path = build_work_layout(Path("work"), "a" * 64)
        assert path == Path("work") / ("a" * 16)

    def test_with_ocr_config(self):
        path = build_work_layout(Path("work"), "a" * 64, ocr_config_fingerprint="b" * 64)
        assert path == Path("work") / ("a" * 16) / ("b" * 16)

    def test_with_correction_config(self):
        path = build_work_layout(
            Path("work"), "a" * 64,
            ocr_config_fingerprint="b" * 64,
            correction_config_fingerprint="c" * 64,
        )
        assert path == Path("work") / ("a" * 16) / ("b" * 16) / "corrections" / ("c" * 16)

    def test_different_sources_different_dirs(self):
        p1 = build_work_layout(Path("work"), "a" * 64)
        p2 = build_work_layout(Path("work"), "b" * 64)
        assert p1 != p2

    def test_different_ocr_configs_different_dirs(self):
        p1 = build_work_layout(Path("work"), "a" * 64, ocr_config_fingerprint="b" * 64)
        p2 = build_work_layout(Path("work"), "a" * 64, ocr_config_fingerprint="c" * 64)
        assert p1 != p2


class TestManifest:
    def test_write_and_read_roundtrip(self, tmp_path):
        manifest = RunManifest(
            source_fingerprint="s" * 64,
            source_path="/path/to/book.pdf",
            source_page_count=100,
            selected_page_range=(1, 100),
            ocr_config_fingerprint="o" * 64,
            ocr_config={"lang": "fas"},
            tool_versions={"python": "3.14"},
            status="completed",
        )
        write_manifest(manifest, tmp_path)
        read = read_manifest(tmp_path)
        assert read is not None
        assert read.source_fingerprint == "s" * 64
        assert read.source_page_count == 100
        assert read.selected_page_range == (1, 100)
        assert read.ocr_config == {"lang": "fas"}
        assert read.status == "completed"

    def test_atomic_write(self, tmp_path):
        manifest = RunManifest(
            source_fingerprint="s" * 64,
            source_path="/p.pdf",
            source_page_count=1,
            selected_page_range=(1, 1),
        )
        write_manifest(manifest, tmp_path)
        # No .tmp files left
        tmp_files = list(tmp_path.glob("*.tmp*"))
        assert len(tmp_files) == 0

    def test_read_missing(self, tmp_path):
        assert read_manifest(tmp_path) is None

    def test_read_corrupted(self, tmp_path):
        (tmp_path / "manifest.json").write_text("not valid json {{{")
        assert read_manifest(tmp_path) is None

    def test_manifest_includes_artifact_hashes(self, tmp_path):
        manifest = RunManifest(
            source_fingerprint="s" * 64,
            source_path="/p.pdf",
            source_page_count=1,
            selected_page_range=(1, 1),
            artifact_hashes={"output.txt": "h" * 64},
        )
        write_manifest(manifest, tmp_path)
        read = read_manifest(tmp_path)
        assert read is not None
        assert read.artifact_hashes == {"output.txt": "h" * 64}

    def test_manifest_includes_page_statuses(self, tmp_path):
        manifest = RunManifest(
            source_fingerprint="s" * 64,
            source_path="/p.pdf",
            source_page_count=3,
            selected_page_range=(1, 3),
            page_statuses={
                "page-000001": "ocr_ok",
                "page-000002": "ocr_empty",
                "page-000003": "ocr_ok",
            },
        )
        write_manifest(manifest, tmp_path)
        read = read_manifest(tmp_path)
        assert read is not None
        assert read.page_statuses["page-000002"] == "ocr_empty"


class TestCacheValidation:
    def test_matching_fingerprints_valid(self, tmp_path):
        manifest = RunManifest(
            source_fingerprint="s" * 64,
            source_path="/p.pdf",
            source_page_count=1,
            selected_page_range=(1, 1),
            ocr_config_fingerprint="o" * 64,
        )
        write_manifest(manifest, tmp_path)
        assert validate_cache(tmp_path, "s" * 64, "o" * 64)

    def test_mismatched_source_invalid(self, tmp_path):
        manifest = RunManifest(
            source_fingerprint="s" * 64,
            source_path="/p.pdf",
            source_page_count=1,
            selected_page_range=(1, 1),
            ocr_config_fingerprint="o" * 64,
        )
        write_manifest(manifest, tmp_path)
        assert not validate_cache(tmp_path, "x" * 64, "o" * 64)

    def test_mismatched_config_invalid(self, tmp_path):
        manifest = RunManifest(
            source_fingerprint="s" * 64,
            source_path="/p.pdf",
            source_page_count=1,
            selected_page_range=(1, 1),
            ocr_config_fingerprint="o" * 64,
        )
        write_manifest(manifest, tmp_path)
        assert not validate_cache(tmp_path, "s" * 64, "x" * 64)

    def test_missing_manifest_invalid(self, tmp_path):
        assert not validate_cache(tmp_path, "s" * 64, "o" * 64)
