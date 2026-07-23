"""Characterization tests for farsi_book_ocr.normalize_text."""

from farsi_book_ocr.normalize_text import normalize


class TestReplacements:
    """Each replacement in the REPLACEMENTS dict should work."""

    def test_arabic_yeh_preserved(self):
        assert "ي" in normalize("ي")  # Arabic yeh preserved by default

    def test_alef_maksura_preserved(self):
        assert "ى" in normalize("ى")  # Alef maksura preserved by default

    def test_arabic_kaf_preserved(self):
        assert "ك" in normalize("ك")  # Arabic kaf preserved by default

    def test_tatweel_removed(self):
        assert normalize("ـ") == ""  # ـ removed, layout preserved

    def test_rtl_mark_removed(self):
        assert normalize("‏") == ""  # RTL mark removed, layout preserved

    def test_ltr_mark_removed(self):
        assert normalize("‎") == ""  # LTR mark removed, layout preserved

    def test_all_replacements_applied_in_one_pass(self):
        text = "كتابى ـ‏‎"
        result = normalize(text)
        # Invisible characters stripped
        assert "ـ" not in result  # tatweel gone
        assert "‏" not in result  # RTL mark gone
        assert "‎" not in result  # LTR mark gone
        # Arabic letters preserved (not converted to Persian)
        assert "ك" in result  # Arabic kaf preserved
        assert "ى" in result  # Alef maksura preserved


class TestWhitespace:
    """Whitespace normalization rules.

    The default normalize() preserves layout (no whitespace collapsing).
    Use normalize_arabic_safe() for whitespace normalization.
    """

    def test_carriage_return_normalized(self):
        assert normalize("line1\r\nline2") == "line1\nline2"

    def test_bare_cr_normalized(self):
        assert normalize("line1\rline2") == "line1\nline2"

    def test_multiple_spaces_preserved_by_default(self):
        assert normalize("word1     word2") == "word1     word2"

    def test_tabs_preserved_by_default(self):
        assert normalize("word1  \t  word2") == "word1  \t  word2"

    def test_no_trailing_newline_added(self):
        assert normalize("text") == "text"

    def test_not_stripped(self):
        assert normalize("  text  ") == "  text  "

    def test_extra_blank_lines_preserved(self):
        assert normalize("a\n\n\n\n\nb") == "a\n\n\n\n\nb"


class TestRealText:
    """Normalization of short Persian phrases."""

    def test_persian_sentence(self):
        text = "سلام دنيا"  # Arabic yeh
        result = normalize(text)
        assert "ي" in result  # Arabic yeh preserved by default

    def test_mixed_arabic_persian_letters(self):
        text = "كتاب من روى ميز است"
        result = normalize(text)
        assert "ك" in result  # Arabic kaf preserved
        assert "ي" in result  # Arabic yeh preserved

    def test_bidi_marks_in_persian(self):
        text = "‏سلام‎"
        result = normalize(text)
        assert result == "سلام"


class TestEdgeCases:
    """Edge case handling."""

    def test_empty_string(self):
        assert normalize("") == ""

    def test_only_whitespace(self):
        assert normalize("   \n\n  ") == "   \n\n  "

    def test_only_replacement_chars(self):
        result = normalize("يكـ")
        assert "ي" in result  # Arabic yeh preserved
        assert "ك" in result  # Arabic kaf preserved
        assert "ـ" not in result  # Kashida stripped

    def test_no_changes_needed(self):
        result = normalize("already fine text")
        assert result == "already fine text"
