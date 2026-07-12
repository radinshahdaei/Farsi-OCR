"""Characterization tests for farsi_book_ocr.normalize_text."""

from farsi_book_ocr.normalize_text import normalize


class TestReplacements:
    """Each replacement in the REPLACEMENTS dict should work."""

    def test_arabic_yeh_to_persian_yeh(self):
        assert normalize("ي") == "ی\n"  # ي → ی

    def test_alef_maksura_to_persian_yeh(self):
        assert normalize("ى") == "ی\n"  # ى → ی

    def test_arabic_kaf_to_persian_kaf(self):
        assert normalize("ك") == "ک\n"  # ك → ک

    def test_tatweel_removed(self):
        assert normalize("ـ") == "\n"  # ـ removed

    def test_rtl_mark_removed(self):
        assert normalize("‏") == "\n"  # RTL mark

    def test_ltr_mark_removed(self):
        assert normalize("‎") == "\n"  # LTR mark

    def test_all_replacements_applied_in_one_pass(self):
        text = "كتابى ـ‏‎"
        result = normalize(text)
        assert "ك" not in result  # Arabic kaf gone
        assert "ى" not in result  # alef maksura gone
        assert "ـ" not in result  # tatweel gone
        assert "‏" not in result  # RTL mark gone
        assert "‎" not in result  # LTR mark gone
        assert "ک" in result  # Persian kaf present
        assert "ی" in result  # Persian yeh present


class TestWhitespace:
    """Whitespace normalization rules."""

    def test_carriage_return_normalized(self):
        assert normalize("line1\r\nline2") == "line1\nline2\n"

    def test_bare_cr_normalized(self):
        assert normalize("line1\rline2") == "line1\nline2\n"

    def test_multiple_spaces_collapsed(self):
        assert normalize("word1     word2") == "word1 word2\n"

    def test_tabs_collapsed_with_spaces(self):
        assert normalize("word1  \t  word2") == "word1 word2\n"

    def test_trailing_newline_added(self):
        assert normalize("text") == "text\n"

    def test_stripped(self):
        assert normalize("  text  ") == "text\n"

    def test_extra_blank_lines_reduced(self):
        assert normalize("a\n\n\n\n\nb") == "a\n\n\nb\n"


class TestRealText:
    """Normalization of short Persian phrases."""

    def test_persian_sentence(self):
        text = "سلام دنيا"  # Arabic yeh
        result = normalize(text)
        assert "ی" in result  # Persian yeh

    def test_mixed_arabic_persian_letters(self):
        text = "كتاب من روى ميز است"
        result = normalize(text)
        assert "کتاب" in result
        assert "روی" in result

    def test_bidi_marks_in_persian(self):
        text = "‏سلام‎"
        result = normalize(text)
        assert result.strip() == "سلام"


class TestEdgeCases:
    """Edge case handling."""

    def test_empty_string(self):
        assert normalize("") == "\n"

    def test_only_whitespace(self):
        assert normalize("   \n\n  ") == "\n"

    def test_only_replacement_chars(self):
        result = normalize("يكـ")
        assert result == "یک\n"

    def test_no_changes_needed(self):
        result = normalize("already fine text")
        assert result == "already fine text\n"
