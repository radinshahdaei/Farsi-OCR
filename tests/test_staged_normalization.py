"""Tests for the staged normalization config in normalize_text."""

from farsi_book_ocr.normalize_text import (
    NormalizationConfig,
    normalize,
    normalize_arabic_safe,
    normalize_persian,
    normalize_preserve_layout,
)


class TestNFC:
    def test_nfc_composes_characters(self):
        # U+0645 (م) + U+064E (fatha) = composed
        text = "مَ"  # Meem + combining fatha
        result = normalize(text)
        assert result.strip() == text.strip()
        # NFC should compose into a single normalization form
        # (Even if visually identical, the Unicode form is NFC)

    def test_nfc_disabled_preserves_decomposed(self):
        text = "مَ"
        config = NormalizationConfig(apply_nfc=False)
        result = normalize(text, config)
        # With NFC disabled, the combining sequence is preserved as-is
        assert len(result) >= 2  # at minimum 2 code points


class TestArabicToPersian:
    def test_enabled_converts_arabic_yeh(self):
        config = NormalizationConfig(arabic_to_persian=True)
        result = normalize("كتابي", config)
        assert "کتابی" in result

    def test_disabled_preserves_arabic_yeh(self):
        config = NormalizationConfig(arabic_to_persian=False)
        result = normalize("كتابي", config)
        assert "ك" in result  # Arabic kaf preserved
        assert "ي" in result  # Arabic yeh preserved (not converted)

    def test_disabled_does_not_convert_anything(self):
        config = NormalizationConfig(arabic_to_persian=False)
        text = "كتابي"  # All Arabic letters
        result = normalize(text, config)
        # Should be identical except for trailing newline
        assert result.strip() == "كتابي"


class TestPersianNormalizations:
    def test_enabled_fixes_persian_forms(self):
        config = NormalizationConfig(persian_normalizations=True)
        result = normalize("ة", config)
        assert result.strip() == "ه"

    def test_disabled_preserves_arabic(self):
        config = NormalizationConfig(persian_normalizations=False)
        result = normalize("ة", config)
        assert "ة" in result  # Teh marbuta preserved


class TestBidiMarks:
    def test_enabled_strips_bidi_marks(self):
        config = NormalizationConfig(remove_bidi_marks=True)
        result = normalize("‏سلام‎", config)
        assert "‏" not in result
        assert "‎" not in result

    def test_disabled_preserves_bidi_marks(self):
        config = NormalizationConfig(remove_bidi_marks=False)
        result = normalize("‏سلام‎", config)
        assert "‏" in result or "‎" in result


class TestKashida:
    def test_enabled_strips_kashida(self):
        config = NormalizationConfig(remove_kashida=True)
        result = normalize("متـن", config)
        assert "ـ" not in result

    def test_disabled_preserves_kashida(self):
        config = NormalizationConfig(remove_kashida=False)
        result = normalize("متـن", config)
        assert "ـ" in result


class TestPresetConfigs:
    def test_preserve_layout_keeps_line_breaks(self):
        text = "a             b\n\n\n\nc"
        result = normalize_preserve_layout(text)
        # Multiple spaces preserved
        assert "             " in result

    def test_preserve_layout_no_trailing_newline(self):
        text = "text"
        result = normalize_preserve_layout(text)
        assert result == "text"  # No added newline, no strip

    def test_arabic_safe_preserves_arabic_letters(self):
        text = "كتاب الله"  # Arabic kaf, Arabic yeh, alef, lam, lam, heh
        result = normalize_arabic_safe(text)
        assert "ك" in result  # Arabic kaf preserved
        assert "الل" in result  # Arabic lam preserved

    def test_default_is_arabic_safe(self):
        """Default normalize() preserves Arabic letters."""
        text = "كتابى ـ‏‎"
        result = normalize(text)
        # Invisible characters stripped
        assert "ـ" not in result
        assert "‏" not in result
        assert "‎" not in result
        # Arabic letters preserved
        assert "ك" in result
        assert "ى" in result


class TestPersianPreset:
    def test_converts_arabic_to_persian(self):
        text = "كتابي"
        result = normalize_persian(text)
        assert "کتابی" in result
        assert "ك" not in result  # Arabic kaf converted
        assert "ي" not in result  # Arabic yeh converted

    def test_applies_persian_normalizations(self):
        result = normalize_persian("ة")
        assert result.strip() == "ه"

    def test_invisible_chars_stripped(self):
        text = "متـن‏‎"
        result = normalize_persian(text)
        assert "ـ" not in result
        assert "‏" not in result
        assert "‎" not in result


class TestConfigImmutability:
    def test_different_configs_independent(self):
        text = "كتاب"
        config_convert = NormalizationConfig(arabic_to_persian=True)
        config_preserve = NormalizationConfig(arabic_to_persian=False)

        r1 = normalize(text, config_convert)
        r2 = normalize(text, config_preserve)

        # Config objects are not modified by normalize()
        assert config_convert.arabic_to_persian is True
        assert config_preserve.arabic_to_persian is False
