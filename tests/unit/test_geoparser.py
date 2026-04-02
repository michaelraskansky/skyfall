"""Unit tests for geoparser dictionary lookup and preposition extraction."""

import pytest

from processing.geoparser import _dictionary_lookup, _extract_location_candidate


class TestDictionaryLookup:
    """Tests for _dictionary_lookup(text)."""

    def test_arabic_gaza(self):
        result = _dictionary_lookup("أخبار من غزة اليوم")
        assert result is not None
        assert result == pytest.approx((31.5, 34.47), abs=0.01)

    def test_english_gaza(self):
        result = _dictionary_lookup("News from Gaza today")
        assert result is not None
        assert result == pytest.approx((31.5, 34.47), abs=0.01)

    def test_arabic_tehran(self):
        result = _dictionary_lookup("تقارير من طهران")
        assert result is not None
        assert result == pytest.approx((35.69, 51.39), abs=0.01)

    def test_arabic_damascus(self):
        result = _dictionary_lookup("أحداث في دمشق")
        assert result is not None
        assert result == pytest.approx((33.51, 36.29), abs=0.01)

    def test_arabic_beirut(self):
        result = _dictionary_lookup("انفجار في بيروت")
        assert result is not None
        assert result == pytest.approx((33.89, 35.50), abs=0.01)

    def test_english_case_insensitive(self):
        result = _dictionary_lookup("TEHRAN reports suggest")
        assert result is not None
        assert result == pytest.approx((35.69, 51.39), abs=0.01)

    def test_no_location_text(self):
        result = _dictionary_lookup("No location mentioned here at all")
        assert result is None

    def test_unknown_location(self):
        result = _dictionary_lookup("News from Atlantis today")
        assert result is None

    def test_empty_string(self):
        result = _dictionary_lookup("")
        assert result is None

    def test_multiple_locations_returns_first(self):
        # Gaza appears before Damascus in the text
        result = _dictionary_lookup("غزة ودمشق")
        assert result is not None
        assert result == pytest.approx((31.5, 34.47), abs=0.01)


class TestPrepositionExtraction:
    """Tests for _extract_location_candidate(text)."""

    def test_fi_in(self):
        result = _extract_location_candidate("انفجار في حي الشجاعية")
        assert result == "حي الشجاعية"

    def test_ala_on(self):
        result = _extract_location_candidate("قصف على المنطقة الجنوبية")
        assert result == "المنطقة الجنوبية"

    def test_min_from(self):
        result = _extract_location_candidate("صواريخ من الضفة الغربية")
        assert result == "الضفة الغربية"

    def test_no_preposition(self):
        result = _extract_location_candidate("هذا خبر بدون موقع")
        assert result is None

    def test_max_three_words(self):
        result = _extract_location_candidate("انفجار في كلمة واحدة اثنتان ثلاث أربع")
        words = result.split() if result else []
        assert len(words) <= 3
