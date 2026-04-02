"""Unit tests for geoparser dictionary lookup."""

import pytest

from processing.geoparser import _dictionary_lookup


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
