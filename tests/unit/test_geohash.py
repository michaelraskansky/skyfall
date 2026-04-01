"""Tests for the geohash encoding and neighbor computation module."""

import pytest

from processing.geohash import encode, neighbors


class TestEncode:
    """Tests for geohash encode()."""

    def test_houston_precision4(self):
        # Houston, TX ~29.76, -95.37 -> "9vk1" at precision 4
        result = encode(29.76, -95.37, precision=4)
        assert result == "9vk1"

    def test_london_precision4(self):
        # London ~51.5074, -0.1278 -> "gcpv" at precision 4
        result = encode(51.5074, -0.1278, precision=4)
        assert result == "gcpv"

    def test_tokyo_precision4(self):
        # Tokyo ~35.6762, 139.6503 -> "xn76" at precision 4
        result = encode(35.6762, 139.6503, precision=4)
        assert result == "xn76"

    def test_houston_precision5(self):
        result = encode(29.76, -95.37, precision=5)
        assert result.startswith("9vk1")
        assert len(result) == 5

    def test_london_precision5(self):
        result = encode(51.5074, -0.1278, precision=5)
        assert result.startswith("gcpv")
        assert len(result) == 5

    def test_tokyo_precision5(self):
        result = encode(35.6762, 139.6503, precision=5)
        assert result.startswith("xn76")
        assert len(result) == 5

    def test_default_precision_is_4(self):
        result = encode(29.76, -95.37)
        assert len(result) == 4

    def test_north_pole(self):
        result = encode(90.0, 0.0, precision=4)
        assert len(result) == 4
        assert isinstance(result, str)

    def test_south_pole(self):
        result = encode(-90.0, 0.0, precision=4)
        assert len(result) == 4
        assert isinstance(result, str)

    def test_dateline_positive(self):
        result = encode(0.0, 180.0, precision=4)
        assert len(result) == 4
        assert isinstance(result, str)

    def test_dateline_negative(self):
        result = encode(0.0, -180.0, precision=4)
        assert len(result) == 4
        assert isinstance(result, str)

    def test_origin(self):
        result = encode(0.0, 0.0, precision=4)
        assert len(result) == 4
        assert isinstance(result, str)

    def test_precision_1(self):
        result = encode(29.76, -95.37, precision=1)
        assert len(result) == 1

    def test_all_chars_in_base32_alphabet(self):
        base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
        result = encode(29.76, -95.37, precision=8)
        for ch in result:
            assert ch in base32


class TestNeighbors:
    """Tests for geohash neighbors()."""

    def test_returns_8_neighbors(self):
        result = neighbors("9vk1")
        assert len(result) == 8

    def test_neighbors_are_unique(self):
        result = neighbors("9vk1")
        assert len(result) == len(set(result))

    def test_does_not_include_self(self):
        gh = "9vk1"
        result = neighbors(gh)
        assert gh not in result

    def test_neighbors_same_precision(self):
        gh = "9vk1"
        result = neighbors(gh)
        for n in result:
            assert len(n) == len(gh)

    def test_neighbors_precision5(self):
        gh = "9vk1b"
        result = neighbors(gh)
        assert len(result) == 8
        for n in result:
            assert len(n) == 5

    def test_nearby_locations_share_prefix_or_are_neighbors(self):
        """Two nearby points should either share a geohash or be neighbors."""
        gh1 = encode(29.76, -95.37, precision=6)
        gh2 = encode(29.76, -95.36, precision=6)
        assert gh1 == gh2 or gh2 in neighbors(gh1)

    def test_neighbors_of_known_geohash(self):
        # Just verify it returns strings from the base32 alphabet
        base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
        result = neighbors("gcpv")
        for n in result:
            for ch in n:
                assert ch in base32


class TestEncodeValidation:
    """Tests for input validation in geohash encode()."""

    def test_invalid_latitude_raises(self):
        with pytest.raises(ValueError):
            encode(91.0, 0.0, precision=4)

    def test_invalid_longitude_raises(self):
        with pytest.raises(ValueError):
            encode(0.0, 181.0, precision=4)

    def test_precision_zero_raises(self):
        with pytest.raises(ValueError):
            encode(0.0, 0.0, precision=0)
