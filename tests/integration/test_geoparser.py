"""Integration tests for geoparser Nominatim fallback."""

import pytest
from unittest.mock import patch, AsyncMock

from processing.geoparser import geoparse, _cache


@pytest.fixture(autouse=True)
def clear_cache():
    _cache.clear()
    yield
    _cache.clear()


class TestNominatimFallback:
    @pytest.mark.asyncio
    async def test_nominatim_called_when_no_dict_hit(self):
        with patch(
            "processing.geoparser._nominatim_geocode", new_callable=AsyncMock
        ) as mock_geo:
            mock_geo.return_value = (31.52, 34.48)
            result = await geoparse("انفجار في حي الشجاعية")
        assert result is not None
        assert abs(result[0] - 31.52) < 0.01

    @pytest.mark.asyncio
    async def test_nominatim_failure_returns_none(self):
        with patch(
            "processing.geoparser._nominatim_geocode", new_callable=AsyncMock
        ) as mock_geo:
            mock_geo.return_value = None
            result = await geoparse("انفجار في مكان غير معروف تماما")
        assert result is None

    @pytest.mark.asyncio
    async def test_dictionary_hit_skips_nominatim(self):
        with patch(
            "processing.geoparser._nominatim_geocode", new_callable=AsyncMock
        ) as mock_geo:
            result = await geoparse("قصف على غزة")
        mock_geo.assert_not_called()
        assert result is not None

    @pytest.mark.asyncio
    async def test_cache_prevents_repeat_calls(self):
        with patch(
            "processing.geoparser._nominatim_geocode", new_callable=AsyncMock
        ) as mock_geo:
            mock_geo.return_value = (31.52, 34.48)
            await geoparse("قصف في حي الشجاعية")
            await geoparse("قصف في حي الشجاعية")
        assert mock_geo.call_count == 1
