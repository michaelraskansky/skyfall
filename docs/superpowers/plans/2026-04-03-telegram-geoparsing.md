# Telegram Geoparsing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract lat/lon from Telegram messages in the listener so events enter the pipeline with coordinates for geohash-based correlation.

**Architecture:** New `processing/geoparser.py` module with two-tier resolution: dictionary lookup (~35 Middle East locations) then Nominatim geocoder fallback. Called from Telegram listener's `_on_message` before constructing the RawEvent.

**Tech Stack:** geopy (already installed), asyncio, regex for preposition extraction

---

## File Structure

### New Files

- `processing/geoparser.py` — Two-tier geoparser: dictionary + Nominatim fallback
- `tests/unit/test_geoparser.py` — Unit tests for dictionary lookup and preposition extraction
- `tests/integration/test_geoparser.py` — Integration tests with mocked Nominatim

### Modified Files

- `ingestion/social_listener.py:135-152` — Add geoparse call in `_on_message`

---

### Task 1: Geoparser Module — Dictionary Lookup

**Files:**
- Create: `processing/geoparser.py`
- Create: `tests/unit/test_geoparser.py`

- [ ] **Step 1: Write failing tests for dictionary lookup**

Create `tests/unit/test_geoparser.py`:

```python
"""Unit tests for the geoparser module."""

import pytest

from processing.geoparser import geoparse, _dictionary_lookup


class TestDictionaryLookup:
    def test_arabic_gaza(self):
        result = _dictionary_lookup("قصف عنيف على غزة")
        assert result is not None
        lat, lon = result
        assert abs(lat - 31.5) < 0.1
        assert abs(lon - 34.47) < 0.1

    def test_english_gaza(self):
        result = _dictionary_lookup("Heavy shelling in Gaza")
        assert result is not None
        lat, lon = result
        assert abs(lat - 31.5) < 0.1
        assert abs(lon - 34.47) < 0.1

    def test_arabic_tehran(self):
        result = _dictionary_lookup("انطلاق صواريخ من طهران")
        assert result is not None
        lat, lon = result
        assert abs(lat - 35.69) < 0.1
        assert abs(lon - 51.39) < 0.1

    def test_arabic_damascus(self):
        result = _dictionary_lookup("غارات على دمشق")
        assert result is not None
        lat, lon = result
        assert abs(lat - 33.51) < 0.1
        assert abs(lon - 36.29) < 0.1

    def test_arabic_beirut(self):
        result = _dictionary_lookup("انفجار في بيروت")
        assert result is not None
        lat, lon = result
        assert abs(lat - 33.89) < 0.1
        assert abs(lon - 35.50) < 0.1

    def test_english_case_insensitive(self):
        result = _dictionary_lookup("Missile strike on TEHRAN")
        assert result is not None

    def test_no_location(self):
        result = _dictionary_lookup("حالة الطقس اليوم جميلة")
        assert result is None

    def test_unknown_location(self):
        result = _dictionary_lookup("explosion in Atlantis")
        assert result is None

    def test_multiple_locations_returns_first(self):
        """When text has multiple locations, return the first one found."""
        result = _dictionary_lookup("صواريخ من طهران إلى تل أبيب")
        assert result is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_geoparser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'processing.geoparser'`

- [ ] **Step 3: Implement dictionary lookup**

Create `processing/geoparser.py`:

```python
"""
Geoparser — Location Extraction from Text
===========================================

Two-tier location resolution for extracting lat/lon from unstructured
text (primarily Arabic news from Telegram):

1. **Dictionary lookup** — instant substring match against ~35 known
   Middle East locations.
2. **Nominatim fallback** — async geocoder for locations not in the
   dictionary.
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional, Tuple

# Type alias for coordinates
Coords = Tuple[float, float]

# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1: Dictionary of known locations
# ═══════════════════════════════════════════════════════════════════════════════

# Maps location names (Arabic + English) to (latitude, longitude).
# Arabic names are checked first since SabrenNewss posts in Arabic.
_LOCATIONS: dict[str, Coords] = {
    # Palestine / Israel
    "غزة": (31.5, 34.47),
    "Gaza": (31.5, 34.47),
    "رفح": (31.3, 34.24),
    "Rafah": (31.3, 34.24),
    "خان يونس": (31.34, 34.3),
    "Khan Younis": (31.34, 34.3),
    "جباليا": (31.53, 34.48),
    "Jabalia": (31.53, 34.48),
    "القدس": (31.77, 35.23),
    "Jerusalem": (31.77, 35.23),
    "تل أبيب": (32.08, 34.78),
    "Tel Aviv": (32.08, 34.78),
    "حيفا": (32.79, 34.99),
    "Haifa": (32.79, 34.99),
    "بئر السبع": (31.25, 34.79),
    "Beersheba": (31.25, 34.79),
    "نابلس": (32.22, 35.25),
    "Nablus": (32.22, 35.25),
    "جنين": (32.46, 35.3),
    "Jenin": (32.46, 35.3),
    # Lebanon
    "بيروت": (33.89, 35.50),
    "Beirut": (33.89, 35.50),
    "الضاحية": (33.86, 35.53),
    "Dahiyeh": (33.86, 35.53),
    "صور": (33.27, 35.20),
    "Tyre": (33.27, 35.20),
    "النبطية": (33.38, 35.48),
    "Nabatieh": (33.38, 35.48),
    # Syria
    "دمشق": (33.51, 36.29),
    "Damascus": (33.51, 36.29),
    "حلب": (36.20, 37.15),
    "Aleppo": (36.20, 37.15),
    "حمص": (34.73, 36.72),
    "Homs": (34.73, 36.72),
    "اللاذقية": (35.52, 35.78),
    "Latakia": (35.52, 35.78),
    # Iran
    "طهران": (35.69, 51.39),
    "Tehran": (35.69, 51.39),
    "أصفهان": (32.65, 51.68),
    "Isfahan": (32.65, 51.68),
    "تبريز": (38.08, 46.29),
    "Tabriz": (38.08, 46.29),
    "شيراز": (29.59, 52.58),
    "Shiraz": (29.59, 52.58),
    # Iraq
    "بغداد": (33.31, 44.37),
    "Baghdad": (33.31, 44.37),
    "أربيل": (36.19, 44.01),
    "Erbil": (36.19, 44.01),
    "الموصل": (36.34, 43.14),
    "Mosul": (36.34, 43.14),
    "البصرة": (30.51, 47.81),
    "Basra": (30.51, 47.81),
    # Yemen
    "صنعاء": (15.35, 44.21),
    "Sanaa": (15.35, 44.21),
    "عدن": (12.79, 45.02),
    "Aden": (12.79, 45.02),
    "الحديدة": (14.80, 42.95),
    "Hodeidah": (14.80, 42.95),
    "مأرب": (15.46, 45.32),
    "Marib": (15.46, 45.32),
    # Saudi Arabia
    "الرياض": (24.71, 46.67),
    "Riyadh": (24.71, 46.67),
    "جدة": (21.49, 39.19),
    "Jeddah": (21.49, 39.19),
    # Jordan
    "عمّان": (31.95, 35.93),
    "Amman": (31.95, 35.93),
    # Egypt
    "القاهرة": (30.04, 31.24),
    "Cairo": (30.04, 31.24),
}

# Build a regex pattern matching any known location name.
# Sort by length descending so longer names match first (e.g. "خان يونس" before "يونس").
_DICT_KEYS_SORTED = sorted(_LOCATIONS.keys(), key=len, reverse=True)
_DICT_PATTERN = re.compile(
    "|".join(re.escape(k) for k in _DICT_KEYS_SORTED),
    re.IGNORECASE,
)


def _dictionary_lookup(text: str) -> Coords | None:
    """Scan text for any known location name. Return (lat, lon) or None."""
    match = _DICT_PATTERN.search(text)
    if match:
        key = match.group()
        # Try exact key first, then case-insensitive search
        if key in _LOCATIONS:
            return _LOCATIONS[key]
        for k, v in _LOCATIONS.items():
            if k.lower() == key.lower():
                return v
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_geoparser.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add processing/geoparser.py tests/unit/test_geoparser.py
git commit -m "feat: add geoparser with dictionary lookup for ~35 Middle East locations"
```

---

### Task 2: Geoparser — Preposition Extraction and Nominatim Fallback

**Files:**
- Modify: `processing/geoparser.py`
- Create: `tests/integration/test_geoparser.py`

- [ ] **Step 1: Write failing tests for preposition extraction**

Add to `tests/unit/test_geoparser.py`:

```python
from processing.geoparser import _extract_location_candidate


class TestPrepositionExtraction:
    def test_fi_in(self):
        """'في' (in) followed by location."""
        result = _extract_location_candidate("انفجار في حي الشجاعية")
        assert result == "حي الشجاعية"

    def test_ala_on(self):
        """'على' (on) followed by location."""
        result = _extract_location_candidate("قصف على المنطقة الجنوبية")
        assert result == "المنطقة الجنوبية"

    def test_min_from(self):
        """'من' (from) followed by location."""
        result = _extract_location_candidate("صواريخ من الضفة الغربية")
        assert result == "الضفة الغربية"

    def test_no_preposition(self):
        result = _extract_location_candidate("هذا خبر بدون موقع")
        assert result is None

    def test_shamaal_north(self):
        """'شمال' (north of) followed by location."""
        result = _extract_location_candidate("قصف شمال المدينة")
        assert result == "المدينة"

    def test_max_three_words(self):
        """Extract at most 3 words after preposition."""
        result = _extract_location_candidate("انفجار في كلمة واحدة اثنتان ثلاث أربع")
        # Should take at most 3 words
        words = result.split() if result else []
        assert len(words) <= 3
```

- [ ] **Step 2: Write failing integration tests for Nominatim**

Create `tests/integration/test_geoparser.py`:

```python
"""Integration tests for geoparser Nominatim fallback."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from processing.geoparser import geoparse, _nominatim_geocode, _cache


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear geoparser cache between tests."""
    _cache.clear()
    yield
    _cache.clear()


class TestNominatimFallback:
    async def test_nominatim_called_when_no_dict_hit(self):
        """Unknown location should trigger Nominatim."""
        mock_location = MagicMock()
        mock_location.latitude = 31.52
        mock_location.longitude = 34.48

        with patch("processing.geoparser._nominatim_geocode", new_callable=AsyncMock) as mock_geo:
            mock_geo.return_value = (31.52, 34.48)
            result = await geoparse("انفجار في حي الشجاعية")

        assert result is not None
        lat, lon = result
        assert abs(lat - 31.52) < 0.01

    async def test_nominatim_failure_returns_none(self):
        """Nominatim timeout or error should return None gracefully."""
        with patch("processing.geoparser._nominatim_geocode", new_callable=AsyncMock) as mock_geo:
            mock_geo.return_value = None
            result = await geoparse("انفجار في مكان غير معروف تماما")

        assert result is None

    async def test_dictionary_hit_skips_nominatim(self):
        """Known location should NOT call Nominatim."""
        with patch("processing.geoparser._nominatim_geocode", new_callable=AsyncMock) as mock_geo:
            result = await geoparse("قصف على غزة")

        mock_geo.assert_not_called()
        assert result is not None

    async def test_cache_prevents_repeat_nominatim_calls(self):
        """Same text should only call Nominatim once."""
        with patch("processing.geoparser._nominatim_geocode", new_callable=AsyncMock) as mock_geo:
            mock_geo.return_value = (31.52, 34.48)
            await geoparse("قصف في حي الشجاعية")
            await geoparse("قصف في حي الشجاعية")

        assert mock_geo.call_count == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_geoparser.py::TestPrepositionExtraction tests/integration/test_geoparser.py -v`
Expected: FAIL — `_extract_location_candidate` and `_nominatim_geocode` don't exist, `geoparse` not fully implemented.

- [ ] **Step 4: Implement preposition extraction, Nominatim fallback, and geoparse**

Add to `processing/geoparser.py` (after the dictionary lookup code):

```python
# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2: Preposition extraction + Nominatim geocoding
# ═══════════════════════════════════════════════════════════════════════════════

# Arabic prepositions that typically precede location names.
_PREPOSITIONS = [
    "في",        # in
    "على",       # on/at
    "من",        # from
    "إلى",       # to
    "شمال",      # north of
    "جنوب",      # south of
    "شرق",       # east of
    "غرب",       # west of
    "قرب",       # near
    "استهداف",   # targeting
]

_PREP_PATTERN = re.compile(
    r"(?:" + "|".join(re.escape(p) for p in _PREPOSITIONS) + r")\s+"
    r"([\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+){0,2})",
    re.UNICODE,
)


def _extract_location_candidate(text: str) -> str | None:
    """
    Extract a candidate location string by finding Arabic words
    after a spatial preposition. Returns 1-3 Arabic words or None.
    """
    match = _PREP_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


# In-memory cache: candidate string → Coords or None
_cache: dict[str, Coords | None] = {}

# Rate-limit lock for Nominatim (1 req/sec)
_nominatim_lock = asyncio.Lock()


async def _nominatim_geocode(candidate: str) -> Coords | None:
    """
    Geocode a location string via geopy Nominatim.
    Rate-limited to 1 request/sec. Returns (lat, lon) or None.
    """
    from geopy.adapters import AioHTTPAdapter
    from geopy.geocoders import Nominatim

    async with _nominatim_lock:
        try:
            async with Nominatim(
                user_agent="skyfall-geoparser",
                adapter_factory=AioHTTPAdapter,
            ) as geolocator:
                location = await geolocator.geocode(
                    candidate,
                    language="ar",
                    timeout=5,
                )
                if location:
                    print(
                        f'[GEOPARSE] Nominatim: "{candidate}" → '
                        f"({location.latitude}, {location.longitude})",
                        flush=True,
                    )
                    return (location.latitude, location.longitude)
                else:
                    print(
                        f'[GEOPARSE] Nominatim: no result for "{candidate}"',
                        flush=True,
                    )
                    return None
        except Exception as e:
            print(f'[GEOPARSE] Nominatim error for "{candidate}": {e}', flush=True)
            return None
        finally:
            await asyncio.sleep(1)  # Nominatim ToS: max 1 req/sec


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


async def geoparse(text: str) -> Coords | None:
    """
    Extract geographic coordinates from text.

    Tier 1: Dictionary lookup (instant, ~35 known locations).
    Tier 2: Preposition extraction + Nominatim geocoding (async, ~200ms).

    Returns (lat, lon) or None if no location found.
    """
    # Tier 1: dictionary
    result = _dictionary_lookup(text)
    if result:
        print(f'[GEOPARSE] Dictionary hit → ({result[0]}, {result[1]})', flush=True)
        return result

    # Tier 2: extract candidate, check cache, then Nominatim
    candidate = _extract_location_candidate(text)
    if not candidate:
        print("[GEOPARSE] No location extracted from text", flush=True)
        return None

    if candidate in _cache:
        return _cache[candidate]

    coords = await _nominatim_geocode(candidate)
    _cache[candidate] = coords
    return coords
```

- [ ] **Step 5: Run all geoparser tests**

Run: `uv run pytest tests/unit/test_geoparser.py tests/integration/test_geoparser.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add processing/geoparser.py tests/unit/test_geoparser.py tests/integration/test_geoparser.py
git commit -m "feat: add Nominatim fallback with preposition extraction and caching"
```

---

### Task 3: Integrate Geoparser into Telegram Listener

**Files:**
- Modify: `ingestion/social_listener.py:135-152`

- [ ] **Step 1: Update `_on_message` handler**

Replace the `_on_message` function body in `ingestion/social_listener.py` (lines 136-152):

```python
    @client.on(tg_events.NewMessage(chats=channels))
    async def _on_message(tg_event):
        text: str = tg_event.raw_text or ""
        if not _matches_keywords(text):
            return

        # Extract location from the message text
        from processing.geoparser import geoparse

        coords = await geoparse(text)
        lat, lon = coords if coords else (None, None)

        event = RawEvent(
            source=EventSource.SOCIAL_MEDIA,
            latitude=lat,
            longitude=lon,
            raw_payload={
                "platform": "telegram",
                "channel": str(tg_event.chat_id),
                "text": text,
                "message_id": tg_event.id,
            },
            description=f"Telegram keyword match: {text[:200]}",
        )
        print(
            f"[TELEGRAM] Keyword hit: {event.description[:120]}"
            f" → coords=({lat}, {lon})",
            flush=True,
        )
        await event_queue.put(event)
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS (existing + new geoparser tests)

- [ ] **Step 3: Commit**

```bash
git add ingestion/social_listener.py
git commit -m "feat: integrate geoparser into Telegram listener for lat/lon extraction"
```

---

### Task 4: Build, Push, Deploy

- [ ] **Step 1: Build and push Docker image**

```bash
docker build --platform linux/amd64 -t skyfall:latest .
aws ecr get-login-password --profile iron5dev --region eu-central-1 | docker login --username AWS --password-stdin 820885688406.dkr.ecr.eu-central-1.amazonaws.com
docker tag skyfall:latest 820885688406.dkr.ecr.eu-central-1.amazonaws.com/skyfall:latest
docker push 820885688406.dkr.ecr.eu-central-1.amazonaws.com/skyfall:latest
```

- [ ] **Step 2: Deploy**

```bash
aws ecs update-service --cluster skyfall --service skyfall --force-new-deployment --profile iron5dev --region eu-central-1
```

- [ ] **Step 3: Verify logs**

Wait ~90 seconds, then check CloudWatch logs for:
- `[TELEGRAM] Client connected, listening to ['SabrenNewss']`
- Any `[GEOPARSE]` log lines when messages come in

- [ ] **Step 4: Commit plan completion**

```bash
git add docs/superpowers/plans/2026-04-03-telegram-geoparsing.md
git commit -m "docs: add geoparsing implementation plan"
```
