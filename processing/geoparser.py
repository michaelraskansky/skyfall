"""Geoparser — extract geographic coordinates from text.

Tier 1: Dictionary lookup against ~35 known Middle-East locations.
Tier 2: Preposition extraction + Nominatim geocoding fallback.
"""

from __future__ import annotations

import asyncio
import re
from typing import Tuple

Coords = Tuple[float, float]

# ---------------------------------------------------------------------------
# Preposition extraction
# ---------------------------------------------------------------------------

_PREPOSITIONS = [
    "في",       # in
    "على",      # on
    "من",       # from
    "إلى",      # to
    "شمال",     # north of
    "جنوب",     # south of
    "شرق",      # east of
    "غرب",      # west of
    "قرب",      # near
    "استهداف",  # targeting
]

_PREP_PATTERN = re.compile(
    r"(?:" + "|".join(re.escape(p) for p in _PREPOSITIONS) + r")\s+"
    r"([\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+){0,2})",
    re.UNICODE,
)

# ---------------------------------------------------------------------------
# Nominatim cache & rate-limit lock
# ---------------------------------------------------------------------------

_cache: dict[str, Coords | None] = {}
_nominatim_lock = asyncio.Lock()


# Mapping of location names (Arabic + English) to (lat, lon).
_LOCATIONS: dict[str, Coords] = {
    # --- Palestine / Israel ---
    "غزة": (31.50, 34.47),
    "Gaza": (31.50, 34.47),
    "رفح": (31.30, 34.24),
    "Rafah": (31.30, 34.24),
    "خان يونس": (31.34, 34.30),
    "Khan Younis": (31.34, 34.30),
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
    "جنين": (32.46, 35.30),
    "Jenin": (32.46, 35.30),
    # --- Lebanon ---
    "بيروت": (33.89, 35.50),
    "Beirut": (33.89, 35.50),
    "الضاحية": (33.85, 35.50),
    "Dahiyeh": (33.85, 35.50),
    "صور": (33.27, 35.20),
    "Tyre": (33.27, 35.20),
    "النبطية": (33.38, 35.49),
    "Nabatieh": (33.38, 35.49),
    # --- Syria ---
    "دمشق": (33.51, 36.29),
    "Damascus": (33.51, 36.29),
    "حلب": (36.20, 37.15),
    "Aleppo": (36.20, 37.15),
    "حمص": (34.73, 36.72),
    "Homs": (34.73, 36.72),
    "اللاذقية": (35.53, 35.78),
    "Latakia": (35.53, 35.78),
    # --- Iran ---
    "طهران": (35.69, 51.39),
    "Tehran": (35.69, 51.39),
    "أصفهان": (32.65, 51.68),
    "Isfahan": (32.65, 51.68),
    "تبريز": (38.08, 46.29),
    "Tabriz": (38.08, 46.29),
    "شيراز": (29.59, 52.58),
    "Shiraz": (29.59, 52.58),
    # --- Iraq ---
    "بغداد": (33.31, 44.37),
    "Baghdad": (33.31, 44.37),
    "أربيل": (36.19, 44.01),
    "Erbil": (36.19, 44.01),
    "الموصل": (36.34, 43.12),
    "Mosul": (36.34, 43.12),
    "البصرة": (30.51, 47.81),
    "Basra": (30.51, 47.81),
    # --- Yemen ---
    "صنعاء": (15.35, 44.21),
    "Sanaa": (15.35, 44.21),
    "عدن": (12.78, 45.02),
    "Aden": (12.78, 45.02),
    "الحديدة": (14.80, 42.95),
    "Hodeidah": (14.80, 42.95),
    "مأرب": (15.46, 45.32),
    "Marib": (15.46, 45.32),
    # --- Saudi Arabia ---
    "الرياض": (24.71, 46.68),
    "Riyadh": (24.71, 46.68),
    "جدة": (21.54, 39.17),
    "Jeddah": (21.54, 39.17),
    # --- Jordan ---
    "عمان": (31.95, 35.93),
    "Amman": (31.95, 35.93),
    # --- Egypt ---
    "القاهرة": (30.04, 31.24),
    "Cairo": (30.04, 31.24),
}

# Build a case-insensitive regex that matches any known location name.
# Keys are sorted longest-first so that multi-word names match before their
# sub-strings (e.g. "Khan Younis" before "Khan").
_sorted_keys = sorted(_LOCATIONS.keys(), key=len, reverse=True)
_DICT_PATTERN: re.Pattern[str] = re.compile(
    "|".join(re.escape(k) for k in _sorted_keys),
    re.IGNORECASE,
)


def _dictionary_lookup(text: str) -> Coords | None:
    """Scan *text* for the first known location name and return its coords.

    Returns ``None`` when no location is found.
    """
    m = _DICT_PATTERN.search(text)
    if m is None:
        return None
    # Look up using the canonical (case-preserved) key.  Because the regex is
    # case-insensitive we need to find the matching key explicitly.
    matched = m.group()
    # Try exact match first (covers Arabic and correctly-cased English).
    if matched in _LOCATIONS:
        return _LOCATIONS[matched]
    # Fall back to case-insensitive lookup for English variants.
    matched_lower = matched.lower()
    for key, coords in _LOCATIONS.items():
        if key.lower() == matched_lower:
            return coords
    return None  # pragma: no cover — unreachable if regex and dict are in sync


# ---------------------------------------------------------------------------
# Tier 2 helpers
# ---------------------------------------------------------------------------


def _extract_location_candidate(text: str) -> str | None:
    """Extract a location candidate following an Arabic preposition.

    Returns up to 3 Arabic words after a known preposition, or ``None``.
    """
    m = _PREP_PATTERN.search(text)
    if m is None:
        return None
    return m.group(1)


async def _nominatim_geocode(candidate: str) -> Coords | None:
    """Geocode *candidate* via Nominatim (rate-limited to 1 req/sec)."""
    from geopy.adapters import AioHTTPAdapter
    from geopy.geocoders import Nominatim

    async with _nominatim_lock:
        try:
            async with Nominatim(
                user_agent="skyfall-geoparser", adapter_factory=AioHTTPAdapter
            ) as geolocator:
                location = await geolocator.geocode(
                    candidate, language="ar", timeout=5
                )
            if location is not None:
                print(
                    f"[GEOPARSE] Nominatim hit: {candidate!r} -> "
                    f"({location.latitude}, {location.longitude})"
                )
                return (location.latitude, location.longitude)
            print(f"[GEOPARSE] Nominatim miss: {candidate!r}")
            return None
        except Exception as exc:
            print(f"[GEOPARSE] Nominatim error for {candidate!r}: {exc}")
            return None
        finally:
            await asyncio.sleep(1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def geoparse(text: str) -> Coords | None:
    """Return coordinates for a location mentioned in *text*, or ``None``.

    Resolution order:
    1. Dictionary lookup (fast, offline).
    2. Preposition extraction + Nominatim geocoding (cached).
    """
    # Tier 1 — dictionary
    coords = _dictionary_lookup(text)
    if coords is not None:
        print(f"[GEOPARSE] Dictionary hit for text")
        return coords

    # Tier 2 — preposition extraction + Nominatim
    candidate = _extract_location_candidate(text)
    if candidate is None:
        print("[GEOPARSE] No location candidate extracted")
        return None

    # Check cache
    if candidate in _cache:
        print(f"[GEOPARSE] Cache hit: {candidate!r}")
        return _cache[candidate]

    # Query Nominatim
    result = await _nominatim_geocode(candidate)
    _cache[candidate] = result
    return result
