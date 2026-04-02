"""Geoparser — extract geographic coordinates from text.

Tier 1: Dictionary lookup against ~35 known Middle-East locations.
"""

from __future__ import annotations

import re
from typing import Tuple

Coords = Tuple[float, float]

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
