# Telegram Geoparsing Pipeline

## Overview

Extract geographic coordinates from Telegram messages in the listener before they enter the pipeline, so events arrive with lat/lon for geohash-based correlation.

## Architecture

Two-tier location resolution in a new `processing/geoparser.py` module, called from the Telegram listener's `_on_message` handler. Dictionary lookup first (instant), Nominatim geocoder fallback (async, ~200ms). Results cached in-memory.

## Module: `processing/geoparser.py`

### Public Interface

```python
async def geoparse(text: str) -> tuple[float, float] | None:
    """Return (lat, lon) or None if no location found."""
```

### Tier 1: Dictionary Lookup

A `dict[str, tuple[float, float]]` mapping ~35 location names (Arabic + English) to coordinates. The text is scanned for any dictionary key as a substring match (case-insensitive for English, exact for Arabic).

**Locations:**

| Region | Names |
|---|---|
| Palestine/Israel | غزة/Gaza, رفح/Rafah, خان يونس/Khan Younis, جباليا/Jabalia, القدس/Jerusalem, تل أبيب/Tel Aviv, حيفا/Haifa, بئر السبع/Beersheba, نابلس/Nablus, جنين/Jenin |
| Lebanon | بيروت/Beirut, الضاحية/Dahiyeh, صور/Tyre, النبطية/Nabatieh |
| Syria | دمشق/Damascus, حلب/Aleppo, حمص/Homs, اللاذقية/Latakia |
| Iran | طهران/Tehran, أصفهان/Isfahan, تبريز/Tabriz, شيراز/Shiraz |
| Iraq | بغداد/Baghdad, أربيل/Erbil, الموصل/Mosul, البصرة/Basra |
| Yemen | صنعاء/Sanaa, عدن/Aden, الحديدة/Hodeidah, مأرب/Marib |
| Saudi Arabia | الرياض/Riyadh, جدة/Jeddah |
| Jordan | عمّان/Amman |
| Egypt | القاهرة/Cairo |

Returns the first match found while scanning the text. Arabic names are checked before English to prioritize the language SabrenNewss uses.

### Tier 2: Nominatim Fallback

When no dictionary hit:

1. **Extract candidate:** Scan for Arabic prepositions that precede location names: "في" (in), "على" (on), "من" (from), "إلى" (to), "شمال" (north of), "جنوب" (south of), "قرب" (near), "استهداف" (targeting). Take the 1-3 words following the preposition.

2. **Geocode:** Send candidate to geopy Nominatim with `language=ar`.

3. **Rate limit:** Max 1 request/sec (Nominatim ToS). Enforced via `asyncio.Lock` + `asyncio.sleep(1)`.

4. **Cache:** In-memory `dict[str, tuple[float, float] | None]`. Cache both hits and misses. No TTL — place names don't move.

5. **Failure:** Timeout or no result → log and return None.

### Logging

- Dictionary hit: `[GEOPARSE] Dictionary: "غزة" → (31.5, 34.47)`
- Nominatim hit: `[GEOPARSE] Nominatim: "حي الشجاعية" → (31.52, 34.48)`
- Nominatim miss: `[GEOPARSE] Nominatim: no result for "المنطقة الشرقية"`
- No location found: `[GEOPARSE] No location extracted from text`

Uses `print()` for CloudWatch visibility (same pattern as current Telegram listener, pending structlog fix).

## Integration: Telegram Listener

In `ingestion/social_listener.py`, the `_on_message` handler changes:

```
Current:  keyword match → RawEvent(no lat/lon) → queue
New:      keyword match → geoparse(text) → RawEvent(with lat/lon if found) → queue
```

1. Text matches keyword
2. Call `await geoparse(text)`
3. If returns (lat, lon) → set on RawEvent
4. If returns None → emit event without coordinates (same as today)

No changes to triage loop, correlation engine, or any other module.

## Testing

### Unit tests (`tests/unit/test_geoparser.py`)

- Dictionary lookup: Arabic name returns correct lat/lon
- Dictionary lookup: English name returns same lat/lon
- Dictionary lookup: unknown name returns None
- Preposition extraction: "قصف في غزة" extracts "غزة" → dictionary hit
- Cache: second call for same text doesn't hit Nominatim

### Integration tests (`tests/integration/test_geoparser.py`)

- Mock Nominatim: text with unknown location → preposition extraction → Nominatim call → returns coordinates
- Nominatim failure: timeout → returns None gracefully

## Dependencies

- `geopy` — already installed (in pyproject.toml)
- No new dependencies needed

## Files

- Create: `processing/geoparser.py`
- Create: `tests/unit/test_geoparser.py`
- Create: `tests/integration/test_geoparser.py`
- Modify: `ingestion/social_listener.py` (add geoparse call in `_on_message`)
