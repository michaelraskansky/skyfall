# SATCAT Enrichment Design

Enrich trajectory alerts with satellite identity data from Space-Track's SATCAT catalog.

## Problem

Current alerts show "NORAD 53493" — an opaque ID. Operators need to know what the object is (STARLINK-4361), who launched it (US), when (2022-08-12), and what type it is (PAYLOAD vs DEBRIS vs ROCKET BODY).

## Architecture

### New module: `ingestion/satcat_lookup.py`

`SatcatLookup` class with `connect()`/`close()` lifecycle (matches CorrelationEngine/ObjectTracker pattern). Authenticates against Space-Track on connect, queries SATCAT on-demand per NORAD_CAT_ID.

**API endpoint:**
```
https://www.space-track.org/basicspacedata/query/class/satcat/NORAD_CAT_ID/{id}/format/json
```

**Caching:** In-memory dict keyed by NORAD_CAT_ID. SATCAT data is static — cache lives for process lifetime, no invalidation needed.

**Return model:**
```python
class SatcatInfo(BaseModel):
    norad_cat_id: str
    object_name: str = "UNKNOWN"
    country: str = "UNKNOWN"
    launch_date: str = ""
    object_type: str = ""
    rcs_size: str = ""
```

**Failure mode:** Returns `None` on any error. Enrichment is optional — alerts fire without it.

**Re-auth:** If a SATCAT query returns HTTP 401, re-authenticate once and retry.

### Model change: `models.py`

Add `satcat_info: Optional[SatcatInfo] = None` to `CorrelatedEvent`.

### Pipeline integration: `main.py`

In triage loop, after `tracker.track_observation(event)`:
1. Call `satcat_lookup.get_info(norad_id)` (cache hit or API call)
2. Attach `SatcatInfo` to the `CorrelatedEvent` when building trajectory events
3. Include object name in summary string

SatcatLookup lifecycle managed in `main()` alongside engine and tracker.

### Formatter: `output/formatter.py`

When `event.satcat_info` is present, add `"object_info"` block to alert payload.

### Alerter: `output/alerter.py`

Add object identity section to Slack blocks and Discord embeds.

## Files Changed

| File | Change |
|------|--------|
| `ingestion/satcat_lookup.py` | New — SatcatLookup class, SatcatInfo model, in-memory cache |
| `models.py` | Add satcat_info field to CorrelatedEvent |
| `main.py` | Initialize SatcatLookup, call in triage loop |
| `output/formatter.py` | Add object_info block |
| `output/alerter.py` | Add object info to Slack/Discord |
| `tests/unit/test_satcat_lookup.py` | Unit tests for cache, model |
| `tests/unit/test_formatter.py` | Test object_info in payload |
