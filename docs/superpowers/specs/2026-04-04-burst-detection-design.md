# Temporal Burst Detection

## Overview

Detect regional-scale events by identifying bursts of activity across multiple sensor sources within a short time window, regardless of geographic location. Complements the existing geohash-based point correlation.

## Problem

The geohash correlation engine matches events in the same ~20km cell. Regional events like missile barrages generate signals across thousands of kilometers (launches from Iran, aircraft rerouting over Mediterranean, sirens in Israel, Telegram reports from multiple locations). These never correlate geographically but are clearly the same event.

## Architecture

New `processing/burst_detector.py` module with a `BurstDetector` class. Called from the triage loop on every ingested event, alongside the existing geohash correlation. Independent — both run in parallel.

## Module: `processing/burst_detector.py`

### Public Interface

```python
class BurstDetector:
    def check(self, event: RawEvent) -> CorrelatedEvent | None
```

### Sliding Window

- Maintains a `deque` of `(timestamp, source, event)` tuples
- On every `check()` call: append the new event, prune entries older than 5 minutes
- After pruning: count distinct sources in the window
- If 5+ events from 3+ distinct sources AND cooldown has elapsed → emit burst alert
- Otherwise return None

### Burst Alert

When threshold is met, constructs a `CorrelatedEvent`:
- `severity`: CRITICAL
- `classification`: REGIONAL_EVENT (new enum value)
- `contributing_events`: all events currently in the window
- `corroborating_sources`: sorted list of distinct source names
- `latitude/longitude`: from the first contributing event that has coordinates
- `summary`: "REGIONAL EVENT: {N} events from {M} sources in 5 min: {source list}"

### Cooldown

After firing a burst alert, suppress further alerts for 5 minutes. One alert per burst, not one per event. Cooldown resets after 5 minutes of no burst.

### Parameters

- `window_sec`: 5 minutes (300 seconds)
- `min_events`: 5
- `min_sources`: 3
- `cooldown_sec`: 5 minutes (300 seconds)

## Integration

In the triage loop (`main.py`), after `engine.ingest(event)`:

```python
burst_result = burst_detector.check(event)
if burst_result:
    await alert_queue.put(burst_result)
```

Runs on every event regardless of source or whether LLM triage is needed. Independent of geohash correlation.

## Model Change

Add to `EventClassification` enum in `models.py`:
```python
REGIONAL_EVENT = "regional_event"
```

## Testing (`tests/unit/test_burst_detector.py`)

- Below event threshold: 4 events from 3 sources → None
- Below source threshold: 5 events from 2 sources → None
- At threshold: 5 events from 3 sources → fires CorrelatedEvent with all 5 contributing
- Cooldown: burst fires, then more events within cooldown → no second burst
- Window expiry: 5 events spread over 6 minutes → None
- Coordinates: picks first event with lat/lon
- Classification: REGIONAL_EVENT

## Files

- Create: `processing/burst_detector.py`
- Create: `tests/unit/test_burst_detector.py`
- Modify: `models.py` — add REGIONAL_EVENT to EventClassification
- Modify: `main.py` — instantiate BurstDetector, call check() in triage loop

## Dependencies

None — pure in-memory Python logic.
