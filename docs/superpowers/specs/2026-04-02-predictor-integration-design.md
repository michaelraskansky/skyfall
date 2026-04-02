# Predictor Integration Design

Wire the orphaned `DebrisTrajectoryPredictor` into the live pipeline, add a DynamoDB access pattern for object-based queries, and build a mock data injector for local end-to-end testing.

## Problem

The trajectory prediction engine (`trajectory/predictor.py`) is fully implemented but disconnected from the pipeline. The DynamoDB table uses `PK=geohash#<4-char>` which supports spatial correlation but cannot query by object ID. TIP observations for the same NORAD_CAT_ID arrive 30+ minutes apart and expire under the 6-minute correlation TTL before a second observation arrives.

## Approved Approach: Dual-Write (Option B)

Write a second item per observation into the same DynamoDB table with `PK=object#<NORAD_CAT_ID>` and a 24-hour TTL. This preserves the existing geohash correlation unchanged while enabling efficient object-based time-series queries.

## Section 1: DynamoDB Dual-Write & Object Tracker

### New module: `processing/object_tracker.py`

**Schema for object-tracking items:**

| Field | Value |
|-------|-------|
| `pk` | `object#<NORAD_CAT_ID>` |
| `sk` | `<ISO-timestamp>#<event_id>` — UTC, zero-padded milliseconds for lexicographic sort |
| `expires_at` | `int(time.time()) + 86400` — 24h TTL, integer Unix epoch seconds (DynamoDB TTL requirement) |
| `latitude` | Decimal string |
| `longitude` | Decimal string |
| `altitude_m` | Decimal string — extracted from TIP payload, default `10000.0` |
| `timestamp` | ISO-8601 UTC string |
| `source` | Event source value |
| `noise_profile` | `"satellite"` for Space-Track TIPs |
| `description` | Event description |
| `raw_payload` | JSON-serialized payload |

**Public API:**
- `track_observation(event: RawEvent) -> None` — Dual-writes the object-keyed item. Only called when `raw_payload` contains `NORAD_CAT_ID`.
- `get_observations(object_id: str) -> list[SensorObservation]` — Queries `PK=object#<object_id>`, returns trajectory-engine-compatible `SensorObservation` objects sorted by timestamp ASC.

**Connection management:** Receives the DynamoDB table reference from the same `aioboto3` session used by `CorrelationEngine`. Initialized in `main.py` alongside the engine.

**Altitude extraction:** Space-Track TIPs report position at ~10km altitude. Extract from `raw_payload` if present, otherwise default to `10000.0`.

**Trigger condition:** Any `RawEvent` whose `raw_payload` contains a `NORAD_CAT_ID` key.

## Section 2: Trigger Logic & Predictor Wiring

### Changes to `main.py` triage_loop

After existing `engine.ingest()` and `engine.try_correlate()`, add:

1. Check `event.raw_payload.get("NORAD_CAT_ID")`.
2. If present, call `tracker.track_observation(event)`.
3. Call `tracker.get_observations(norad_id)` for the full sorted history.
4. If `len(observations) >= 2`:
   - Build `TrajectoryRequest(object_id=norad_id, observations=observations)`.
   - Run `DebrisTrajectoryPredictor().predict(request)` via `asyncio.to_thread()` (non-blocking — protects the event loop from CPU-bound numpy math).
   - Construct a `CorrelatedEvent` with `severity=CRITICAL`, `classification=DEBRIS_REENTRY`, attach the `ImpactPrediction`.
   - Push to `alert_queue`.

### Model change: `models.py`

Add optional field to `CorrelatedEvent`:
```python
impact_prediction: Optional[ImpactPrediction] = None
```

This requires importing `ImpactPrediction` from `trajectory.models`.

### Output changes: `output/formatter.py`

When `event.impact_prediction` is present, append an `impact_prediction` block to the alert payload containing: impact lat/lon, time of impact, seconds until impact, terminal velocity, and 95% confidence ellipse semi-axes.

### Output changes: `output/alerter.py`

Add impact prediction details to Slack blocks and Discord embeds when present in the payload.

### Alert frequency

Every observation >= 2 triggers a new prediction and alert. This means observations #2, #3, #4 all fire separate alerts. Accepted for V1 — debounce/cooldown deferred to a future sprint.

### Predictor instantiation

Fresh `DebrisTrajectoryPredictor` per prediction run. No shared state — the EKF is stateful and tied to a specific observation sequence.

### Ballistic coefficient

Default `50.0 kg/m²`. Not derivable from TIP data. Configurable via `config.py` in a future iteration.

## Section 3: Mock Data Injector

### New script: `scripts/inject_mock_trajectory.py`

Standalone CLI script that generates 4 synthetic TIP observations and POSTs them to the local webhook server.

**Trajectory:** Debris re-entering over US Southwest. Start at 37.0N, 115.0W, 120km altitude. Heading SSW at ~2 km/s. Four observations at relative offsets of 0s, 25s, 55s, 90s.

**Mechanics:**
- POSTs to `http://localhost:8000/api/v1/test-event` with `source: "spacetrack"`.
- Each payload includes `NORAD_CAT_ID: "99999"`, `LAT`, `LON`, `DECAY_EPOCH`, and `ALTITUDE_M`.
- Observation timestamps are spaced 25s apart (matching demo trajectory physics) but POSTed 2s apart in wall-clock time.
- Adds Gaussian noise per the demo's noise profiles: satellite, thermal, social_media, thermal.
- Uses synchronous `httpx` client.
- Prints HTTP response for each POST.

**Expected behavior against `docker compose up`:**
1. Observation 1: ingested + dual-written to `object#99999`. No prediction.
2. Observation 2: prediction fires, CRITICAL alert with impact coordinates sent to Slack/Discord.
3. Observations 3-4: refined predictions, additional alerts.

## Files Changed

| File | Change |
|------|--------|
| `processing/object_tracker.py` | **New** — ObjectTracker class |
| `models.py` | Add `impact_prediction` field to `CorrelatedEvent` |
| `main.py` | Import tracker/predictor, add tracking logic in triage_loop, lifecycle management |
| `output/formatter.py` | Add impact_prediction block to alert payload |
| `output/alerter.py` | Add impact details to Slack/Discord message blocks |
| `scripts/inject_mock_trajectory.py` | **New** — mock data injection script |
| `infra/template.yaml` | No changes — same table, no GSI needed |
