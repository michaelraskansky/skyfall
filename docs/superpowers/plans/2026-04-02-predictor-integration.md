# Predictor Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the orphaned `DebrisTrajectoryPredictor` into the live pipeline via a DynamoDB dual-write pattern, trigger logic in the triage loop, and a mock data injector for local E2E testing.

**Architecture:** A new `ObjectTracker` writes object-keyed items (PK=`object#<NORAD_CAT_ID>`) to the same DynamoDB table with 24h TTL. The triage loop in `main.py` detects events with NORAD_CAT_ID, tracks them, queries history, and triggers the EKF predictor via `asyncio.to_thread()`. The resulting `ImpactPrediction` is wrapped in a `CorrelatedEvent` and pushed through the existing alert pipeline.

**Tech Stack:** Python 3.11+, aioboto3, DynamoDB (single-table), numpy (EKF), FastAPI, httpx, pytest + moto

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `processing/object_tracker.py` | Create | DynamoDB dual-write and object-based observation queries |
| `models.py` | Modify | Add `impact_prediction` field to `CorrelatedEvent` |
| `main.py` | Modify | Wire tracker + predictor into triage loop |
| `output/formatter.py` | Modify | Add impact prediction block to alert payload |
| `output/alerter.py` | Modify | Add impact details to Slack/Discord message blocks |
| `scripts/inject_mock_trajectory.py` | Create | Mock TIP observation injector for local testing |
| `tests/integration/test_object_tracker.py` | Create | Integration tests for ObjectTracker |
| `tests/unit/test_formatter.py` | Modify | Test impact prediction in formatted payload |
| `tests/unit/test_models.py` | Modify | Test new CorrelatedEvent field |

---

### Task 1: Add `impact_prediction` field to `CorrelatedEvent`

**Files:**
- Modify: `models.py:83-99`
- Modify: `tests/unit/test_models.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_models.py`:

```python
from trajectory.models import ImpactPrediction
from datetime import datetime, timezone


def test_correlated_event_impact_prediction_default_none():
    """CorrelatedEvent.impact_prediction defaults to None."""
    event = CorrelatedEvent()
    assert event.impact_prediction is None


def test_correlated_event_accepts_impact_prediction():
    """CorrelatedEvent can hold an ImpactPrediction."""
    prediction = ImpactPrediction(
        object_id="99999",
        impact_latitude=35.0,
        impact_longitude=-115.0,
        impact_altitude_m=0.0,
        time_of_impact_utc=datetime(2026, 4, 1, 15, 0, 0, tzinfo=timezone.utc),
        seconds_until_impact=120.0,
        terminal_velocity_m_s=200.0,
        covariance_position_enu=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    event = CorrelatedEvent(impact_prediction=prediction)
    assert event.impact_prediction is not None
    assert event.impact_prediction.object_id == "99999"
    assert event.impact_prediction.impact_latitude == 35.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_models.py::test_correlated_event_impact_prediction_default_none tests/unit/test_models.py::test_correlated_event_accepts_impact_prediction -v`
Expected: FAIL — `CorrelatedEvent` has no field `impact_prediction`

- [ ] **Step 3: Write minimal implementation**

In `models.py`, add the import at the top (inside `TYPE_CHECKING` to avoid circular imports at runtime, with a direct import for Pydantic validation):

```python
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass
```

Then add the field to `CorrelatedEvent` (after `corroborating_sources`):

```python
    impact_prediction: Optional[trajectory.models.ImpactPrediction] = None
```

However, since Pydantic needs the actual type at validation time and `from __future__ import annotations` defers evaluation, we need a direct import. Add at the top of the file, after the existing imports:

```python
from trajectory.models import ImpactPrediction
```

Then add the field to `CorrelatedEvent`:

```python
    corroborating_sources: list[str] = Field(default_factory=list)
    impact_prediction: Optional[ImpactPrediction] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_models.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add models.py tests/unit/test_models.py
git commit -m "feat: add impact_prediction field to CorrelatedEvent"
```

---

### Task 2: Implement `ObjectTracker` — dual-write and query

**Files:**
- Create: `processing/object_tracker.py`
- Create: `tests/integration/test_object_tracker.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/integration/test_object_tracker.py`:

```python
"""Integration tests for the DynamoDB-backed ObjectTracker."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

from models import EventSource, RawEvent
from processing.object_tracker import ObjectTracker


@pytest.fixture
async def tracker(aws_credentials):
    """Create an ObjectTracker connected to moto DynamoDB."""
    with mock_aws():
        client = boto3.client("dynamodb", region_name="us-east-1")
        client.create_table(
            TableName="skyfall-events",
            KeySchema=[
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        with patch("processing.object_tracker.settings") as mock_settings:
            mock_settings.aws_region = "us-east-1"
            mock_settings.dynamodb_table_name = "skyfall-events"
            mock_settings.dynamodb_endpoint_url = ""

            t = ObjectTracker()
            await t.connect()
            yield t
            await t.close()


def _make_tip_event(
    norad_id: str = "99999",
    lat: float = 37.0,
    lon: float = -115.0,
    altitude_m: float = 120000.0,
    event_id: str | None = None,
    timestamp: datetime | None = None,
) -> RawEvent:
    """Create a RawEvent mimicking a Space-Track TIP message."""
    ts = timestamp or datetime.now(timezone.utc)
    kwargs = {
        "source": EventSource.SPACETRACK,
        "latitude": lat,
        "longitude": lon,
        "description": f"TIP: NORAD {norad_id} re-entry predicted",
        "timestamp": ts,
        "raw_payload": {
            "NORAD_CAT_ID": norad_id,
            "LAT": str(lat),
            "LON": str(lon),
            "ALTITUDE_M": str(altitude_m),
            "DECAY_EPOCH": ts.isoformat(),
        },
    }
    if event_id:
        kwargs["event_id"] = event_id
    return RawEvent(**kwargs)


@pytest.mark.asyncio
async def test_track_and_retrieve_single_observation(tracker):
    """A tracked observation should be retrievable by object ID."""
    event = _make_tip_event(norad_id="55555")
    await tracker.track_observation(event)

    obs = await tracker.get_observations("55555")
    assert len(obs) == 1
    assert obs[0].latitude == pytest.approx(37.0, abs=0.01)
    assert obs[0].longitude == pytest.approx(-115.0, abs=0.01)
    assert obs[0].altitude_m == pytest.approx(120000.0, abs=1.0)
    assert obs[0].noise_profile == "satellite"


@pytest.mark.asyncio
async def test_observations_sorted_by_timestamp(tracker):
    """Multiple observations should be returned in chronological order."""
    base = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    event1 = _make_tip_event(
        norad_id="55555", lat=37.0, altitude_m=120000.0,
        event_id="ev001", timestamp=base,
    )
    event2 = _make_tip_event(
        norad_id="55555", lat=36.8, altitude_m=100000.0,
        event_id="ev002", timestamp=base + timedelta(seconds=30),
    )
    event3 = _make_tip_event(
        norad_id="55555", lat=36.5, altitude_m=80000.0,
        event_id="ev003", timestamp=base + timedelta(seconds=60),
    )

    # Insert out of order
    await tracker.track_observation(event3)
    await tracker.track_observation(event1)
    await tracker.track_observation(event2)

    obs = await tracker.get_observations("55555")
    assert len(obs) == 3
    assert obs[0].timestamp < obs[1].timestamp < obs[2].timestamp
    assert obs[0].altitude_m > obs[1].altitude_m > obs[2].altitude_m


@pytest.mark.asyncio
async def test_different_objects_isolated(tracker):
    """Observations for different NORAD IDs must not cross-contaminate."""
    event_a = _make_tip_event(norad_id="11111", event_id="a1")
    event_b = _make_tip_event(norad_id="22222", event_id="b1")

    await tracker.track_observation(event_a)
    await tracker.track_observation(event_b)

    obs_a = await tracker.get_observations("11111")
    obs_b = await tracker.get_observations("22222")
    assert len(obs_a) == 1
    assert len(obs_b) == 1


@pytest.mark.asyncio
async def test_returns_empty_for_unknown_object(tracker):
    """Querying an object with no observations returns an empty list."""
    obs = await tracker.get_observations("00000")
    assert obs == []


@pytest.mark.asyncio
async def test_expires_at_is_integer_unix_epoch(tracker):
    """The expires_at field must be an integer for DynamoDB TTL."""
    event = _make_tip_event(norad_id="55555")
    await tracker.track_observation(event)

    # Query raw DynamoDB item to inspect expires_at
    response = await tracker._table.query(
        KeyConditionExpression="pk = :pk",
        ExpressionAttributeValues={":pk": "object#55555"},
    )
    items = response.get("Items", [])
    assert len(items) == 1
    expires_at = items[0]["expires_at"]
    # moto returns Decimal; int() should not raise
    assert int(expires_at) > 0


@pytest.mark.asyncio
async def test_sk_timestamp_format_zero_padded(tracker):
    """The SK timestamp must be zero-padded UTC ISO for lexicographic sort."""
    ts = datetime(2026, 1, 5, 3, 7, 9, 1000, tzinfo=timezone.utc)
    event = _make_tip_event(norad_id="55555", timestamp=ts, event_id="ev001")
    await tracker.track_observation(event)

    response = await tracker._table.query(
        KeyConditionExpression="pk = :pk",
        ExpressionAttributeValues={":pk": "object#55555"},
    )
    items = response.get("Items", [])
    assert len(items) == 1
    sk = items[0]["sk"]
    # SK should start with zero-padded ISO: 2026-01-05T03:07:09.001000+00:00
    assert sk.startswith("2026-01-05T03:07:09.001000")


@pytest.mark.asyncio
async def test_altitude_defaults_to_10km_when_missing(tracker):
    """When ALTITUDE_M is absent from raw_payload, default to 10000.0."""
    event = RawEvent(
        source=EventSource.SPACETRACK,
        latitude=37.0,
        longitude=-115.0,
        description="TIP event",
        raw_payload={"NORAD_CAT_ID": "55555", "LAT": "37.0", "LON": "-115.0"},
    )
    await tracker.track_observation(event)
    obs = await tracker.get_observations("55555")
    assert len(obs) == 1
    assert obs[0].altitude_m == pytest.approx(10000.0, abs=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/integration/test_object_tracker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'processing.object_tracker'`

- [ ] **Step 3: Implement `processing/object_tracker.py`**

Create `processing/object_tracker.py`:

```python
"""
Object Tracker (DynamoDB Dual-Write)
=====================================

Writes object-keyed items to the same DynamoDB table used by the
correlation engine.  This enables efficient time-series queries
for a specific NORAD_CAT_ID across any spatial distance.

DynamoDB layout (object-tracking items)
---------------------------------------
- PK: ``object#<NORAD_CAT_ID>``
- SK: ``<ISO-timestamp>#<event_id>`` (UTC, zero-padded ms for lexicographic sort)
- TTL: ``expires_at`` = now + 86400 (24 hours, independent of correlation TTL)
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

import aioboto3
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from models import RawEvent
from trajectory.models import SensorObservation

logger = logging.getLogger(__name__)

_OBJECT_TTL_SEC = 86_400  # 24 hours
_DEFAULT_ALTITUDE_M = 10_000.0  # Space-Track TIPs report at ~10 km


class ObjectTracker:
    """DynamoDB-backed tracker for per-object observation time-series."""

    def __init__(self) -> None:
        self._session: aioboto3.Session | None = None
        self._table_name = settings.dynamodb_table_name
        self._table = None
        self._resource = None

    async def connect(self) -> None:
        """Open the DynamoDB connection."""
        self._session = aioboto3.Session()
        kwargs = {"region_name": settings.aws_region}
        if settings.dynamodb_endpoint_url:
            kwargs["endpoint_url"] = settings.dynamodb_endpoint_url
        self._resource = await self._session.resource("dynamodb", **kwargs).__aenter__()
        self._table = await self._resource.Table(self._table_name)
        logger.info("ObjectTracker connected to DynamoDB table %s", self._table_name)

    async def close(self) -> None:
        """Shut down the DynamoDB connection."""
        if self._resource:
            await self._resource.__aexit__(None, None, None)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def track_observation(self, event: RawEvent) -> None:
        """Write an object-keyed item for trajectory tracking."""
        norad_id = event.raw_payload.get("NORAD_CAT_ID")
        if not norad_id:
            return

        altitude_m = _DEFAULT_ALTITUDE_M
        raw_alt = event.raw_payload.get("ALTITUDE_M")
        if raw_alt is not None:
            try:
                altitude_m = float(raw_alt)
            except (ValueError, TypeError):
                pass

        pk = f"object#{norad_id}"
        # Zero-padded UTC ISO timestamp for lexicographic sort
        ts_str = event.timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
        sk = f"{ts_str}#{event.event_id}"
        expires_at = int(time.time()) + _OBJECT_TTL_SEC

        item = {
            "pk": pk,
            "sk": sk,
            "event_id": event.event_id,
            "source": event.source.value,
            "latitude": str(event.latitude) if event.latitude is not None else None,
            "longitude": str(event.longitude) if event.longitude is not None else None,
            "altitude_m": str(altitude_m),
            "timestamp": event.timestamp.isoformat(),
            "noise_profile": "satellite",
            "description": event.description,
            "raw_payload": json.dumps(event.raw_payload),
            "expires_at": expires_at,
        }
        item = {k: v for k, v in item.items() if v is not None}

        await self._table.put_item(Item=item)
        logger.debug(
            "Tracked observation for object %s (event %s)",
            norad_id, event.event_id,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def get_observations(self, object_id: str) -> list[SensorObservation]:
        """Query all observations for an object, sorted by timestamp ASC."""
        response = await self._table.query(
            KeyConditionExpression="pk = :pk",
            ExpressionAttributeValues={":pk": f"object#{object_id}"},
            ScanIndexForward=True,  # ascending sort key order
        )

        observations = []
        for item in response.get("Items", []):
            observations.append(
                SensorObservation(
                    timestamp=item["timestamp"],
                    latitude=float(item["latitude"]),
                    longitude=float(item["longitude"]),
                    altitude_m=float(item["altitude_m"]),
                    noise_profile=item.get("noise_profile", "satellite"),
                )
            )
        return observations
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/integration/test_object_tracker.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add processing/object_tracker.py tests/integration/test_object_tracker.py
git commit -m "feat: add ObjectTracker for DynamoDB dual-write by NORAD ID"
```

---

### Task 3: Add impact prediction block to the formatter

**Files:**
- Modify: `output/formatter.py`
- Modify: `tests/unit/test_formatter.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_formatter.py`:

```python
from trajectory.models import ImpactPrediction


def _make_impact_prediction() -> ImpactPrediction:
    """Build a minimal ImpactPrediction for testing."""
    return ImpactPrediction(
        object_id="99999",
        impact_latitude=35.123,
        impact_longitude=-115.456,
        impact_altitude_m=0.0,
        time_of_impact_utc=datetime(2026, 4, 1, 15, 0, 0, tzinfo=timezone.utc),
        seconds_until_impact=120.0,
        terminal_velocity_m_s=250.5,
        covariance_position_enu=[
            [40000.0, 100.0, 0.0],
            [100.0, 35000.0, 0.0],
            [0.0, 0.0, 90000.0],
        ],
    )
```

Add the `datetime` and `timezone` imports to the top of the test file (alongside the existing `from datetime import datetime, timezone`).

Then add these test methods to the `TestFormatAlertPayload` class:

```python
    def test_impact_prediction_included_when_present(self):
        event = _make_correlated_event()
        event.impact_prediction = _make_impact_prediction()
        alert = format_alert_payload(event)["alert"]
        ip = alert["impact_prediction"]
        assert ip is not None
        assert ip["object_id"] == "99999"
        assert ip["impact_latitude"] == 35.123
        assert ip["impact_longitude"] == -115.456
        assert ip["terminal_velocity_m_s"] == 250.5
        assert ip["seconds_until_impact"] == 120.0

    def test_impact_prediction_has_confidence_ellipse(self):
        event = _make_correlated_event()
        event.impact_prediction = _make_impact_prediction()
        alert = format_alert_payload(event)["alert"]
        ip = alert["impact_prediction"]
        assert "confidence_ellipse_95pct_m" in ip
        ellipse = ip["confidence_ellipse_95pct_m"]
        assert "semi_major" in ellipse
        assert "semi_minor" in ellipse
        assert ellipse["semi_major"] > 0
        assert ellipse["semi_minor"] > 0

    def test_no_impact_prediction_when_absent(self):
        event = _make_correlated_event()
        assert event.impact_prediction is None
        alert = format_alert_payload(event)["alert"]
        assert alert.get("impact_prediction") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_formatter.py::TestFormatAlertPayload::test_impact_prediction_included_when_present tests/unit/test_formatter.py::TestFormatAlertPayload::test_impact_prediction_has_confidence_ellipse tests/unit/test_formatter.py::TestFormatAlertPayload::test_no_impact_prediction_when_absent -v`
Expected: FAIL — `impact_prediction` key not in alert dict

- [ ] **Step 3: Implement the formatter changes**

In `output/formatter.py`, add at the top:

```python
import math
```

Then at the end of `format_alert_payload`, before the `return` statement, add the impact prediction block. Replace the return block (the `return { "alert": { ... } }` starting at line 55) with:

```python
    impact_block = None
    if event.impact_prediction:
        ip = event.impact_prediction
        # Compute 95% confidence ellipse semi-axes from the 2x2 horizontal covariance
        cov_ee = ip.covariance_position_enu[0][0]
        cov_en = ip.covariance_position_enu[0][1]
        cov_nn = ip.covariance_position_enu[1][1]
        trace = cov_ee + cov_nn
        det = cov_ee * cov_nn - cov_en * cov_en
        discriminant = max((trace / 2) ** 2 - det, 0)
        lam1 = trace / 2 + math.sqrt(discriminant)
        lam2 = trace / 2 - math.sqrt(discriminant)
        semi_major = 2.0 * math.sqrt(max(lam1, 0))  # 95% ≈ 2σ
        semi_minor = 2.0 * math.sqrt(max(lam2, 0))

        impact_block = {
            "object_id": ip.object_id,
            "impact_latitude": ip.impact_latitude,
            "impact_longitude": ip.impact_longitude,
            "time_of_impact_utc": ip.time_of_impact_utc.isoformat(),
            "seconds_until_impact": ip.seconds_until_impact,
            "terminal_velocity_m_s": ip.terminal_velocity_m_s,
            "confidence_ellipse_95pct_m": {
                "semi_major": round(semi_major, 1),
                "semi_minor": round(semi_minor, 1),
            },
        }

    return {
        "alert": {
            "correlation_id": event.correlation_id,
            "severity": event.severity.value,
            "classification": event.classification.value,
            "timestamp_utc": event.timestamp.astimezone(timezone.utc).isoformat(),
            "coordinates": (
                {"lat": event.latitude, "lon": event.longitude}
                if event.latitude is not None
                else None
            ),
            "summary": event.summary,
            "corroborating_sources": event.corroborating_sources,
            "llm_analysis": llm_block,
            "contributing_events": contributing_summaries,
            "impact_prediction": impact_block,
        }
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_formatter.py -v`
Expected: All tests PASS (both old and new)

- [ ] **Step 5: Commit**

```bash
git add output/formatter.py tests/unit/test_formatter.py
git commit -m "feat: add impact prediction block to alert payload formatter"
```

---

### Task 4: Add impact prediction to Slack and Discord message blocks

**Files:**
- Modify: `output/alerter.py:38-95` (Slack) and `output/alerter.py:98-146` (Discord)

- [ ] **Step 1: Implement the Slack block addition**

In `output/alerter.py`, in the `_post_slack` function, after the existing blocks list (after the last `"type": "section"` block that prints the JSON payload, around line 89), insert a conditional impact prediction block. Replace the `slack_body` construction with:

```python
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{severity}: {payload['alert']['classification']}",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": summary,
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*Correlation ID:* `{payload['alert']['correlation_id']}`\n"
                    f"*Sources:* {', '.join(payload['alert']['corroborating_sources'])}\n"
                    f"*Coordinates:* {payload['alert'].get('coordinates', 'N/A')}\n"
                    f"*Timestamp:* {payload['alert']['timestamp_utc']}"
                ),
            },
        },
    ]

    ip = payload["alert"].get("impact_prediction")
    if ip:
        ellipse = ip.get("confidence_ellipse_95pct_m", {})
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f":dart: *Impact Prediction*\n"
                    f"*Object:* `{ip['object_id']}`\n"
                    f"*Impact:* ({ip['impact_latitude']}, {ip['impact_longitude']})\n"
                    f"*ETA:* {ip['seconds_until_impact']:.0f}s ({ip['time_of_impact_utc']})\n"
                    f"*Terminal velocity:* {ip['terminal_velocity_m_s']:.0f} m/s\n"
                    f"*95% ellipse:* {ellipse.get('semi_major', '?')}m x {ellipse.get('semi_minor', '?')}m"
                ),
            },
        })

    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"```{json.dumps(payload, indent=2, default=str)[:2900]}```",
        },
    })

    slack_body = {
        "text": f":rotating_light: *{severity} ALERT* :rotating_light:\n{summary}",
        "blocks": blocks,
    }
```

- [ ] **Step 2: Implement the Discord embed addition**

In `_post_discord`, add the impact prediction as additional fields in the embed. Replace the `discord_body` fields list with:

```python
    fields = [
        {
            "name": "Correlation ID",
            "value": f"`{payload['alert']['correlation_id']}`",
            "inline": True,
        },
        {
            "name": "Sources",
            "value": ", ".join(payload["alert"]["corroborating_sources"]),
            "inline": True,
        },
        {
            "name": "Coordinates",
            "value": str(payload["alert"].get("coordinates", "N/A")),
            "inline": True,
        },
    ]

    ip = payload["alert"].get("impact_prediction")
    if ip:
        ellipse = ip.get("confidence_ellipse_95pct_m", {})
        fields.append({
            "name": "Impact Prediction",
            "value": (
                f"**Object:** `{ip['object_id']}`\n"
                f"**Impact:** ({ip['impact_latitude']}, {ip['impact_longitude']})\n"
                f"**ETA:** {ip['seconds_until_impact']:.0f}s\n"
                f"**Terminal velocity:** {ip['terminal_velocity_m_s']:.0f} m/s\n"
                f"**95% ellipse:** {ellipse.get('semi_major', '?')}m x {ellipse.get('semi_minor', '?')}m"
            ),
        })

    fields.append({
        "name": "Full Payload",
        "value": f"```json\n{json.dumps(payload, indent=2, default=str)[:900]}\n```",
    })

    discord_body = {
        "content": f"**{severity} ALERT**",
        "embeds": [
            {
                "title": f"{severity}: {payload['alert']['classification']}",
                "description": summary,
                "color": 0xFF0000 if severity == "CRITICAL" else 0xFFA500,
                "fields": fields,
            }
        ],
    }
```

- [ ] **Step 3: Run existing alerter tests to verify no regression**

Run: `python -m pytest tests/integration/test_alerter.py -v`
Expected: All existing tests PASS

- [ ] **Step 4: Commit**

```bash
git add output/alerter.py
git commit -m "feat: add impact prediction details to Slack and Discord alerts"
```

---

### Task 5: Wire tracker and predictor into the triage loop

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add imports**

At the top of `main.py`, add:

```python
from processing.object_tracker import ObjectTracker
from trajectory.models import TrajectoryRequest
from trajectory.predictor import DebrisTrajectoryPredictor
```

- [ ] **Step 2: Add tracker lifecycle to `main()`**

In the `main()` function, after `engine = CorrelationEngine()` and `await engine.connect()`, add:

```python
    tracker = ObjectTracker()
    await tracker.connect()
```

Pass `tracker` to `triage_loop`:

```python
        asyncio.create_task(triage_loop(raw_queue, alert_queue, engine, tracker), name="triage"),
```

In the shutdown section, after `await engine.close()`, add:

```python
    await tracker.close()
```

- [ ] **Step 3: Update `triage_loop` signature and add tracking logic**

Update the function signature:

```python
async def triage_loop(
    raw_queue: asyncio.Queue[RawEvent],
    alert_queue: asyncio.Queue[CorrelatedEvent],
    engine: CorrelationEngine,
    tracker: ObjectTracker,
) -> None:
```

Add the import at the top of the file (already done in step 1). Then, inside the `try` block in `triage_loop`, after the existing `engine.try_correlate(...)` calls (both the `needs_llm` branch and the `else` branch), add the trajectory tracking logic. Place this before the `except Exception:` line, at the same indentation level as the `if needs_llm:` block:

```python
            # ── Trajectory tracking for objects with NORAD_CAT_ID ────────
            norad_id = event.raw_payload.get("NORAD_CAT_ID")
            if norad_id:
                await tracker.track_observation(event)
                observations = await tracker.get_observations(str(norad_id))

                if len(observations) >= 2:
                    logger.info(
                        "Triggering trajectory prediction for NORAD %s (%d observations)",
                        norad_id, len(observations),
                    )
                    request = TrajectoryRequest(
                        object_id=str(norad_id),
                        observations=observations,
                    )
                    predictor = DebrisTrajectoryPredictor()
                    prediction = await asyncio.to_thread(predictor.predict, request)

                    trajectory_event = CorrelatedEvent(
                        severity=EventSeverity.CRITICAL,
                        classification=EventClassification.DEBRIS_REENTRY,
                        latitude=prediction.impact_latitude,
                        longitude=prediction.impact_longitude,
                        contributing_events=[event],
                        summary=(
                            f"TRAJECTORY PREDICTION: NORAD {norad_id} "
                            f"impact at ({prediction.impact_latitude}, {prediction.impact_longitude}) "
                            f"in {prediction.seconds_until_impact:.0f}s, "
                            f"terminal velocity {prediction.terminal_velocity_m_s:.0f} m/s"
                        ),
                        corroborating_sources=["spacetrack"],
                        impact_prediction=prediction,
                    )
                    await alert_queue.put(trajectory_event)
                    logger.info(
                        "Trajectory prediction queued for NORAD %s: impact at (%.4f, %.4f)",
                        norad_id, prediction.impact_latitude, prediction.impact_longitude,
                    )
```

Also add `EventClassification` to the imports from `models` if not already present (check line 69 — it imports `CorrelatedEvent, EventSource, RawEvent` but not `EventClassification` or `EventSeverity`):

```python
from models import CorrelatedEvent, EventClassification, EventSeverity, EventSource, RawEvent
```

- [ ] **Step 4: Run existing tests to verify no regression**

Run: `python -m pytest tests/ -v --timeout=30`
Expected: All existing tests PASS

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat: wire ObjectTracker and DebrisTrajectoryPredictor into triage loop"
```

---

### Task 6: Create the mock data injector script

**Files:**
- Create: `scripts/inject_mock_trajectory.py`

- [ ] **Step 1: Create the scripts directory**

```bash
mkdir -p scripts
```

- [ ] **Step 2: Write the script**

Create `scripts/inject_mock_trajectory.py`:

```python
#!/usr/bin/env python3
"""
Mock Trajectory Injector
=========================

Generates 4 synthetic TIP observations for a fake NORAD_CAT_ID and
POSTs them to the local /api/v1/test-event endpoint with a 2-second
delay between each.

Usage::

    python scripts/inject_mock_trajectory.py

Requires the local stack to be running (docker compose up).
"""

import random
import sys
import time
from datetime import datetime, timedelta, timezone

import httpx

BASE_URL = "http://localhost:8000"
ENDPOINT = f"{BASE_URL}/api/v1/test-event"
NORAD_CAT_ID = "99999"

# Ground truth trajectory: re-entry over US Southwest
# Start: 37.0°N, 115.0°W, 120km altitude, heading SSW ~2 km/s
LAT0, LON0, ALT0 = 37.0, -115.0, 120_000.0
V_LAT = -0.006   # deg/s southward
V_LON = -0.003   # deg/s westward
V_ALT = -800.0   # m/s downward

OFFSETS_SEC = [0, 25, 55, 90]
NOISE_PROFILES = ["satellite", "thermal", "social_media", "thermal"]

# Noise standard deviations per profile (lat_deg, lon_deg, alt_m)
NOISE_SIGMA = {
    "satellite":    (0.002, 0.002, 400),
    "thermal":      (0.008, 0.008, 1500),
    "social_media": (0.020, 0.020, 4000),
}


def generate_observations() -> list[dict]:
    """Generate 4 noisy synthetic TIP observations."""
    random.seed(42)
    base_time = datetime.now(timezone.utc) - timedelta(seconds=max(OFFSETS_SEC))
    observations = []

    for i, t in enumerate(OFFSETS_SEC):
        true_lat = LAT0 + V_LAT * t
        true_lon = LON0 + V_LON * t
        true_alt = ALT0 + V_ALT * t + 0.5 * (-9.81) * t ** 2
        true_alt = max(true_alt, 0)

        profile = NOISE_PROFILES[i]
        sigma = NOISE_SIGMA[profile]
        noisy_lat = round(true_lat + random.gauss(0, sigma[0]), 6)
        noisy_lon = round(true_lon + random.gauss(0, sigma[1]), 6)
        noisy_alt = round(max(true_alt + random.gauss(0, sigma[2]), 500.0), 1)

        ts = base_time + timedelta(seconds=t)
        observations.append({
            "source": "spacetrack",
            "latitude": noisy_lat,
            "longitude": noisy_lon,
            "description": (
                f"TIP: NORAD {NORAD_CAT_ID} re-entry predicted "
                f"{ts.isoformat()} UTC @ ({noisy_lat}, {noisy_lon})"
            ),
            "NORAD_CAT_ID": NORAD_CAT_ID,
            "LAT": str(noisy_lat),
            "LON": str(noisy_lon),
            "ALTITUDE_M": str(noisy_alt),
            "DECAY_EPOCH": ts.isoformat(),
            "WINDOW": "5",
            "HIGH_INTEREST": "Y",
            "timestamp": ts.isoformat(),
        })

    return observations


def main() -> None:
    print("=" * 60)
    print("  MOCK TRAJECTORY INJECTOR")
    print("=" * 60)
    print(f"\n  Target:   {ENDPOINT}")
    print(f"  Object:   NORAD {NORAD_CAT_ID}")
    print(f"  Points:   {len(OFFSETS_SEC)} observations")
    print(f"  Delay:    2s between each POST\n")

    # Check health first
    try:
        health = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        print(f"  Health check: {health.json()}\n")
    except httpx.ConnectError:
        print("  ERROR: Cannot connect to local server.")
        print("  Make sure 'docker compose up' is running.\n")
        sys.exit(1)

    observations = generate_observations()

    for i, obs in enumerate(observations):
        print(f"── Observation {i + 1}/{len(observations)} ──")
        print(f"  lat={obs['latitude']}, lon={obs['longitude']}, alt={obs['ALTITUDE_M']}m")
        print(f"  timestamp={obs['timestamp']}")
        print(f"  POSTing to {ENDPOINT}...")

        resp = httpx.post(ENDPOINT, json=obs, timeout=10.0)
        print(f"  Response: {resp.status_code} {resp.json()}")

        if i < len(observations) - 1:
            print(f"  Waiting 2 seconds...\n")
            time.sleep(2)
        else:
            print()

    print("=" * 60)
    print("  All observations injected.")
    print("  Check your Slack/Discord for impact prediction alerts.")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Make the script executable**

```bash
chmod +x scripts/inject_mock_trajectory.py
```

- [ ] **Step 4: Verify the script parses without errors**

Run: `python -c "import scripts.inject_mock_trajectory"` will fail because of the directory structure. Instead:

```bash
python -c "exec(open('scripts/inject_mock_trajectory.py').read().split(\"if __name__\")[0]); print('OK: script parses, generated', len(generate_observations()), 'observations')"
```

Expected: `OK: script parses, generated 4 observations`

- [ ] **Step 5: Commit**

```bash
git add scripts/inject_mock_trajectory.py
git commit -m "feat: add mock trajectory injector script for local E2E testing"
```

---

### Task 7: Integration test — full trajectory trigger path

**Files:**
- Create: `tests/integration/test_trajectory_trigger.py`

This test verifies the complete flow: dual-write → query → predictor → alert, matching what `triage_loop` does but in an isolated test.

- [ ] **Step 1: Write the integration test**

Create `tests/integration/test_trajectory_trigger.py`:

```python
"""Integration test: object tracking → trajectory prediction → alert payload."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

from models import (
    CorrelatedEvent,
    EventClassification,
    EventSeverity,
    EventSource,
    RawEvent,
)
from output.formatter import format_alert_payload
from processing.object_tracker import ObjectTracker
from trajectory.models import TrajectoryRequest
from trajectory.predictor import DebrisTrajectoryPredictor


@pytest.fixture
async def tracker(aws_credentials):
    """Create an ObjectTracker connected to moto DynamoDB."""
    with mock_aws():
        client = boto3.client("dynamodb", region_name="us-east-1")
        client.create_table(
            TableName="skyfall-events",
            KeySchema=[
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        with patch("processing.object_tracker.settings") as mock_settings:
            mock_settings.aws_region = "us-east-1"
            mock_settings.dynamodb_table_name = "skyfall-events"
            mock_settings.dynamodb_endpoint_url = ""

            t = ObjectTracker()
            await t.connect()
            yield t
            await t.close()


def _make_tip_event(
    norad_id: str,
    lat: float,
    lon: float,
    altitude_m: float,
    event_id: str,
    timestamp: datetime,
) -> RawEvent:
    return RawEvent(
        event_id=event_id,
        source=EventSource.SPACETRACK,
        latitude=lat,
        longitude=lon,
        description=f"TIP: NORAD {norad_id}",
        timestamp=timestamp,
        raw_payload={
            "NORAD_CAT_ID": norad_id,
            "LAT": str(lat),
            "LON": str(lon),
            "ALTITUDE_M": str(altitude_m),
            "DECAY_EPOCH": timestamp.isoformat(),
        },
    )


@pytest.mark.asyncio
async def test_two_observations_trigger_prediction(tracker):
    """Two TIP observations for the same NORAD ID produce a valid ImpactPrediction."""
    base = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    ev1 = _make_tip_event("99999", 37.0, -115.0, 120000.0, "e1", base)
    ev2 = _make_tip_event("99999", 36.85, -115.075, 100000.0, "e2", base + timedelta(seconds=25))

    await tracker.track_observation(ev1)
    await tracker.track_observation(ev2)

    observations = await tracker.get_observations("99999")
    assert len(observations) == 2

    request = TrajectoryRequest(object_id="99999", observations=observations)
    predictor = DebrisTrajectoryPredictor()
    prediction = await asyncio.to_thread(predictor.predict, request)

    assert prediction.object_id == "99999"
    assert prediction.seconds_until_impact > 0
    assert prediction.terminal_velocity_m_s > 0
    assert prediction.impact_altitude_m == pytest.approx(0.0, abs=500.0)


@pytest.mark.asyncio
async def test_prediction_wraps_in_correlated_event(tracker):
    """The prediction output integrates cleanly into CorrelatedEvent and formatter."""
    base = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    ev1 = _make_tip_event("99999", 37.0, -115.0, 120000.0, "e1", base)
    ev2 = _make_tip_event("99999", 36.85, -115.075, 100000.0, "e2", base + timedelta(seconds=25))

    await tracker.track_observation(ev1)
    await tracker.track_observation(ev2)

    observations = await tracker.get_observations("99999")
    request = TrajectoryRequest(object_id="99999", observations=observations)
    predictor = DebrisTrajectoryPredictor()
    prediction = await asyncio.to_thread(predictor.predict, request)

    correlated = CorrelatedEvent(
        severity=EventSeverity.CRITICAL,
        classification=EventClassification.DEBRIS_REENTRY,
        latitude=prediction.impact_latitude,
        longitude=prediction.impact_longitude,
        contributing_events=[ev2],
        summary=f"Impact at ({prediction.impact_latitude}, {prediction.impact_longitude})",
        corroborating_sources=["spacetrack"],
        impact_prediction=prediction,
    )

    payload = format_alert_payload(correlated)
    alert = payload["alert"]
    assert alert["severity"] == "CRITICAL"
    assert alert["impact_prediction"] is not None
    assert alert["impact_prediction"]["object_id"] == "99999"
    assert alert["impact_prediction"]["confidence_ellipse_95pct_m"]["semi_major"] > 0
```

- [ ] **Step 2: Run the integration test**

Run: `python -m pytest tests/integration/test_trajectory_trigger.py -v`
Expected: Both tests PASS

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest tests/ -v --timeout=30`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_trajectory_trigger.py
git commit -m "test: add integration test for full trajectory trigger path"
```
