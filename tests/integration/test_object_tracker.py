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
