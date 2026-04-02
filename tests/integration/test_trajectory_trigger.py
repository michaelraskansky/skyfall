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
