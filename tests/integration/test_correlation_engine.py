"""Integration tests for the DynamoDB-backed CorrelationEngine."""

from __future__ import annotations

from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

from models import EventSeverity, EventSource, LLMParsedEvent
from processing.correlation_engine import CorrelationEngine
from tests.conftest import make_raw_event


@pytest.fixture
async def engine(aws_credentials):
    """Create a CorrelationEngine connected to moto DynamoDB."""
    with mock_aws():
        # Create table via sync boto3
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

        with patch("processing.correlation_engine.settings") as mock_settings:
            mock_settings.aws_region = "us-east-1"
            mock_settings.dynamodb_table_name = "skyfall-events"
            mock_settings.dynamodb_endpoint_url = ""
            mock_settings.correlation_window_sec = 300
            mock_settings.min_confidence_score = 6
            mock_settings.geohash_precision = 4

            eng = CorrelationEngine()
            await eng.connect()
            yield eng
            await eng.close()


def _high_confidence_llm(classification: str = "debris_reentry") -> LLMParsedEvent:
    """Return an LLMParsedEvent that passes correlation gates."""
    return LLMParsedEvent(
        is_valid_anomaly=True,
        approximate_origin="Houston, TX",
        debris_trajectory_or_blast_radius="10km NE",
        event_classification=classification,
        confidence_score=8,
    )


# ── Ingestion tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_stores_event(engine):
    """Ingesting an event should make it queryable in its geohash cell."""
    event = make_raw_event(source=EventSource.FIRMS, lat=29.76, lon=-95.37)
    await engine.ingest(event)

    from processing.geohash import encode

    gh = encode(29.76, -95.37, precision=4)
    pk = f"geohash#{gh}"
    items = await engine._query_cell(pk)

    assert len(items) == 1
    assert items[0]["event_id"] == event.event_id


@pytest.mark.asyncio
async def test_ingest_sets_ttl(engine):
    """Ingested events must have an expires_at attribute for DynamoDB TTL."""
    event = make_raw_event()
    await engine.ingest(event)

    from processing.geohash import encode

    gh = encode(29.76, -95.37, precision=4)
    pk = f"geohash#{gh}"
    items = await engine._query_cell(pk)

    assert len(items) == 1
    assert "expires_at" in items[0]
    # expires_at should be a number (DynamoDB stores it as Decimal)
    assert int(items[0]["expires_at"]) > 0


@pytest.mark.asyncio
async def test_ingest_without_coordinates(engine):
    """Events without lat/lon should be stored under geohash#none."""
    event = make_raw_event(lat=None, lon=None)
    await engine.ingest(event)

    items = await engine._query_cell("geohash#none")
    assert len(items) == 1
    assert items[0]["event_id"] == event.event_id


# ── Correlation tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_two_sources_same_area_elevates_to_critical(engine):
    """FIRMS + ADSB in the same geohash cell should produce CRITICAL."""
    firms_event = make_raw_event(
        source=EventSource.FIRMS, lat=29.76, lon=-95.37, event_id="firms001"
    )
    adsb_event = make_raw_event(
        source=EventSource.ADSB, lat=29.76, lon=-95.37, event_id="adsb001"
    )
    await engine.ingest(firms_event)
    await engine.ingest(adsb_event)

    result = await engine.try_correlate(firms_event, _high_confidence_llm())
    assert result is not None
    assert result.severity == EventSeverity.CRITICAL
    assert "adsb" in result.corroborating_sources
    assert "firms" in result.corroborating_sources


@pytest.mark.asyncio
async def test_single_source_does_not_elevate(engine):
    """Two events from the same source should NOT produce CRITICAL."""
    event1 = make_raw_event(
        source=EventSource.FIRMS, lat=29.76, lon=-95.37, event_id="firms001"
    )
    event2 = make_raw_event(
        source=EventSource.FIRMS, lat=29.76, lon=-95.37, event_id="firms002"
    )
    await engine.ingest(event1)
    await engine.ingest(event2)

    result = await engine.try_correlate(event1, _high_confidence_llm())
    assert result is not None
    assert result.severity != EventSeverity.CRITICAL


@pytest.mark.asyncio
async def test_low_confidence_skipped(engine):
    """Events below the confidence threshold should return None."""
    event = make_raw_event()
    await engine.ingest(event)

    low_conf = LLMParsedEvent(
        is_valid_anomaly=True,
        approximate_origin="Somewhere",
        event_classification="unknown",
        confidence_score=3,
    )
    result = await engine.try_correlate(event, low_conf)
    assert result is None


@pytest.mark.asyncio
async def test_invalid_anomaly_skipped(engine):
    """Events flagged as not a valid anomaly should return None."""
    event = make_raw_event()
    await engine.ingest(event)

    not_anomaly = LLMParsedEvent(
        is_valid_anomaly=False,
        approximate_origin="Somewhere",
        event_classification="unknown",
        confidence_score=8,
    )
    result = await engine.try_correlate(event, not_anomaly)
    assert result is None


@pytest.mark.asyncio
async def test_distant_events_not_correlated(engine):
    """Houston and London events should NOT correlate (different geohash cells)."""
    houston = make_raw_event(
        source=EventSource.FIRMS, lat=29.76, lon=-95.37, event_id="houston01"
    )
    london = make_raw_event(
        source=EventSource.ADSB, lat=51.51, lon=-0.13, event_id="london01"
    )
    await engine.ingest(houston)
    await engine.ingest(london)

    result = await engine.try_correlate(houston, _high_confidence_llm())
    assert result is not None
    assert result.severity != EventSeverity.CRITICAL
    assert len(result.corroborating_sources) == 1
