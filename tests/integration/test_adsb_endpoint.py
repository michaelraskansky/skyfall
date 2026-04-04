"""Integration tests for POST /api/v1/adsb push endpoint."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from ingestion.emergency_webhook import app, set_event_queue
from models import EventSource, RawEvent


@pytest.fixture
def adsb_client():
    queue: asyncio.Queue[RawEvent] = asyncio.Queue()
    set_event_queue(queue)
    client = TestClient(app)
    yield client, queue
    set_event_queue(None)


def _aircraft(hex="aabbcc", **overrides):
    """Factory for a single aircraft state dict."""
    base = {
        "hex": hex,
        "callsign": "TEST1",
        "lat": 32.0,
        "lon": 34.8,
        "track": 90.0,
        "alt_m": 10000,
        "on_ground": False,
    }
    base.update(overrides)
    return base


class TestAdsbEndpoint:
    """Tests for POST /api/v1/adsb."""

    def test_empty_batch_accepted(self, adsb_client):
        client, queue = adsb_client
        resp = client.post("/api/v1/adsb", json={"aircraft": []})
        assert resp.status_code == 202
        body = resp.json()
        assert body["total"] == 0
        assert body["anomalies"] == 0

    def test_normal_aircraft_no_anomaly(self, adsb_client):
        client, queue = adsb_client
        resp = client.post("/api/v1/adsb", json={"aircraft": [_aircraft()]})
        assert resp.status_code == 202
        assert resp.json()["anomalies"] == 0
        assert queue.empty()

    def test_squawk_7700_emits_event(self, adsb_client):
        client, queue = adsb_client
        resp = client.post("/api/v1/adsb", json={
            "aircraft": [_aircraft(squawk="7700")],
        })
        assert resp.status_code == 202
        assert resp.json()["anomalies"] == 1
        event = queue.get_nowait()
        assert event.source == EventSource.ADSB
        assert "MAYDAY" in event.description

    def test_watch_hex_emits_event(self, adsb_client):
        client, queue = adsb_client
        from ingestion import emergency_webhook
        emergency_webhook._adsb_detector._watch_hexes = {"aabbcc"}
        try:
            resp = client.post("/api/v1/adsb", json={
                "aircraft": [_aircraft(hex="aabbcc")],
            })
            assert resp.json()["anomalies"] == 1
            event = queue.get_nowait()
            assert "aabbcc" in event.description
        finally:
            emergency_webhook._adsb_detector._watch_hexes = set()

    def test_heading_reroute_emits_event(self, adsb_client):
        client, queue = adsb_client
        # First batch: establish heading
        client.post("/api/v1/adsb", json={
            "aircraft": [_aircraft(track=90.0)],
        })
        assert queue.empty()
        # Second batch: 90° change
        resp = client.post("/api/v1/adsb", json={
            "aircraft": [_aircraft(track=180.0)],
        })
        assert resp.json()["anomalies"] == 1
        event = queue.get_nowait()
        assert "rerouted" in event.description

    def test_on_ground_skipped(self, adsb_client):
        client, queue = adsb_client
        resp = client.post("/api/v1/adsb", json={
            "aircraft": [_aircraft(squawk="7700", on_ground=True)],
        })
        assert resp.json()["anomalies"] == 0
        assert queue.empty()

    def test_queue_not_initialized_returns_503(self):
        set_event_queue(None)
        client = TestClient(app)
        resp = client.post("/api/v1/adsb", json={"aircraft": []})
        assert resp.status_code == 503

    def test_response_format(self, adsb_client):
        client, queue = adsb_client
        resp = client.post("/api/v1/adsb", json={
            "aircraft": [
                _aircraft(hex="a1", squawk="7700"),
                _aircraft(hex="a2"),
                _aircraft(hex="a3", squawk="7500"),
            ],
        })
        body = resp.json()
        assert body["status"] == "accepted"
        assert body["total"] == 3
        assert body["anomalies"] == 2
