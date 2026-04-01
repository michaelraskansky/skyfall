"""Integration tests for the emergency webhook FastAPI endpoint."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from ingestion.emergency_webhook import app, set_event_queue
from models import EventSource, RawEvent


@pytest.fixture
def webhook_client():
    queue: asyncio.Queue[RawEvent] = asyncio.Queue()
    set_event_queue(queue)
    client = TestClient(app)
    yield client, queue
    set_event_queue(None)


VALID_PAYLOAD = {
    "source_system": "NWS",
    "event_type": "explosion",
    "latitude": 29.76,
    "longitude": -95.37,
    "description": "Large explosion reported near refinery",
}


class TestEmergencyWebhook:
    """Tests for POST /api/v1/emergency and GET /health."""

    def test_valid_payload_accepted(self, webhook_client):
        client, _queue = webhook_client
        resp = client.post("/api/v1/emergency", json=VALID_PAYLOAD)
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "accepted"
        assert "event_id" in body

    def test_event_pushed_to_queue(self, webhook_client):
        client, queue = webhook_client
        client.post("/api/v1/emergency", json=VALID_PAYLOAD)
        assert not queue.empty()
        event = queue.get_nowait()
        assert event.source == EventSource.EMERGENCY_WEBHOOK

    def test_missing_required_field_rejected(self, webhook_client):
        client, _queue = webhook_client
        # Missing both source_system and event_type
        resp = client.post("/api/v1/emergency", json={"description": "incomplete"})
        assert resp.status_code == 422

    def test_queue_not_initialized_returns_503(self):
        set_event_queue(None)
        client = TestClient(app)
        resp = client.post("/api/v1/emergency", json=VALID_PAYLOAD)
        assert resp.status_code == 503

    def test_health_endpoint(self, webhook_client):
        client, _queue = webhook_client
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["checks"]["queue"] is True
