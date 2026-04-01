"""
Emergency Webhook Receiver (FastAPI)
=====================================

A lightweight FastAPI application that exposes a single POST endpoint:

    POST /api/v1/emergency

Local emergency / weather broadcast systems can push JSON payloads here.
Each incoming payload is validated, wrapped in a ``RawEvent``, and placed
onto the shared ``asyncio.Queue`` for triage.

The app is designed to be run by ``uvicorn`` inside the main orchestrator,
sharing the same event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from models import EventSource, RawEvent

logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Debris Tracker – Emergency Webhook",
    version="0.1.0",
)

# This will be set at startup by the orchestrator so that the endpoint
# can push events into the shared pipeline.
_event_queue: asyncio.Queue[RawEvent] | None = None


def set_event_queue(q: asyncio.Queue[RawEvent]) -> None:
    """Inject the shared event queue (called once at startup)."""
    global _event_queue
    _event_queue = q


# ── Request schema ────────────────────────────────────────────────────────────


class EmergencyPayload(BaseModel):
    """Schema for incoming emergency webhook POSTs."""

    source_system: str = Field(
        ..., description="Name of the upstream system (e.g. 'NWS', 'local_ems')"
    )
    event_type: str = Field(
        ..., description="Free-text event type (e.g. 'explosion', 'hazmat')"
    )
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    description: str = ""
    metadata: dict = Field(default_factory=dict)


# ── Endpoint ──────────────────────────────────────────────────────────────────


@app.post("/api/v1/emergency", status_code=202)
async def receive_emergency(payload: EmergencyPayload, request: Request):
    """
    Accept an emergency broadcast payload, wrap it in a RawEvent, and
    push it into the processing pipeline.
    """
    if _event_queue is None:
        raise HTTPException(
            status_code=503, detail="Event queue not initialized yet."
        )

    event = RawEvent(
        source=EventSource.EMERGENCY_WEBHOOK,
        latitude=payload.latitude,
        longitude=payload.longitude,
        raw_payload=payload.model_dump(),
        description=(
            f"[{payload.source_system}] {payload.event_type}: "
            f"{payload.description}"
        ),
    )

    await _event_queue.put(event)
    logger.info("Emergency webhook received: %s", event.description)

    return {"status": "accepted", "event_id": event.event_id}


@app.get("/health")
async def health():
    """Simple liveness probe for load balancers / container orchestrators."""
    return {"status": "ok"}
