"""
Webhook Receiver (FastAPI)
===========================

Exposes secured POST endpoints for external event ingestion:

    POST /api/v1/emergency   — generic emergency events
    POST /api/v1/siren       — Pikud HaOref siren alerts (structured)
    POST /api/v1/test-event  — synthetic events for testing
    GET  /health             — readiness probe (unauthenticated)

All POST endpoints require ``X-API-Key`` header when ``API_KEY`` is configured.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from config import settings
from models import EventSource, RawEvent
from ingestion.siren_listener import (
    SirenEvent,
    ALERT_CATEGORIES,
    DRILL_CATEGORIES,
    CATEGORY_LABELS,
    WATCH_ZONES,
    ZONE_COORDINATES,
    _TITLE_EVENT_ENDED,
)
from ingestion.adsb_detector import AircraftDetector, AircraftState

logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Skyfall – Event Ingestion API",
    version="0.2.0",
)

_event_queue: asyncio.Queue[RawEvent] | None = None
_siren_callback = None


def _parse_watch_hex_codes(raw: str) -> set[str]:
    return {h.strip().lower() for h in raw.split(",") if h.strip()}

_adsb_detector = AircraftDetector(
    watch_hex_codes=_parse_watch_hex_codes(settings.adsb_watch_hex_codes),
)


def set_event_queue(q: asyncio.Queue[RawEvent]) -> None:
    """Inject the shared event queue (called once at startup)."""
    global _event_queue
    _event_queue = q


def set_siren_callback(cb) -> None:
    """Inject the siren callback for trajectory correlation."""
    global _siren_callback
    _siren_callback = cb


# ── API Key Auth ──────────────────────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Depends(_api_key_header)):
    """Verify the API key if one is configured."""
    if not settings.api_key:
        return  # Auth disabled
    if api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Request schemas ───────────────────────────────────────────────────────────


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


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/api/v1/emergency", status_code=202, dependencies=[Depends(verify_api_key)])
async def receive_emergency(payload: EmergencyPayload, request: Request):
    """Accept an emergency broadcast payload and push into the pipeline."""
    if _event_queue is None:
        raise HTTPException(status_code=503, detail="Event queue not initialized yet.")

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


@app.post("/api/v1/siren", status_code=202, dependencies=[Depends(verify_api_key)])
async def receive_siren(request: Request):
    """
    Accept a raw Pikud HaOref alert JSON (pushed from Israeli proxy).
    Runs the same logic as poll_sirens: category filter, zone matching,
    coordinate lookup, and siren callback for trajectory correlation.
    """
    if _event_queue is None:
        raise HTTPException(status_code=503, detail="Event queue not initialized yet.")

    body = await request.json()

    # Accept single alert or array
    alerts = body if isinstance(body, list) else [body]
    accepted = []

    for alert in alerts:
        alert_id = str(alert.get("id", ""))
        if not alert_id:
            continue

        title = alert.get("title", "")
        zones = alert.get("data", [])
        category = str(alert.get("cat", ""))

        # Category filter — same logic as poll_sirens
        if category in DRILL_CATEGORIES:
            continue
        if category and category not in ALERT_CATEGORIES:
            continue

        # Zone matching
        matched = [z for z in zones if z in WATCH_ZONES]
        if not matched:
            # Still ingest for correlation even if not a watch zone,
            # but don't trigger siren callback
            pass

        is_active = title != _TITLE_EVENT_ENDED

        # Coordinates from matched watch zones
        if matched:
            lats = [ZONE_COORDINATES[z][0] for z in matched if z in ZONE_COORDINATES]
            lons = [ZONE_COORDINATES[z][1] for z in matched if z in ZONE_COORDINATES]
            lat = sum(lats) / len(lats) if lats else None
            lon = sum(lons) / len(lons) if lons else None
        else:
            lat, lon = None, None

        cat_label = CATEGORY_LABELS.get(category, category or "unknown")
        status = "ACTIVE SIREN" if is_active else "EVENT ENDED"
        zone_str = ", ".join(matched) if matched else ", ".join(zones[:3])

        print(
            f"[SIREN] {cat_label} {status} in {zone_str} (alert {alert_id})",
            flush=True,
        )

        # Push to pipeline
        raw_event = RawEvent(
            source=EventSource.SIREN,
            latitude=lat,
            longitude=lon,
            raw_payload=alert,
            description=(
                f"[SIREN {status}] {cat_label} | "
                f"Zones: {zone_str} | Title: {title}"
            ),
        )
        await _event_queue.put(raw_event)

        # Trigger siren callback for trajectory correlation
        if matched and _siren_callback and is_active:
            siren_event = SirenEvent(
                alert_id=alert_id,
                title=title,
                category=category,
                zones=zones,
                matched_watch_zones=matched,
                description=title,
                is_active=is_active,
            )
            await _siren_callback(siren_event)

        accepted.append({"alert_id": alert_id, "category": cat_label, "zones": zone_str})

    return {"status": "accepted", "alerts": accepted, "count": len(accepted)}


@app.post("/api/v1/adsb", status_code=202, dependencies=[Depends(verify_api_key)])
async def receive_adsb(request: Request):
    """
    Accept a batch of aircraft states from an ADS-B proxy and run
    anomaly detection (emergency squawk, watch hex, heading reroute).
    Only anomalies are pushed into the pipeline.
    """
    if _event_queue is None:
        raise HTTPException(status_code=503, detail="Event queue not initialized yet.")

    body = await request.json()
    raw_aircraft = body.get("aircraft", [])

    states = []
    for ac in raw_aircraft:
        try:
            states.append(AircraftState(**ac))
        except Exception:
            logger.debug("Skipping invalid aircraft state: %s", ac)
            continue

    events = _adsb_detector.process_batch(states)

    for event in events:
        await _event_queue.put(event)

    return {"status": "accepted", "total": len(states), "anomalies": len(events)}


@app.post("/api/v1/test-event", status_code=202, dependencies=[Depends(verify_api_key)])
async def inject_test_event(request: Request):
    """Inject a synthetic sensor event for testing."""
    if _event_queue is None:
        raise HTTPException(status_code=503, detail="Event queue not initialized yet.")

    body = await request.json()
    source_map = {s.value: s for s in EventSource}
    source = source_map.get(body.get("source", ""), EventSource.FIRMS)

    event = RawEvent(
        source=source,
        latitude=body.get("latitude"),
        longitude=body.get("longitude"),
        raw_payload=body,
        description=body.get("description", "Test event"),
    )

    await _event_queue.put(event)
    return {"status": "accepted", "event_id": event.event_id, "source": source.value}


@app.get("/health")
async def health():
    """Readiness probe for load balancers / container orchestrators."""
    checks = {"queue": _event_queue is not None}
    all_ok = all(checks.values())
    return {
        "status": "ok" if all_ok else "degraded",
        "checks": checks,
    }
