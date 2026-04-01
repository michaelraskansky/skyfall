"""
Stateful Correlation Engine
============================

The correlation engine is the "brain" of the triage layer.  It maintains
a sliding window of recent sensor events (backed by Redis) and decides
when independent signals corroborate each other.

Algorithm
---------
1. Every ``RawEvent`` is stored in Redis as a geo-indexed entry with a TTL
   equal to the correlation window (default 5 minutes).
2. When the LLM parser flags an event as a valid anomaly with sufficient
   confidence, the engine queries Redis for *other* sensor events within
   a configurable radius (~50 km) and time window.
3. If **two or more distinct sensor sources** agree, the event is elevated
   to ``CRITICAL`` and a ``CorrelatedEvent`` is emitted.

Redis data layout
-----------------
- ``events:geo``  – a Redis GEO set mapping event IDs to lat/lon.
- ``events:{id}`` – a Redis HASH holding the serialised ``RawEvent`` JSON,
                     with a TTL of ``CORRELATION_WINDOW_SEC``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import redis.asyncio as aioredis

if TYPE_CHECKING:
    from asyncio import Queue

from config import settings
from models import (
    CorrelatedEvent,
    EventClassification,
    EventSeverity,
    EventSource,
    LLMParsedEvent,
    RawEvent,
)

logger = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────
# Radius in kilometres used for geo-proximity matching.
CORRELATION_RADIUS_KM: float = 50.0


class CorrelationEngine:
    """
    Stateful engine backed by Redis that cross-references sensor events.

    Usage::

        engine = CorrelationEngine()
        await engine.connect()

        # Called by every ingestion sensor:
        await engine.ingest(raw_event)

        # Called after LLM triage returns a high-confidence result:
        correlated = await engine.try_correlate(raw_event, llm_result)
    """

    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None
        self._window = settings.correlation_window_sec
        self._min_confidence = settings.min_confidence_score

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the Redis connection pool."""
        self._redis = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
        logger.info("Correlation engine connected to Redis at %s", settings.redis_url)

    async def close(self) -> None:
        """Shut down the Redis connection pool."""
        if self._redis:
            await self._redis.close()

    # ── Ingestion ─────────────────────────────────────────────────────────

    async def ingest(self, event: RawEvent) -> None:
        """
        Store a raw sensor event in Redis with geo-index and TTL.

        Every event is recorded regardless of source so that later
        correlation queries can find neighbouring signals.
        """
        assert self._redis is not None, "Call connect() first"

        key = f"events:{event.event_id}"
        payload = event.model_dump_json()

        # Store the event hash with a TTL.
        await self._redis.set(key, payload, ex=self._window)

        # Geo-index if coordinates are available.
        if event.latitude is not None and event.longitude is not None:
            await self._redis.geoadd(
                "events:geo",
                (event.longitude, event.latitude, event.event_id),
            )
            # The GEO set itself has no TTL, so we rely on key expiry for
            # the detail hash.  Stale geo entries are harmless – the
            # correlation step checks for the detail key before counting.

        logger.debug("Ingested event %s from %s", event.event_id, event.source.value)

    # ── Correlation ───────────────────────────────────────────────────────

    async def try_correlate(
        self,
        trigger_event: RawEvent,
        llm_result: LLMParsedEvent,
        output_queue: Queue[CorrelatedEvent] | None = None,
    ) -> CorrelatedEvent | None:
        """
        Attempt to correlate *trigger_event* with other recent sensor data.

        Returns a ``CorrelatedEvent`` (possibly CRITICAL) if correlation
        succeeds, or ``None`` if the event stands alone.

        If *output_queue* is provided, CRITICAL events are automatically
        pushed onto it for the alerting layer.
        """
        assert self._redis is not None

        # Gate: only correlate high-confidence LLM results.
        if not llm_result.is_valid_anomaly:
            return None
        if llm_result.confidence_score < self._min_confidence:
            logger.debug(
                "Event %s below confidence threshold (%d < %d)",
                trigger_event.event_id,
                llm_result.confidence_score,
                self._min_confidence,
            )
            return None

        # ── Find nearby events in the geo index ──────────────────────────
        contributing: list[RawEvent] = [trigger_event]
        sources_seen: set[str] = {trigger_event.source.value}

        if trigger_event.latitude is not None and trigger_event.longitude is not None:
            nearby_ids: list = await self._redis.georadius(
                "events:geo",
                trigger_event.longitude,
                trigger_event.latitude,
                CORRELATION_RADIUS_KM,
                unit="km",
            )

            for eid in nearby_ids:
                if eid == trigger_event.event_id:
                    continue

                raw_json = await self._redis.get(f"events:{eid}")
                if raw_json is None:
                    continue  # expired – stale geo entry

                neighbour = RawEvent.model_validate_json(raw_json)

                # Only count if it's from a *different* sensor source.
                if neighbour.source.value not in sources_seen:
                    sources_seen.add(neighbour.source.value)
                    contributing.append(neighbour)

        # ── Decide severity ──────────────────────────────────────────────
        if len(sources_seen) >= 2:
            severity = EventSeverity.CRITICAL
        elif llm_result.confidence_score >= 8:
            severity = EventSeverity.HIGH
        elif llm_result.confidence_score >= 5:
            severity = EventSeverity.MEDIUM
        else:
            severity = EventSeverity.LOW

        # Map the LLM classification string to our enum.
        try:
            classification = EventClassification(llm_result.event_classification)
        except ValueError:
            classification = EventClassification.UNKNOWN

        correlated = CorrelatedEvent(
            severity=severity,
            classification=classification,
            latitude=trigger_event.latitude,
            longitude=trigger_event.longitude,
            contributing_events=contributing,
            llm_analysis=llm_result,
            summary=(
                f"{severity.value} – {classification.value}: "
                f"{llm_result.approximate_origin}. "
                f"Corroborated by {len(sources_seen)} source(s): "
                f"{', '.join(sorted(sources_seen))}."
            ),
            corroborating_sources=sorted(sources_seen),
        )

        logger.info(
            "Correlation result: %s (sources=%s)",
            correlated.severity.value,
            correlated.corroborating_sources,
        )

        # Auto-push CRITICAL events to the output queue.
        if severity == EventSeverity.CRITICAL and output_queue is not None:
            await output_queue.put(correlated)

        return correlated
