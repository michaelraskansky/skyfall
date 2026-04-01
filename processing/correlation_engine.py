"""
Stateful Correlation Engine (DynamoDB)
=======================================

Cross-references sensor events using DynamoDB with geohash-based
spatial indexing.

Algorithm
---------
1. Every RawEvent is stored in DynamoDB keyed by geohash cell + timestamp,
   with a TTL for automatic cleanup.
2. When the LLM parser flags a high-confidence event, the engine queries
   the event's geohash cell plus its 8 neighbors for corroborating signals.
3. If two or more distinct sensor sources agree within the time window,
   the event is elevated to CRITICAL.

DynamoDB layout
---------------
- PK: ``geohash#<4-char>`` - groups events by ~20km geographic cell
- SK: ``<ISO-timestamp>#<event_id>`` - enables time-range queries
- TTL: ``expires_at`` - auto-cleanup after correlation window
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import aioboto3
from tenacity import retry, stop_after_attempt, wait_exponential

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
from processing.geohash import encode, neighbors

logger = logging.getLogger(__name__)

_NO_COORDS_GEOHASH = "none"


class CorrelationEngine:
    """DynamoDB-backed engine that cross-references sensor events."""

    def __init__(self) -> None:
        self._session: aioboto3.Session | None = None
        self._table_name = settings.dynamodb_table_name
        self._window = settings.correlation_window_sec
        self._min_confidence = settings.min_confidence_score
        self._precision = settings.geohash_precision
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
        logger.info("Correlation engine connected to DynamoDB table %s", self._table_name)

    async def close(self) -> None:
        """Shut down the DynamoDB connection."""
        if self._resource:
            await self._resource.__aexit__(None, None, None)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def ingest(self, event: RawEvent) -> None:
        """Store a raw sensor event in DynamoDB with geohash key and TTL."""
        if event.latitude is not None and event.longitude is not None:
            gh = encode(event.latitude, event.longitude, precision=self._precision)
        else:
            gh = _NO_COORDS_GEOHASH

        pk = f"geohash#{gh}"
        sk = f"{event.timestamp.isoformat()}#{event.event_id}"
        expires_at = int(time.time()) + self._window + 60

        item = {
            "pk": pk,
            "sk": sk,
            "event_id": event.event_id,
            "source": event.source.value,
            "latitude": str(event.latitude) if event.latitude is not None else None,
            "longitude": str(event.longitude) if event.longitude is not None else None,
            "description": event.description,
            "raw_payload": json.dumps(event.raw_payload),
            "timestamp": event.timestamp.isoformat(),
            "expires_at": expires_at,
        }
        item = {k: v for k, v in item.items() if v is not None}

        await self._table.put_item(Item=item)
        logger.debug("Ingested event %s from %s into cell %s", event.event_id, event.source.value, pk)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _query_cell(self, pk: str) -> list[dict]:
        """Query all items in a single geohash cell within the time window."""
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=self._window)).isoformat()
        response = await self._table.query(
            KeyConditionExpression="pk = :pk AND sk >= :cutoff",
            ExpressionAttributeValues={
                ":pk": pk,
                ":cutoff": cutoff,
            },
        )
        return response.get("Items", [])

    async def try_correlate(
        self,
        trigger_event: RawEvent,
        llm_result: LLMParsedEvent,
        output_queue: Queue[CorrelatedEvent] | None = None,
    ) -> CorrelatedEvent | None:
        """Attempt to correlate trigger_event with other recent sensor data."""
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

        contributing: list[RawEvent] = [trigger_event]
        sources_seen: set[str] = {trigger_event.source.value}

        if trigger_event.latitude is not None and trigger_event.longitude is not None:
            gh = encode(
                trigger_event.latitude, trigger_event.longitude,
                precision=self._precision,
            )
            cells_to_query = [f"geohash#{gh}"] + [
                f"geohash#{n}" for n in neighbors(gh)
            ]

            for cell_pk in cells_to_query:
                items = await self._query_cell(cell_pk)
                for item in items:
                    eid = item.get("event_id")
                    if eid == trigger_event.event_id:
                        continue
                    source_val = item.get("source", "")
                    if source_val not in sources_seen:
                        sources_seen.add(source_val)
                        lat = float(item["latitude"]) if item.get("latitude") else None
                        lon = float(item["longitude"]) if item.get("longitude") else None
                        neighbour = RawEvent(
                            event_id=eid,
                            source=EventSource(source_val),
                            latitude=lat,
                            longitude=lon,
                            description=item.get("description", ""),
                            timestamp=datetime.fromisoformat(item["timestamp"]),
                            raw_payload=json.loads(item.get("raw_payload", "{}")),
                        )
                        contributing.append(neighbour)

        if len(sources_seen) >= 2:
            severity = EventSeverity.CRITICAL
        elif llm_result.confidence_score >= 8:
            severity = EventSeverity.HIGH
        elif llm_result.confidence_score >= 5:
            severity = EventSeverity.MEDIUM
        else:
            severity = EventSeverity.LOW

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
                f"{severity.value} - {classification.value}: "
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

        if severity == EventSeverity.CRITICAL and output_queue is not None:
            await output_queue.put(correlated)

        return correlated
