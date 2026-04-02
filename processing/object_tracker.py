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
