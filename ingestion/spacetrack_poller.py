"""
Space-Track TIP (Tracking and Impact Prediction) Poller
========================================================

Polls Space-Track.org for TIP messages — official US Space Command
notifications of objects about to re-enter Earth's atmosphere.

Each TIP message includes:
- Predicted decay epoch (when it re-enters)
- Uncertainty window (minutes)
- Predicted lat/lon at ~10km altitude
- Whether the object is flagged as high-interest

TIP events are structured (lat/lon/epoch), so they bypass LLM triage
and go straight to the correlation engine.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    from asyncio import Queue

from config import settings
from models import EventSource, RawEvent

logger = logging.getLogger(__name__)

_LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
_TIP_URL = (
    "https://www.space-track.org/basicspacedata/query"
    "/class/tip"
    "/DECAY_EPOCH/%3Enow-3"
    "/orderby/NORAD_CAT_ID"
    "/format/json"
)


async def poll_spacetrack(event_queue: Queue[RawEvent]) -> None:
    """
    Long-running coroutine that polls Space-Track TIP endpoint
    and pushes re-entry predictions into *event_queue*.
    """
    identity = settings.spacetrack_identity
    password = settings.spacetrack_password
    interval = settings.spacetrack_poll_interval_sec

    if not identity or not password:
        logger.warning("Space-Track credentials not set – poller disabled.")
        await asyncio.Event().wait()

    logger.info("Space-Track TIP poller starting – interval=%ds", interval)

    seen_ids: set[str] = set()

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # Authenticate (cookie-based session)
                async with session.post(
                    _LOGIN_URL,
                    data={"identity": identity, "password": password},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        logger.error("Space-Track login failed: HTTP %d", resp.status)
                        await asyncio.sleep(interval)
                        continue

                # Fetch TIP messages
                async with session.get(
                    _TIP_URL,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        logger.error("Space-Track TIP query failed: HTTP %d", resp.status)
                        await asyncio.sleep(interval)
                        continue

                    tips = await resp.json()

                for tip in tips:
                    tip_id = tip.get("ID", "")
                    if tip_id in seen_ids:
                        continue
                    seen_ids.add(tip_id)

                    lat = tip.get("LAT")
                    lon = tip.get("LON")
                    norad_id = tip.get("NORAD_CAT_ID", "?")
                    decay_epoch = tip.get("DECAY_EPOCH", "?")
                    window = tip.get("WINDOW", "?")
                    high_interest = tip.get("HIGH_INTEREST", "N")

                    event = RawEvent(
                        source=EventSource.SPACETRACK,
                        latitude=float(lat) if lat else None,
                        longitude=float(lon) if lon else None,
                        raw_payload=tip,
                        description=(
                            f"TIP: NORAD {norad_id} re-entry predicted "
                            f"{decay_epoch} UTC (±{window} min) "
                            f"@ ({lat}, {lon})"
                            f"{' [HIGH INTEREST]' if high_interest == 'Y' else ''}"
                        ),
                    )
                    logger.info("Space-Track TIP: %s", event.description)
                    await event_queue.put(event)

            except Exception:
                logger.exception("Space-Track poll error")

            # Prune dedup set
            if len(seen_ids) > 10_000:
                seen_ids.clear()

            await asyncio.sleep(interval)
